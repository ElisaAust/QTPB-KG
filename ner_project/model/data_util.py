# model/data_util.py

import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

def read_data(file_path):
    """
    读取 CoNLL 格式数据：每行 “token[空格或制表]tag”，句子间以空行分隔。
    返回：
      sentences: List[List[str]]
      labels:    List[List[str]]
    """
    sentences, labels = [], []
    sent, lab = [], []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                if sent:
                    sentences.append(sent)
                    labels.append(lab)
                    sent, lab = [], []
                continue
            parts = line.split()
            # 兼容有些行可能格式不对的情况，取前两个作为 token 和 tag
            if len(parts) < 2:
                continue
            token = parts[0]
            tag = parts[-1] # 通常 tag 在最后
            sent.append(token)
            lab.append(tag)
    if sent:
        sentences.append(sent)
        labels.append(lab)
    return sentences, labels

def train_val_split(sentences, labels, val_size=0.1, random_state=42):
    """
    划分训练/验证集
    """
    return train_test_split(
        sentences, labels,
        test_size=val_size,
        random_state=random_state
    )

class NERDataset(Dataset):
    def __init__(self, sentences, labels, tokenizer, tag2idx, max_len=512, stride=128):
        """
        sentences: List[List[str]] (words)
        labels: List[List[str]] (tags)
        max_len: BERT最大长度，建议512
        stride: 滑动窗口步长，用于处理长文本时的重叠部分
        """
        self.tag2idx = tag2idx
        self.features = []

        # 预处理：将所有句子tokenize，如果超长则自动切分
        for sent_tokens, sent_labels in zip(sentences, labels):
            # 使用 tokenizer 的 return_overflowing_tokens 功能处理长文本
            tokenized_inputs = tokenizer(
                sent_tokens,
                is_split_into_words=True,
                max_length=max_len,
                padding="max_length",
                truncation=True,
                stride=stride,  # 设置重叠步长，防止实体被切断
                return_overflowing_tokens=True, # 允许返回多个片段
                return_offsets_mapping=True,
                return_tensors="pt"
            )

            # tokenized_inputs 可能包含多个片段（如果是长文本）
            # overflow_to_sample_mapping 告诉我们每个片段属于哪个原始句子（这里我们在循环里处理，其实不需要这个映射）
            
            num_chunks = len(tokenized_inputs['input_ids'])
            
            for i in range(num_chunks):
                input_ids = tokenized_inputs['input_ids'][i]
                attention_mask = tokenized_inputs['attention_mask'][i]
                word_ids = tokenized_inputs.word_ids(batch_index=i)
                
                label_ids = []
                previous_word_idx = None
                
                for word_idx in word_ids:
                    # 特殊token (CLS, SEP, PAD) 设为 -100
                    if word_idx is None:
                        label_ids.append(-100)
                    elif word_idx != previous_word_idx:
                        # 新的单词，取对应标签
                        label_ids.append(tag2idx[sent_labels[word_idx]])
                    else:
                        # 同一个单词的后续subword（例如 "apple" -> "ap", "##ple"）
                        # 策略：不计算loss (-100) 或者 同标签 (I-xxx)
                        # 这里我们只对单词的第一个subword计算loss
                        label_ids.append(-100) 
                    previous_word_idx = word_idx
                
                self.features.append({
                    'input_ids': input_ids,
                    'attention_mask': attention_mask,
                    'labels': torch.tensor(label_ids, dtype=torch.long)
                })

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx]

def collate_fn(batch):
    """
    将 list[dict] -> batch dict(tensor)
    """
    input_ids = torch.stack([x['input_ids'] for x in batch])
    attention_mask = torch.stack([x['attention_mask'] for x in batch])
    labels = torch.stack([x['labels'] for x in batch])

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels
    }