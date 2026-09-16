# model/data_util.py

import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

def read_data(file_path):
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

            if len(parts) < 2:
                continue
            token = parts[0]
            tag = parts[-1] 
            sent.append(token)
            lab.append(tag)
    if sent:
        sentences.append(sent)
        labels.append(lab)
    return sentences, labels

def train_val_split(sentences, labels, val_size=0.1, random_state=42):
    return train_test_split(
        sentences, labels,
        test_size=val_size,
        random_state=random_state
    )

class NERDataset(Dataset):
    def __init__(self, sentences, labels, tokenizer, tag2idx, max_len=512, stride=128):
        self.tag2idx = tag2idx
        self.features = []

        for sent_tokens, sent_labels in zip(sentences, labels):
            tokenized_inputs = tokenizer(
                sent_tokens,
                is_split_into_words=True,
                max_length=max_len,
                padding="max_length",
                truncation=True,
                stride=stride,  
                return_overflowing_tokens=True, 
                return_offsets_mapping=True,
                return_tensors="pt"
            )
    
            num_chunks = len(tokenized_inputs['input_ids'])
            
            for i in range(num_chunks):
                input_ids = tokenized_inputs['input_ids'][i]
                attention_mask = tokenized_inputs['attention_mask'][i]
                word_ids = tokenized_inputs.word_ids(batch_index=i)
                
                label_ids = []
                previous_word_idx = None
                
                for word_idx in word_ids:
                    if word_idx is None:
                        label_ids.append(-100)
                    elif word_idx != previous_word_idx:
                        label_ids.append(tag2idx[sent_labels[word_idx]])
                    else:
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
    input_ids = torch.stack([x['input_ids'] for x in batch])
    attention_mask = torch.stack([x['attention_mask'] for x in batch])
    labels = torch.stack([x['labels'] for x in batch])

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels
    }
