import sys
import os

# 自动获取当前脚本所在的绝对路径，并将其设为搜索根目录
cur_path = os.path.dirname(os.path.abspath(__file__))
if cur_path not in sys.path:
    sys.path.insert(0, cur_path)
import argparse
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(PROJECT_ROOT, 'model')
if MODEL_DIR not in sys.path:
    sys.path.append(MODEL_DIR)

try:
    from model.data_util import read_data, NERDataset, collate_fn
    from model.bbc import BertBiLSTMCRF_BBC
    from model.bert_bilstm_crf import BertBiLSTMCRF
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    sys.exit(1)

def predict():
    parser = argparse.ArgumentParser()
    parser.add_argument("--attention", type=str, default="ufo", choices=["none", "ufo"])
    parser.add_argument("--model_path", type=str, required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_file = os.path.join(PROJECT_ROOT, 'data/train.txt')
    if not os.path.exists(train_file): 
        train_file = os.path.join(PROJECT_ROOT, 'train.txt')
    
    train_sents, train_tags = read_data(train_file)
    label_set = sorted(set(t for s in train_tags for t in s))
    if 'O' not in label_set: label_set.append('O')
    tag2idx = {t: i for i, t in enumerate(label_set)}
    idx2tag = {i: t for t, i in tag2idx.items()}

    bert_dir = 'E:/PythonProject/ner_project/bert-base'
    tokenizer = BertTokenizerFast.from_pretrained(bert_dir, do_lower_case=True)
    
    if args.attention == "none":
        model = BertBiLSTMCRF_BBC(bert_dir, 256, len(tag2idx)).to(device)
    else:
        model = BertBiLSTMCRF(bert_dir, 256, len(tag2idx)).to(device)
    
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    print("\n" + "="*50)
    print("🚀 交互式识别系统已就绪")
    print("📏 支持长文本处理（自动分段）")
    print("="*50)

    while True:
        original_text = input("\n请输入文本 (输入 'q' 退出): ")
        if original_text.lower() in ['q', 'quit']: break
        if not original_text.strip(): continue

        text_no_space = original_text.replace(' ', '')
        
        # 写入临时文件
        temp_file = os.path.join(PROJECT_ROOT, 'temp_input.txt')
        try:
            with open(temp_file, 'w', encoding='utf-8') as f:
                for char in text_no_space:
                    f.write(f"{char} O\n")
                f.write("\n")
            
            test_sents, test_tags = read_data(temp_file)
        finally:
            if os.path.exists(temp_file):
                os.remove(temp_file)
        
        # 创建数据集（保持 stride=128）
        test_ds = NERDataset(test_sents, test_tags, tokenizer, tag2idx, max_len=512, stride=128)
        test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, collate_fn=collate_fn)

        # 🔴 方案1：只取每个chunk的非重叠部分
        all_tags = []
        processed_len = 0
        STRIDE = 128  # 和 NERDataset 的 stride 保持一致

        model.eval()
        with torch.no_grad():
            for chunk_idx, batch in enumerate(test_loader):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].cpu().numpy()
                
                preds = model(input_ids=input_ids, attention_mask=attention_mask)
                
                # 提取当前chunk的预测
                chunk_tags = []
                for l, p in zip(labels[0], preds[0]):
                    if l != -100:
                        chunk_tags.append(idx2tag[p])
                
                # 🔴 关键：只取非重叠部分
                if chunk_idx == 0:
                    # 第一个chunk：全部保留
                    all_tags.extend(chunk_tags)
                    processed_len = len(chunk_tags)
                else:
                    # 后续chunks：跳过重叠的stride部分
                    if len(chunk_tags) > STRIDE:
                        # 跳过前STRIDE个（重叠部分）
                        all_tags.extend(chunk_tags[STRIDE:])
                        processed_len += len(chunk_tags) - STRIDE
                    # 如果chunk很短（<STRIDE），说明是最后一段，全部保留
                    elif processed_len < len(text_no_space):
                        all_tags.extend(chunk_tags)
                        processed_len += len(chunk_tags)
                
                # 如果已经处理完所有字符，停止
                if processed_len >= len(text_no_space):
                    break
        
        # 确保长度匹配（处理可能的误差）
        if len(all_tags) > len(text_no_space):
            all_tags = all_tags[:len(text_no_space)]
        elif len(all_tags) < len(text_no_space):
            # 补齐不足的部分（理论上不应该发生）
            all_tags.extend(['O'] * (len(text_no_space) - len(all_tags)))
        
        # 插回空格
        full_chars = []
        full_tags = []
        no_space_idx = 0
        
        for orig_char in original_text:
            if orig_char == ' ':
                # 空格继承前一个字符的标签
                if full_tags and full_tags[-1].startswith('I-'):
                    full_chars.append(' ')
                    full_tags.append(full_tags[-1])
                else:
                    full_chars.append(' ')
                    full_tags.append('O')
            else:
                if no_space_idx < len(all_tags):
                    full_chars.append(text_no_space[no_space_idx])
                    full_tags.append(all_tags[no_space_idx])
                    no_space_idx += 1
                else:
                    full_chars.append(orig_char)
                    full_tags.append('O')
        
        # 提取实体
        print(f"\n[识别结果] (文本长度: {len(text_no_space)} 字符, 处理了 {len(test_loader)} 个chunks):")
        curr_entity = ""
        curr_type = ""
        
        for char, tag in zip(full_chars, full_tags):
            if tag.startswith("B-"):
                if curr_entity:
                    print(f"  <{curr_type}>: {curr_entity}")
                curr_entity = char
                curr_type = tag[2:]
            elif tag.startswith("I-") and curr_type == tag[2:]:
                curr_entity += char
            else:
                if curr_entity:
                    print(f"  <{curr_type}>: {curr_entity}")
                curr_entity = ""
                curr_type = ""
        
        if curr_entity:
            print(f"  <{curr_type}>: {curr_entity}")


if __name__ == "__main__":
    predict()