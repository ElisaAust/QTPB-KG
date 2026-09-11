import sys
import os

# 自动获取当前脚本所在的绝对路径，并将其设为搜索根目录
cur_path = os.path.dirname(os.path.abspath(__file__))
if cur_path not in sys.path:
    sys.path.insert(0, cur_path)
import argparse
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast
from seqeval.metrics import classification_report, f1_score
from tqdm import tqdm

# === 1. 固定随机种子 ===
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# === 2. 路径设置 ===
PROJECT_ROOT = os.path.dirname(__file__)
MODEL_DIR = os.path.join(PROJECT_ROOT, 'model')
if MODEL_DIR not in sys.path:
    sys.path.append(MODEL_DIR)

# 导入自定义模块
from model.data_util import read_data, NERDataset, collate_fn
from model.bbc import BertBiLSTMCRF_BBC

# === 3. 训练函数 (已修复 CUDA -100 问题) ===
def train_epoch(model, optimizer, loader, device, o_tag_id):
    """
    o_tag_id: 'O' 标签对应的整数索引，用于填充 -100 的位置
    """
    model.train()
    total_loss = 0.0
    
    for batch in tqdm(loader, desc="Training"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        # 【关键修复】CRF 不支持 -100，必须替换为合法的 tag ID (通常是 'O')
        # 我们使用 torch.where 将所有的 -100 替换为 o_tag_id
        active_labels = torch.where(
            labels == -100, 
            torch.tensor(o_tag_id, device=device), 
            labels
        )

        # 构造输入
        inputs = {
            'input_ids':      input_ids,
            'attention_mask': attention_mask,
            'labels':         active_labels # 传入清洗后的标签
        }
        
        # 前向传播 (返回 loss)
        loss = model(**inputs)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        
    return total_loss / len(loader)

# === 4. 验证函数 ===
def eval_model(model, loader, idx2tag, device):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # 获取模型输出的 emissions (CRF之前的分数)
            # 因为 bbc.py 可能没有封装 decode，我们手动操作：
            bert_out = model.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
            lstm_out, _ = model.bilstm(bert_out)
            emissions = model.classifier(model.dropout(lstm_out))
            
            # 使用 CRF 解码最佳路径
            batch_preds = model.crf.decode(emissions, mask=attention_mask.bool())

            labels = labels.cpu().numpy()
            
            # 对齐预测和标签 (忽略 -100)
            for i, preds in enumerate(batch_preds):
                # 真实的 label 原本包含 -100，需要过滤
                true_labels = [idx2tag[l] for l, p in zip(labels[i], preds) if l != -100]
                pred_labels = [idx2tag[p] for l, p in zip(labels[i], preds) if l != -100]
                
                all_preds.append(pred_labels)
                all_labels.append(true_labels)

    f1 = f1_score(all_labels, all_preds)
    # print(classification_report(all_labels, all_preds)) # 调试时可取消注释
    return f1

# === 5. 主程序 ===
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=7)
    # 显存优化：512 长度建议 batch_size=4 或 8
    parser.add_argument("--batch_size", type=int, default=8) 
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--patience", type=int, default=3)
    args = parser.parse_args()

    set_seed(42)

    # --- 数据路径 ---
    data_dir = os.path.join(PROJECT_ROOT, 'data')
    if not os.path.exists(data_dir): 
        data_dir = PROJECT_ROOT
    
    train_file = os.path.join(data_dir, 'train.txt')
    dev_file = os.path.join(data_dir, 'dev.txt')

    # --- 模型路径 (请根据实际情况修改) ---
    # 建议使用绝对路径，避免文件名混淆
    bert_dir = 'E:/PythonProject/ner_project/bert-base'
    # 如果上面路径不对，尝试: bert_dir = 'bert-base-chinese' 
    
    if not os.path.exists(bert_dir):
        # 如果绝对路径不存在，尝试相对路径
        print(f"Warning: {bert_dir} not found, trying local relative path...")
        bert_dir = 'ner_project/bert-base'

    print(f"Loading BERT from: {bert_dir}")

    # --- 读取数据 & 构建标签 ---
    train_texts, train_tags = read_data(train_file)
    dev_texts, dev_tags = read_data(dev_file)

    label_set = sorted(set(tag for seq in train_tags for tag in seq))
    if 'O' not in label_set: label_set.append('O')
    tag2idx = {tag: i for i, tag in enumerate(label_set)}
    idx2tag = {i: tag for tag, i in tag2idx.items()}

    print(f"Labels: {tag2idx}")

    # --- Tokenizer & Dataset ---
    try:
        tokenizer = BertTokenizerFast.from_pretrained(bert_dir, do_lower_case=True)
    except Exception as e:
        print(f"\n❌ Error loading tokenizer from '{bert_dir}'.")
        print("Please check if the directory exists and contains vocab.txt")
        sys.exit(1)

    # max_len=512, stride=128 用于处理长文本
    train_ds = NERDataset(train_texts, train_tags, tokenizer, tag2idx, max_len=512, stride=128)
    dev_ds = NERDataset(dev_texts, dev_tags, tokenizer, tag2idx, max_len=512, stride=128)
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    dev_loader = DataLoader(dev_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    # --- 初始化模型 ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = BertBiLSTMCRF_BBC(
        bert_model_name=bert_dir,
        lstm_hidden=256,
        num_tags=len(tag2idx),
        lstm_layers=1
    ).to(device)

    # 使用 AdamW 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # --- 训练循环 ---
    best_f1 = 0.0
    patience_counter = 0
    best_path = None
    
    # 获取 'O' 的 ID，用于填充 -100
    o_tag_id = tag2idx['O']

    print(f"\nStart training [No Attention]...")
    print(f"Batch Size: {args.batch_size}, Max Len: 512")

    for epoch in range(1, args.epochs + 1):
        print(f"\n=== Epoch {epoch} ===")
        # 传入 o_tag_id 修复 CUDA 报错
        train_loss = train_epoch(model, optimizer, train_loader, device, o_tag_id)
        print(f"Train Loss: {train_loss:.4f}")

        val_f1 = eval_model(model, dev_loader, idx2tag, device)
        print(f"Validation F1: {val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_path = "best_model_noattn.pt"
            torch.save(model.state_dict(), best_path)
            print(f"✅ New best model saved: {val_f1:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"⚠️ No improvement ({patience_counter}/{args.patience})")
            if patience_counter >= args.patience:
                print("Early stopping!")
                break