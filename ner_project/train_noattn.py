import sys
import os

# Automatically obtain the absolute path of the current script and set it as the search root directory.
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

# === 1. Fix the random seed ===
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# === 2. Path Settings ===
PROJECT_ROOT = os.path.dirname(__file__)
MODEL_DIR = os.path.join(PROJECT_ROOT, 'model')
if MODEL_DIR not in sys.path:
    sys.path.append(MODEL_DIR)

# Import custom modules
from model.data_util import read_data, NERDataset, collate_fn
from model.bbc import BertBiLSTMCRF_BBC

# === 3. Training function ===
def train_epoch(model, optimizer, loader, device, o_tag_id):
    model.train()
    total_loss = 0.0
    
    for batch in tqdm(loader, desc="Training"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        active_labels = torch.where(
            labels == -100, 
            torch.tensor(o_tag_id, device=device), 
            labels
        )

        inputs = {
            'input_ids':      input_ids,
            'attention_mask': attention_mask,
            'labels':         active_labels 
        }
        
        loss = model(**inputs)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        
    return total_loss / len(loader)

# === 4. Validation function ===
def eval_model(model, loader, idx2tag, device):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            bert_out = model.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
            lstm_out, _ = model.bilstm(bert_out)
            emissions = model.classifier(model.dropout(lstm_out))
            
            batch_preds = model.crf.decode(emissions, mask=attention_mask.bool())

            labels = labels.cpu().numpy()
            
            for i, preds in enumerate(batch_preds):
                true_labels = [idx2tag[l] for l, p in zip(labels[i], preds) if l != -100]
                pred_labels = [idx2tag[p] for l, p in zip(labels[i], preds) if l != -100]
                
                all_preds.append(pred_labels)
                all_labels.append(true_labels)

    f1 = f1_score(all_labels, all_preds)
    # print(classification_report(all_labels, all_preds)) # 调试时可取消注释
    return f1

# === 5. Main program ===
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=7)
    parser.add_argument("--batch_size", type=int, default=8) 
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--patience", type=int, default=3)
    args = parser.parse_args()

    set_seed(42)

    # --- Data path ---
    data_dir = os.path.join(PROJECT_ROOT, 'data')
    if not os.path.exists(data_dir): 
        data_dir = PROJECT_ROOT
    
    train_file = os.path.join(data_dir, 'train.txt')
    dev_file = os.path.join(data_dir, 'dev.txt')

    bert_dir = 'E:/PythonProject/ner_project/bert-base'
    
    if not os.path.exists(bert_dir):
        print(f"Warning: {bert_dir} not found, trying local relative path...")
        bert_dir = 'ner_project/bert-base'

    print(f"Loading BERT from: {bert_dir}")

    train_texts, train_tags = read_data(train_file)
    dev_texts, dev_tags = read_data(dev_file)

    label_set = sorted(set(tag for seq in train_tags for tag in seq))
    if 'O' not in label_set: label_set.append('O')
    tag2idx = {tag: i for i, tag in enumerate(label_set)}
    idx2tag = {i: tag for tag, i in tag2idx.items()}

    print(f"Labels: {tag2idx}")

    try:
        tokenizer = BertTokenizerFast.from_pretrained(bert_dir, do_lower_case=True)
    except Exception as e:
        print(f"\n❌ Error loading tokenizer from '{bert_dir}'.")
        print("Please check if the directory exists and contains vocab.txt")
        sys.exit(1)

    train_ds = NERDataset(train_texts, train_tags, tokenizer, tag2idx, max_len=512, stride=128)
    dev_ds = NERDataset(dev_texts, dev_tags, tokenizer, tag2idx, max_len=512, stride=128)
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    dev_loader = DataLoader(dev_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    # --- Initialize the model ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = BertBiLSTMCRF_BBC(
        bert_model_name=bert_dir,
        lstm_hidden=256,
        num_tags=len(tag2idx),
        lstm_layers=1
    ).to(device)

    # Using the AdamW optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # --- Training loop ---
    best_f1 = 0.0
    patience_counter = 0
    best_path = None
    
    o_tag_id = tag2idx['O']

    print(f"\nStart training [No Attention]...")
    print(f"Batch Size: {args.batch_size}, Max Len: 512")

    for epoch in range(1, args.epochs + 1):
        print(f"\n=== Epoch {epoch} ===")
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
