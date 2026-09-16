import os
import sys
import argparse
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
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

from data_util import read_data, NERDataset, collate_fn

# Import the ablation model.
try:
    from ufoonly import BertBiLSTMCRF_UFOOnly
except ImportError:
    print("❌ 错误：无法导入 BertBiLSTMCRF_UFOOnly。请检查 model/ufoonly.py")
    sys.exit(1)

# === 3. Training function ===
def train_epoch(model, optimizer, loader, device, o_tag_id):
    """Train for one epoch"""
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

# === 4. Evaluation function ===
def eval_model(model, loader, idx2tag, device, show_report=False):
    """Evaluation model"""
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].cpu().numpy()
            
            # Directly call the model
            batch_preds = model(input_ids, attention_mask)

            # Align labels
            for i, preds in enumerate(batch_preds):
                true_labels = [idx2tag[l] for l, p in zip(labels[i], preds) if l != -100]
                pred_labels = [idx2tag[p] for l, p in zip(labels[i], preds) if l != -100]
                all_preds.append(pred_labels)
                all_labels.append(true_labels)

    # Calculation metrics
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    
    # Show detailed report
    if show_report:
        print("\n" + "="*30 + " 最终评估报告 " + "="*30)
        print(f"准确率 (Precision): {precision * 100:.2f}%")
        print(f"召回率 (Recall):    {recall * 100:.2f}%")
        print(f"F1 分数 (F1 Score): {f1 * 100:.2f}%")
        print("-" * 70)
        print(classification_report(all_labels, all_preds, digits=4))
    
    return f1, precision, recall

# === 5. Main program ===
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=7)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    # --- Automatically locate the BERT path. ---
    possible_paths = [
        '/root/autodl-tmp/ner_project/bert-base-chinese', 
        '/root/autodl-tmp/ner_project/bert_base',
        'bert-base-chinese',
        'bert_base'
    ]
    bert_dir = 'E:/PythonProject/ner_project/bert-base'
    for p in possible_paths:
        if os.path.exists(p) and os.path.exists(os.path.join(p, 'vocab.txt')):
            bert_dir = p
            break
    print(f"Loading BERT from: {bert_dir}")

    # --- Data Preparation ---
    data_dir = os.path.join(PROJECT_ROOT, 'data')
    if not os.path.exists(data_dir): 
        data_dir = PROJECT_ROOT
    
    train_file = os.path.join(data_dir, 'train.txt')
    dev_file = os.path.join(data_dir, 'dev.txt')
    test_file = os.path.join(data_dir, 'test.txt')

    train_texts, train_tags = read_data(train_file)
    dev_texts, dev_tags = read_data(dev_file)
    test_texts, test_tags = read_data(test_file)

    label_set = sorted(set(tag for seq in train_tags for tag in seq))
    if 'O' not in label_set: 
        label_set.append('O')
    tag2idx = {tag: i for i, tag in enumerate(label_set)}
    idx2tag = {i: tag for tag, i in tag2idx.items()}

    tokenizer = BertTokenizerFast.from_pretrained(bert_dir, do_lower_case=True)
    
    train_ds = NERDataset(train_texts, train_tags, tokenizer, tag2idx, max_len=512, stride=128)
    dev_ds = NERDataset(dev_texts, dev_tags, tokenizer, tag2idx, max_len=512, stride=128)
    test_ds = NERDataset(test_texts, test_tags, tokenizer, tag2idx, max_len=512, stride=128)
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    dev_loader = DataLoader(dev_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    # --- Initialize the model ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"\n{'='*70}")
    print(f"设备: {device}")
    print(f"数据集: Train={len(train_texts)} | Dev={len(dev_texts)} | Test={len(test_texts)}")
    print(f"标签数: {len(tag2idx)}")
    print(f"{'='*70}\n")

    model = BertBiLSTMCRF_UFOOnly(
        bert_model_name=bert_dir,
        lstm_hidden=256,
        num_tags=len(tag2idx),
        lstm_layers=1,
        dropout=0.2
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # --- Training loop ---
    best_f1 = 0.0
    patience_counter = 0
    best_path = "权重/bert_bilstm_ufoonly_best.pt"
    o_tag_id = tag2idx['O']

    print(f"{'='*70}")
    print("开始训练 [消融实验: UFO + Residual, 无LayerNorm]")
    print(f"{'='*70}\n")
    
    for epoch in range(1, args.epochs + 1):
        print(f"\n{'='*70}")
        print(f"Epoch {epoch}/{args.epochs}")
        print(f"{'='*70}")
        
        # train
        train_loss = train_epoch(model, optimizer, train_loader, device, o_tag_id)
        print(f"Train Loss: {train_loss:.4f}")

        # dev
        print("\n>>> 验证集评估 <<<")
        val_f1, val_precision, val_recall = eval_model(model, dev_loader, idx2tag, device, show_report=False)
        print(f"Validation F1: {val_f1 * 100:.2f}% | Precision: {val_precision * 100:.2f}% | Recall: {val_recall * 100:.2f}%")

        # Save the best model
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), best_path)
            print(f"✅ 新的最佳模型 F1: {val_f1 * 100:.2f}% | 已保存至 {best_path}")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"⚠️ 无改善 Patience: {patience_counter}/{args.patience}")
            if patience_counter >= args.patience:
                print("⏹️ Early stopping!")
                break

    # --- Load the best model and test. ---
    print(f"\n{'='*70}")
    print("训练完成！加载最佳模型进行测试...")
    print(f"{'='*70}\n")
    
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path))
        model.to(device)
        model.eval()
        print(f"✅ 已加载最佳模型: {best_path}")
        print(f"✅ 最佳验证 F1: {best_f1 * 100:.2f}%\n")
    
    # Final evaluation on the test set
    print(f"{'='*70}")
    print("测试集最终评估")
    print(f"{'='*70}")
    
    test_f1, test_precision, test_recall = eval_model(model, test_loader, idx2tag, device, show_report=True)
    
    # Final Summary
    print(f"\n{'='*70}")
    print("🎯 实验结果总结")
    print(f"{'='*70}")
    print(f"模型架构: BERT + BiLSTM + UFO + Residual + CRF (无LayerNorm)")
    print(f"最佳验证 F1: {best_f1 * 100:.2f}%")
    print(f"最终测试 F1: {test_f1 * 100:.2f}%")
    print(f"测试 Precision: {test_precision * 100:.2f}%")
    print(f"测试 Recall: {test_recall * 100:.2f}%")
    print(f"模型保存至: {best_path}")
    print(f"{'='*70}\n")
