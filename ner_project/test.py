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
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
from tqdm import tqdm

# === 1. 环境与路径设置 ===
PROJECT_ROOT = os.path.dirname(__file__)
MODEL_DIR = os.path.join(PROJECT_ROOT, 'model')
if MODEL_DIR not in sys.path:
    sys.path.append(MODEL_DIR)

from model.data_util import read_data, NERDataset, collate_fn
from model.bbc import BertBiLSTMCRF_BBC

# 尝试引入 Attention 模型 (兼容性处理)
try:
    from bert_bilstm_crf import BertBiLSTMCRF
except ImportError:
    pass 

def evaluate(model, loader, idx2tag, device):
    model.eval()
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Testing"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].cpu().numpy()
            
            # === 获取预测结果 ===
            if hasattr(model, 'decode'):
                 preds = model.decode(input_ids, attention_mask)
            elif hasattr(model, 'crf'):
                 if hasattr(model, 'bert'):
                     bert_out = model.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
                     lstm_out, _ = model.bilstm(bert_out)
                     
                     if hasattr(model, 'attention_type') and model.attention_type != 'none':
                        if model.attention_type == "ufo":
                            attn_out = model.self_attn(lstm_out, lstm_out, lstm_out)
                        else:
                            attn_out = model.self_attn(lstm_out)
                        
                        attn_out = attn_out + lstm_out
                        lstm_out = model.norm(attn_out)

                     emissions = model.classifier(model.dropout(lstm_out))
                     preds = model.crf.decode(emissions, mask=attention_mask.bool())
            else:
                 emissions = model(input_ids, attention_mask)
                 preds = torch.argmax(emissions, dim=-1).cpu().numpy()

            # === 标签对齐 ===
            for i, p_seq in enumerate(preds):
                l_seq = labels[i]
                true_seq = [idx2tag[l] for l, p in zip(l_seq, p_seq) if l != -100]
                pred_seq = [idx2tag[p] for l, p in zip(l_seq, p_seq) if l != -100]
                
                all_preds.append(pred_seq)
                all_labels.append(true_seq)
    
    # === 核心修改：转换为百分比格式 ===
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)

    print("\n" + "="*30 + " 最终评估报告 " + "="*30)
    # 这里乘以 100 并保留 2 位小数
    print(f"准确率 (Precision): {precision * 100:.2f}%")
    print(f"召回率 (Recall):    {recall * 100:.2f}%")
    print(f"F1 分数 (F1 Score): {f1 * 100:.2f}%")
    print("-" * 60)
    
    # digits=4 让表格显示更精确，虽然表格里还是小数，但更详细
    print(classification_report(all_labels, all_preds, digits=4))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--attention", type=str, default="none", choices=["none", "ufo"])
    parser.add_argument("--model_path", type=str, required=True, help="模型权重路径 (.pt)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # === 自动路径修正 ===
    data_dir = os.path.join(PROJECT_ROOT, 'data')
    if not os.path.exists(data_dir): data_dir = PROJECT_ROOT 
    
    test_file = os.path.join(data_dir, 'test.txt')
    train_file = os.path.join(data_dir, 'train.txt')

    if not os.path.exists(train_file):
        print(f"❌ 错误: 找不到 {train_file}")
        sys.exit(1)

    # === 自动寻找 BERT ===
    possible_paths = [
        'E:/PythonProject/ner_project/bert-base',
    ]
    bert_dir = 'E:\PythonProject/ner_project/bert-base'
    for p in possible_paths:
        if os.path.exists(p) and os.path.exists(os.path.join(p, 'vocab.txt')):
            bert_dir = p
            break
    print(f"Loading BERT from: {bert_dir}")

    # === 准备数据 ===
    _, train_tags = read_data(train_file)
    label_set = sorted(set(t for s in train_tags for t in s))
    if 'O' not in label_set: label_set.append('O')
    tag2idx = {t: i for i, t in enumerate(label_set)}
    idx2tag = {i: t for t, i in tag2idx.items()}

    tokenizer = BertTokenizerFast.from_pretrained(bert_dir, do_lower_case=True)
    test_texts, test_tags = read_data(test_file)
    
    test_ds = NERDataset(test_texts, test_tags, tokenizer, tag2idx, max_len=512, stride=128)
    test_loader = DataLoader(test_ds, batch_size=8, shuffle=False, collate_fn=collate_fn)

    # === 加载模型 ===
    print(f"Loading model... [Mode: {args.attention}]")
    if args.attention == "none":
        model = BertBiLSTMCRF_BBC(bert_dir, 256, len(tag2idx)).to(device)
    else:
        from model.bert_bilstm_crf import BertBiLSTMCRF
        model = BertBiLSTMCRF(bert_dir, 256, len(tag2idx)).to(device)
        
    print(f"Loading weights from: {args.model_path}")
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    
    evaluate(model, test_loader, idx2tag, device)