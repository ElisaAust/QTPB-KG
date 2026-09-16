"""
Improved Error Comparison Script
Supports the correct display of mixed Chinese and English text.
"""

import os
import sys
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast
from tqdm import tqdm
from collections import defaultdict

# Path Settings
PROJECT_ROOT = os.path.dirname(__file__)
MODEL_DIR = os.path.join(PROJECT_ROOT, 'model')
if MODEL_DIR not in sys.path:
    sys.path.append(MODEL_DIR)

from model.data_util import read_data, NERDataset, collate_fn
from model.bbc import BertBiLSTMCRF_BBC
from model.bert_bilstm_crf import BertBiLSTMCRF


def reconstruct_text(tokens, remove_special=True):
    if remove_special:
        # Remove special tokens
        tokens = [t for t in tokens if t not in ['[CLS]', '[SEP]', '[PAD]', '[UNK]']]
    
    reconstructed = []
    for i, token in enumerate(tokens):
        if token.startswith(''):
             This is a subword; concatenate directly (without adding spaces).
            reconstructed.append(token[2:])
        else:
            if i > 0 and not tokens[i-1].startswith(''):
                 If the previous token does not start with and the current token does not start with
                 Determine whether a space needs to be added.
                prev_token = reconstructed[-1] if reconstructed else ''
                current_token = token
                
                 Chinese characters do not require spaces, whereas English words do.
                if prev_token and not _is_chinese_char(prev_token[-1]) and not _is_chinese_char(current_token[0]):
                    reconstructed.append(' ')
            
            reconstructed.append(token)
    
    return ''.join(reconstructed)


def _is_chinese_char(char):
    """Determine whether it is a Chinese character."""
    cp = ord(char)
    return (
        (cp >= 0x4E00 and cp <= 0x9FFF) or
        (cp >= 0x3400 and cp <= 0x4DBF) or
        (cp >= 0x20000 and cp <= 0x2A6DF) or
        (cp >= 0x2A700 and cp <= 0x2B73F) or
        (cp >= 0x2B740 and cp <= 0x2B81F) or
        (cp >= 0x2B820 and cp <= 0x2CEAF) or
        (cp >= 0xF900 and cp <= 0xFAFF) or
        (cp >= 0x2F800 and cp <= 0x2FA1F)
    )


def get_predictions(model, input_ids, attention_mask):
    """Automatically select the prediction method based on the model type."""
    model.eval()
    with torch.no_grad():
        if hasattr(model, 'decode'):
            return model.decode(input_ids, attention_mask)
        else:
            return model(input_ids, attention_mask)


def extract_entities_from_tags(tokens, tags):
    """
    Extract entities from BIO tags.
    
    Returns: List[Dict], where each dict contains {'type', 'start', 'end', 'text', 'tokens'}
    """
    entities = []
    current_entity = None
    
    for i, (token, tag) in enumerate(zip(tokens, tags)):
        if tag.startswith('B-'):
             Save the previous entity
            if current_entity:
                entities.append(current_entity)
            
             Start a new entity
            entity_type = tag[2:]
            current_entity = {
                'type': entity_type,
                'start': i,
                'end': i,
                'tokens': [token],
                'text': token if not token.startswith('') else token[2:]
            }
        
        elif tag.startswith('I-'):
            if current_entity and tag[2:] == current_entity['type']:
                 Continue with the current entity
                current_entity['end'] = i
                current_entity['tokens'].append(token)
                 Intelligent Stitching
                if token.startswith(''):
                    current_entity['text'] += token[2:]
                else:
                     Determine whether a space is needed.
                    if current_entity['text'] and not _is_chinese_char(current_entity['text'][-1]) and not _is_chinese_char(token[0]):
                        current_entity['text'] += ' ' + token
                    else:
                        current_entity['text'] += token
            else:
                 Label mismatch; saving old entity and starting a new one.
                if current_entity:
                    entities.append(current_entity)
                
                entity_type = tag[2:]
                current_entity = {
                    'type': entity_type,
                    'start': i,
                    'end': i,
                    'tokens': [token],
                    'text': token if not token.startswith('') else token[2:]
                }
        
        else:   O-tag
            if current_entity:
                entities.append(current_entity)
                current_entity = None
    
     Save the last entity
    if current_entity:
        entities.append(current_entity)
    
    return entities


def compare_predictions(ufo_preds, base_preds, true_labels, tokens_list):
    """
    Compare the prediction results of the two models
    Returns:
    results: A dictionary containing different types of cases
    stats: Statistical information
    """
    results = {
        'ufo_correct_base_wrong': [],   UFO is correct; baseline is incorrect.
        'base_correct_ufo_wrong': [],   Baseline is correct; UFO is incorrect.
        'both_wrong': [],                All wrong
        'both_correct': []              # All true
    }
    
    stats = defaultdict(lambda: {
        'total': 0,
        'ufo_correct': 0,
        'base_correct': 0,
        'both_correct': 0,
        'ufo_only_correct': 0,
        'base_only_correct': 0
    })
    
    for ufo_tags, base_tags, true_tags, tokens in zip(ufo_preds, base_preds, true_labels, tokens_list):
         Extract Entities
        true_entities = extract_entities_from_tags(tokens, true_tags)
        ufo_entities = extract_entities_from_tags(tokens, ufo_tags)
        base_entities = extract_entities_from_tags(tokens, base_tags)
        
         Convert to a set for easier comparison
        true_set = {(e['type'], e['start'], e['end']) for e in true_entities}
        ufo_set = {(e['type'], e['start'], e['end']) for e in ufo_entities}
        base_set = {(e['type'], e['start'], e['end']) for e in base_entities}
        
         Analyze each real-world entity
        for entity in true_entities:
            key = (entity['type'], entity['start'], entity['end'])
            entity_type = entity['type']
            
            ufo_correct = key in ufo_set
            base_correct = key in base_set
            
            stats[entity_type]['total'] += 1
            
            case = {
                'entity': entity,
                'tokens': tokens,
                'true_tags': true_tags,
                'ufo_tags': ufo_tags,
                'base_tags': base_tags,
                'sentence': reconstruct_text(tokens)
            }
            
            if ufo_correct and base_correct:
                stats[entity_type]['both_correct'] += 1
                stats[entity_type]['ufo_correct'] += 1
                stats[entity_type]['base_correct'] += 1
                results['both_correct'].append(case)
            
            elif ufo_correct and not base_correct:
                stats[entity_type]['ufo_only_correct'] += 1
                stats[entity_type]['ufo_correct'] += 1
                results['ufo_correct_base_wrong'].append(case)
            
            elif not ufo_correct and base_correct:
                stats[entity_type]['base_only_correct'] += 1
                stats[entity_type]['base_correct'] += 1
                results['base_correct_ufo_wrong'].append(case)
            
            else:
                results['both_wrong'].append(case)
    
    return results, stats


def generate_comparison_report(results, stats, output_file):
    """Generate a detailed comparison report"""
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 100 + "\n")
        f.write("NER模型对比分析报告：UFO-Full vs Baseline\n")
        f.write("=" * 100 + "\n\n")
        
         === Perform overall statistics ===
        f.write(" 📊 总体统计\n")
        f.write("-" * 100 + "\n")
        
        ufo_wins = len(results['ufo_correct_base_wrong'])
        base_wins = len(results['base_correct_ufo_wrong'])
        both_wrong = len(results['both_wrong'])
        both_correct = len(results['both_correct'])
        total = ufo_wins + base_wins + both_wrong + both_correct
        
        f.write(f"总实体数: {total}\n")
        f.write(f"两模型都正确: {both_correct} ({both_correct/total*100:.2f}%)\n")
        f.write(f"UFO正确，Baseline错误: {ufo_wins} ({ufo_wins/total*100:.2f}%) ✅\n")
        f.write(f"Baseline正确，UFO错误: {base_wins} ({base_wins/total*100:.2f}%) ⚠️\n")
        f.write(f"两模型都错误: {both_wrong} ({both_wrong/total*100:.2f}%)\n")
        f.write(f"\n净改进: {ufo_wins - base_wins:+d}\n")
        f.write(f"改进率: {(ufo_wins - base_wins)/total*100:+.2f}%\n")
        f.write("\n")
        
        # === Statistics by entity type ===
        f.write("## 📈 按实体类型分析\n")
        f.write("-" * 100 + "\n")
        f.write(f"{'类型':<12} {'总数':<8} {'UFO正确':<10} {'Base正确':<10} {'都对':<8} {'UFO独对':<10} {'Base独对':<10}\n")
        f.write("-" * 100 + "\n")
        
        for entity_type in sorted(stats.keys()):
            stat = stats[entity_type]
            f.write(
                f"{entity_type:<12} {stat['total']:<8} {stat['ufo_correct']:<10} "
                f"{stat['base_correct']:<10} {stat['both_correct']:<8} "
                f"{stat['ufo_only_correct']:<10} {stat['base_only_correct']:<10}\n"
            )
        
        f.write("\n\n")
        
        # === Cases where UFO outperforms the baseline ===
        f.write("## ✅ UFO修正的案例（Baseline错误，UFO正确）\n")
        f.write("=" * 100 + "\n\n")
        
        for i, case in enumerate(results['ufo_correct_base_wrong'][:20], 1):
            entity = case['entity']
            start, end = entity['start'], entity['end']
            
            f.write(f"【案例 {i}】\n")
            f.write(f"实体类型: {entity['type']}\n")
            f.write(f"实体文本: {entity['text']}\n")
            f.write(f"完整句子: {case['sentence']}\n")
            f.write(f"\n详细标注:\n")
            f.write(f"{'Token':<20} {'真实':<15} {'Baseline':<15} {'UFO':<15}\n")
            f.write("-" * 70 + "\n")
            
            for j in range(start, end + 1):
                token = case['tokens'][j]
                true_tag = case['true_tags'][j]
                base_tag = case['base_tags'][j]
                ufo_tag = case['ufo_tags'][j]
                
                # Mark differences
                base_marker = "❌" if base_tag != true_tag else "✓"
                ufo_marker = "✅" if ufo_tag == true_tag else "✗"
                
                # Clear token display
                display_token = token if not token.startswith('##') else token[2:]
                
                f.write(f"{display_token:<20} {true_tag:<15} {base_tag:<15} {base_marker}  {ufo_tag:<15} {ufo_marker}\n")
            
            f.write("\n" + "=" * 100 + "\n\n")
        
        # === Cases where the baseline outperforms UFO ===
        f.write("## ⚠️ UFO退化的案例（Baseline正确，UFO错误）\n")
        f.write("=" * 100 + "\n\n")
        
        for i, case in enumerate(results['base_correct_ufo_wrong'][:10], 1):
            entity = case['entity']
            start, end = entity['start'], entity['end']
            
            f.write(f"【案例 {i}】\n")
            f.write(f"实体类型: {entity['type']}\n")
            f.write(f"实体文本: {entity['text']}\n")
            f.write(f"完整句子: {case['sentence']}\n")
            f.write(f"\n详细标注:\n")
            f.write(f"{'Token':<20} {'真实':<15} {'Baseline':<15} {'UFO':<15}\n")
            f.write("-" * 70 + "\n")
            
            for j in range(start, end + 1):
                token = case['tokens'][j]
                true_tag = case['true_tags'][j]
                base_tag = case['base_tags'][j]
                ufo_tag = case['ufo_tags'][j]
                
                base_marker = "✅" if base_tag == true_tag else "✗"
                ufo_marker = "❌" if ufo_tag != true_tag else "✓"
                
                display_token = token if not token.startswith('##') else token[2:]
                
                f.write(f"{display_token:<20} {true_tag:<15} {base_tag:<15} {base_marker}  {ufo_tag:<15} {ufo_marker}\n")
            
            f.write("\n" + "=" * 100 + "\n\n")
    
    print(f"\n✅ 报告已生成: {output_file}")
    print(f"\n摘要:")
    print(f"  UFO修正的错误: {ufo_wins}")
    print(f"  UFO引入的错误: {base_wins}")
    print(f"  净改进: {ufo_wins - base_wins:+d}")


def compare_and_report(model_ufo, model_base, loader, idx2tag, device, tokenizer, output_file):
    """Primary comparison function"""
    
    all_ufo_preds = []
    all_base_preds = []
    all_true_labels = []
    all_tokens = []
    
    print("\n开始对比预测...")
    
    for batch in tqdm(loader, desc="预测中"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].cpu().numpy()
        
        # Get predictions
        preds_ufo = get_predictions(model_ufo, input_ids, attention_mask)
        preds_base = get_predictions(model_base, input_ids, attention_mask)
        
        # Process each sample
        for i in range(len(input_ids)):
            l_seq = labels[i]
            p_ufo = preds_ufo[i]
            p_base = preds_base[i]
            
            # Convert tokens
            tokens = tokenizer.convert_ids_to_tokens(input_ids[i])
            
            #Alignment: Retain only the positions of valid labels.
            valid_indices = [j for j, l in enumerate(l_seq) if l != -100]
            
            clean_tokens = [tokens[j] for j in valid_indices]
            true_tags = [idx2tag[l_seq[j]] for j in valid_indices]
            ufo_tags = [idx2tag[p_ufo[j]] for j in valid_indices]
            base_tags = [idx2tag[p_base[j]] for j in valid_indices]
            
            all_tokens.append(clean_tokens)
            all_true_labels.append(true_tags)
            all_ufo_preds.append(ufo_tags)
            all_base_preds.append(base_tags)
    
    print("\n分析预测差异...")
    results, stats = compare_predictions(all_ufo_preds, all_base_preds, all_true_labels, all_tokens)
    
    print("\n生成报告...")
    generate_comparison_report(results, stats, output_file)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # BERT path
    bert_dir = 'E:/PythonProject/ner_project/bert-base'
    
    # Load the tokenizer.
    tokenizer = BertTokenizerFast.from_pretrained(bert_dir, do_lower_case=True)
    
    # 1. Constructing label mappings
    print("\n读取训练数据以构建标签映射...")
    _, train_tags = read_data('./data/train.txt')
    label_set = sorted(set(t for s in train_tags for t in s))
    if 'O' not in label_set:
        label_set.append('O')
    tag2idx = {t: i for i, t in enumerate(label_set)}
    idx2tag = {i: t for t, i in tag2idx.items()}
    num_tags = len(tag2idx)
    print(f"标签数量: {num_tags}")
    print(f"标签类型: {list(tag2idx.keys())}")
    
    # 2. Load test data
    print("\n加载测试数据...")
    test_texts, test_tags = read_data('./data/test.txt')
    test_ds = NERDataset(test_texts, test_tags, tokenizer, tag2idx, max_len=512, stride=128)
    test_loader = DataLoader(test_ds, batch_size=8, shuffle=False, collate_fn=collate_fn)
    print(f"测试样本: {len(test_texts)}")
    
    # 3. Load model
    print("\n加载模型...")
    print("  - UFO-Full模型")
    model_ufo = BertBiLSTMCRF(bert_dir, 256, num_tags).to(device)
    model_ufo.load_state_dict(torch.load('权重/bert_bilstm_ufo_best.pt', map_location=device))
    
    print("  - Baseline模型")
    model_base = BertBiLSTMCRF_BBC(bert_dir, 256, num_tags).to(device)
    model_base.load_state_dict(torch.load('权重/best_model_noattn.pt', map_location=device))
    
    # 4. Execution Comparison
    compare_and_report(
        model_ufo, 
        model_base, 
        test_loader, 
        idx2tag, 
        device,
        tokenizer,
        "model_comparison_report.txt"
    )
    
    print("\n✅ 分析完成！请查看 model_comparison_report.txt")
