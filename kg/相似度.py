import argparse
from collections import Counter
import numpy as np


# ══════════════════════════════════════════════════════════════
# 1. Jaccard bigram 相似度（0~10 分，无需任何依赖）
# ══════════════════════════════════════════════════════════════

def jaccard_bigram(a: str, b: str) -> float:
    def bigrams(s):
        return Counter(s[i:i+2] for i in range(len(s) - 1)) if len(s) >= 2 else Counter(s)
    bg_a, bg_b = bigrams(a), bigrams(b)
    if not bg_a and not bg_b:
        return 10.0
    if not bg_a or not bg_b:
        return 0.0
    intersection = sum((bg_a & bg_b).values())
    union        = sum((bg_a | bg_b).values())
    return round((intersection / union) * 10, 2)


# ══════════════════════════════════════════════════════════════
# 2. 用 transformers + torch 直接加载模型，手动 mean pooling
# ══════════════════════════════════════════════════════════════

def load_model(model_path: str):
    """加载本地 BERT 模型和分词器"""
    from transformers import BertTokenizer, BertModel
    print(f"加载分词器：{model_path}")
    tokenizer = BertTokenizer.from_pretrained(model_path)
    print(f"加载模型：{model_path}")
    model = BertModel.from_pretrained(model_path)
    model.eval()
    print("模型加载完成！")
    return tokenizer, model


def mean_pooling(token_embeddings, attention_mask):
    """对 token embeddings 做 attention mask 加权平均"""
    import torch
    mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return (token_embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)


def encode_texts(tokenizer, model, texts: list, batch_size: int = 64) -> dict:
    """批量编码文本，返回 {text: L2归一化向量}"""
    import torch
    all_vecs = []
    total = len(texts)
    for start in range(0, total, batch_size):
        batch = texts[start: start + batch_size]
        print(f"\r  编码进度：{min(start + batch_size, total)}/{total}", end="", flush=True)
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=64,
            return_tensors='pt'
        )
        with torch.no_grad():
            output = model(**encoded)
        vecs  = mean_pooling(output.last_hidden_state, encoded['attention_mask'])
        norms = vecs.norm(dim=1, keepdim=True).clamp(min=1e-9)
        vecs  = (vecs / norms).numpy()
        all_vecs.append(vecs)
    print()
    return {t: v for t, v in zip(texts, np.vstack(all_vecs))}


def sbert_score(vec_a, vec_b) -> float:
    """余弦相似度（已L2归一化），返回 0~10 分"""
    return round(max(0.0, float(np.dot(vec_a, vec_b))) * 10, 2)


# ══════════════════════════════════════════════════════════════
# 3. 工具函数
# ══════════════════════════════════════════════════════════════

def resolve(val: str, merge_map: dict) -> str:
    """递归查找最终代表词，防止链式引用"""
    visited = set()
    while val in merge_map and val not in visited:
        visited.add(val)
        val = merge_map[val]
    return val


# ══════════════════════════════════════════════════════════════
# 4. 主流程
# ══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input',     default=r'all_birds_triples.txt')
    parser.add_argument('--output',    default='all_birds_triples.txt')
    parser.add_argument('--log',       default='align_logs4.tsv')
    parser.add_argument('--model',     default=r'E:\PythonProject\三元组和图谱\sbert-base-chinese-nli')
    parser.add_argument('--pred',      default='食物有',
                        choices=['食物有', '栖息于', '分布于', '形态特征'])
    parser.add_argument('--threshold', type=float, default=7.0,
                        help='综合得分阈值（0~10），默认6.5')
    args = parser.parse_args()

    # ── 读入三元组 ──────────────────────────────────
    print(f"读入：{args.input}")
    triples = []
    with open(args.input, encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 4:  # 有行号，取后三列
                triples.append(parts[1:])
            elif len(parts) == 3:  # 无行号，直接用
                triples.append(parts)
    print(f"共 {len(triples)} 条三元组")

    # ── 提取目标谓词实体值 ──────────────────────────
    freq     = Counter(o for s, p, o in triples if p == args.pred)
    entities = list(freq.keys())
    print(f"\n谓词【{args.pred}】共 {len(entities)} 个唯一实体值")

    # ── 加载模型并编码 ──────────────────────────────
    tokenizer, model = load_model(args.model)
    print(f"\n开始编码...")
    vec_map = encode_texts(tokenizer, model, entities)

    # ── 计算所有实体对得分（上三角）────────────────
    n_pairs = len(entities) * (len(entities) - 1) // 2
    print(f"\n计算两两综合得分（共 {n_pairs} 对）...")
    candidates = []
    for i in range(len(entities)):
        for j in range(i + 1, len(entities)):
            a, b     = entities[i], entities[j]
            j_score  = jaccard_bigram(a, b)
            s_score  = sbert_score(vec_map[a], vec_map[b])
            combined = round(0.5 * j_score + 0.5 * s_score, 2)
            if combined >= args.threshold:
                candidates.append((combined, j_score, s_score, a, b))

    candidates.sort(reverse=True)
    print(f"综合得分 >= {args.threshold} 的候选对：{len(candidates)} 对\n")

    if not candidates:
        print("没有候选对，程序结束。")
        return

    # ── 交互判断 ────────────────────────────────────
    merge_map = {}
    log_rows  = []

    print("═" * 65)
    print("  y → 合并（低频词替换为高频词）")
    print("  n → 跳过")
    print("  s → 停止，保存当前结果")
    print("═" * 65 + "\n")

    shown = 0
    for combined, j_score, s_score, a, b in candidates:
        ra = resolve(a, merge_map)
        rb = resolve(b, merge_map)
        if ra == rb:
            continue

        shown += 1
        fa, fb = freq[ra], freq[rb]

        # 优先保留频率最高的；频率相同时保留字符更短的（更简洁）
        if fa > fb:
            keep, replace = ra, rb   # A 频率更高，保留 A
        elif fb > fa:
            keep, replace = rb, ra   # B 频率更高，保留 B
        else:
            # 频率相同：保留更短的（通常是更通用的词，如"森林"而非"山地森林"）
            keep, replace = (ra, rb) if len(ra) <= len(rb) else (rb, ra)

        print(f"[{shown}]  综合得分 {combined}"
              f"  ( Jaccard {j_score} × 50%  +  SBERT {s_score} × 50% )")
        print(f"  A: 【{ra}】  出现 {freq[ra]} 次")
        print(f"  B: 【{rb}】  出现 {freq[rb]} 次")
        print(f"  → 若选 y：将 『{replace}』 替换为 『{keep}』")

        while True:
            try:
                ans = input("\n  y / n / s > ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                ans = 's'
            if ans in ('y', 'n', 's'):
                break
            print("  请输入 y、n 或 s")

        print()
        if ans == 'y':
            merge_map[replace] = keep
            log_rows.append((ra, rb, combined, j_score, s_score, keep, replace, 'merged'))
            print(f"  ✓ 已记录：『{replace}』→『{keep}』\n")
        elif ans == 'n':
            log_rows.append((ra, rb, combined, j_score, s_score, '-', '-', 'skipped'))
            print(f"  ✗ 跳过\n")
        elif ans == 's':
            log_rows.append((ra, rb, combined, j_score, s_score, '-', '-', 'quit'))
            print("  已停止，保存当前结果...\n")
            break

    # ── 写出三元组 ──────────────────────────────────
    n_replaced  = 0
    out_triples = []
    for s, p, o in triples:
        if p == args.pred and o in merge_map:
            new_o = resolve(o, merge_map)
            if new_o != o:
                n_replaced += 1
            out_triples.append([s, p, new_o])
        else:
            out_triples.append([s, p, o])

    with open(args.output, 'w', encoding='utf-8') as f:
        for s, p, o in out_triples:
            f.write(f"{s}\t{p}\t{o}\n")

            # ── 写出日志 ────────────────────────────────────
    with open(args.log, 'w', encoding='utf-8') as f:
        f.write("实体A\t实体B\t综合得分\tJaccard\tSBERT\t保留词\t被替换词\t操作\n")
        for row in log_rows:
            f.write('\t'.join(str(x) for x in row) + '\n')

    # ── 汇总 ────────────────────────────────────────
    merged_count = sum(1 for r in log_rows if r[-1] == 'merged')
    after_unique = len(set(o for s, p, o in out_triples if p == args.pred))
    print("═" * 50)
    print(f"  展示候选对：{shown} 对")
    print(f"  确认合并：  {merged_count} 对")
    print(f"  替换三元组：{n_replaced} 条")
    print(f"  合并前唯一值：{len(entities)}")
    print(f"  合并后唯一值：{after_unique}")
    print(f"  输出文件：  {args.output}")
    print(f"  操作日志：  {args.log}")
    print("═" * 50)


if __name__ == '__main__':
    main()