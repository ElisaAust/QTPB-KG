from collections import defaultdict
from pathlib import Path

FILES = {
    "train": "train.txt",
    "dev"  : "dev.txt",
    "test" : "test.txt",
}

def parse_file(path):
    """解析 BIO 格式文件，返回 (字符总数, 实体计数dict)"""
    entity_counts = defaultdict(int)
    char_count = 0
    current_entity = None

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:          # 空行为句子分隔
                current_entity = None
                continue

            parts = line.split()
            if len(parts) < 2:
                continue

            char, tag = parts[0], parts[1]

            # 统计字符数（仅统计非空白 token）
            char_count += len(char)

            # 统计实体
            if tag.startswith("B-"):
                current_entity = tag[2:]
                entity_counts[current_entity] += 1
            elif tag == "O":
                current_entity = None

    return char_count, entity_counts


# ── 逐文件统计 ────────────────────────────────────────────────────────────────
all_entity_counts = defaultdict(int)
all_chars         = 0
file_stats        = {}

for split, path in FILES.items():
    chars, entities = parse_file(path)
    file_stats[split] = {"chars": chars, "entities": dict(entities)}
    all_chars += chars
    for ent, cnt in entities.items():
        all_entity_counts[ent] += cnt

# ── 输出 ──────────────────────────────────────────────────────────────────────
SEP  = "=" * 58
SEP2 = "-" * 58

print(SEP)
print(f"{'数据集统计报告':^54}")
print(SEP)

# 各文件分开统计
for split, stat in file_stats.items():
    print(f"\n【{split}.txt】  总字符数: {stat['chars']:,}")
    print(f"  {'实体类别':<12}  {'数量':>6}")
    print(f"  {SEP2[:40]}")
    for ent in sorted(stat["entities"]):
        print(f"  {ent:<12}  {stat['entities'][ent]:>6}")

# 三文件合计
print(f"\n{SEP}")
print(f"【三文件合计】  总字符数: {all_chars:,}")
print(f"  {'实体类别':<12}  {'数量':>6}  {'占比':>7}")
print(f"  {SEP2[:46]}")
total_entities = sum(all_entity_counts.values())
for ent in sorted(all_entity_counts):
    cnt  = all_entity_counts[ent]
    pct  = cnt / total_entities * 100
    print(f"  {ent:<12}  {cnt:>6}  {pct:>6.2f}%")
print(f"  {SEP2[:46]}")
print(f"  {'实体合计':<12}  {total_entities:>6}  {'100.00%':>7}")
print(SEP)