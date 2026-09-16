from collections import defaultdict
from pathlib import Path

FILES = {
    "train": "train.txt",
    "dev"  : "dev.txt",
    "test" : "test.txt",
}

def parse_file(path):
    """Parse a BIO-format file and return (total character count, entity count dictionary)."""
    entity_counts = defaultdict(int)
    char_count = 0
    current_entity = None

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:          # Blank lines serve as sentence separators.
                current_entity = None
                continue

            parts = line.split()
            if len(parts) < 2:
                continue

            char, tag = parts[0], parts[1]

            # Count characters (counting only non-whitespace tokens)
            char_count += len(char)

            # Statistical Entity
            if tag.startswith("B-"):
                current_entity = tag[2:]
                entity_counts[current_entity] += 1
            elif tag == "O":
                current_entity = None

    return char_count, entity_counts


# Per-file statistics
all_entity_counts = defaultdict(int)
all_chars         = 0
file_stats        = {}

for split, path in FILES.items():
    chars, entities = parse_file(path)
    file_stats[split] = {"chars": chars, "entities": dict(entities)}
    all_chars += chars
    for ent, cnt in entities.items():
        all_entity_counts[ent] += cnt

# Output
SEP  = "=" * 58
SEP2 = "-" * 58

print(SEP)
print(f"{'数据集统计报告':^54}")
print(SEP)

# Statistics are compiled separately for each document.
for split, stat in file_stats.items():
    print(f"\n【{split}.txt】  总字符数: {stat['chars']:,}")
    print(f"  {'实体类别':<12}  {'数量':>6}")
    print(f"  {SEP2[:40]}")
    for ent in sorted(stat["entities"]):
        print(f"  {ent:<12}  {stat['entities'][ent]:>6}")

# Combined total of the three documents
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
