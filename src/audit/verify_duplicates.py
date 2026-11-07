
from pathlib import Path
from collections import defaultdict
import hashlib
import pandas as pd
from PIL import Image, ImageDraw

OUTPUT = Path("reports/part1")
OUTPUT.mkdir(parents=True, exist_ok=True)

# 1. Exact duplicate detection using SHA-256
hash_groups = defaultdict(list)

for path in Path("data/processed/primary/images").rglob("*"):
    if path.is_file() and path.suffix.lower() in [".jpeg", ".jpg", ".png"]:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        hash_groups[digest].append(str(path))

exact_groups = [
    paths for paths in hash_groups.values() if len(paths) > 1
]


from itertools import combinations
from collections import Counter

split_pair_counts = Counter()

print("\nEXACT DUPLICATE GROUP DETAILS")

for group_no, group in enumerate(exact_groups, start=1):
    splits = [
        Path(path).parts[-3]
        for path in group
    ]

    print(f"\nGroup {group_no}:")
    for path, split in zip(group, splits):
        print(f"  {split}: {path}")

    for a, b in combinations(sorted(set(splits)), 2):
        split_pair_counts[(a, b)] += 1

print("\nEXACT DUPLICATE GROUPS ACROSS SPLITS:")
for pair, count in split_pair_counts.items():
    print(pair, ":", count)


exact_records = []

for group in exact_groups:
    for i in range(len(group)):
        for j in range(i + 1, len(group)):
            exact_records.append({
                "image_1": group[i],
                "image_2": group[j]
            })

pd.DataFrame(exact_records).to_csv(
    OUTPUT / "exact_duplicate_pairs.csv", index=False
)

print("Exact duplicate groups:", len(exact_groups))
print("Exact duplicate pairs:", len(exact_records))

# 2. Visual inspection of train-test pHash candidates
pairs = pd.read_csv(OUTPUT / "near_duplicate_pairs.csv")

cross = pairs[
    ((pairs["split_1"] == "train") & (pairs["split_2"] == "test")) |
    ((pairs["split_1"] == "test") & (pairs["split_2"] == "train"))
].copy()

print("Train-test candidates:", len(cross))

if cross.empty:
    print("No train-test candidates to inspect.")
    raise SystemExit

thumb_w, thumb_h = 300, 260
label_h = 65
row_h = thumb_h + label_h
sheet = Image.new(
    "RGB",
    (thumb_w * 2, row_h * len(cross)),
    "white"
)
draw = ImageDraw.Draw(sheet)

for idx, row in enumerate(cross.itertuples(index=False)):
    y = idx * row_h

    for col, path in enumerate([row.image_1, row.image_2]):
        with Image.open(path) as img:
            thumb = img.convert("RGB")
            thumb.thumbnail((thumb_w - 10, thumb_h - 10))

        x = col * thumb_w
        sheet.paste(thumb, (x + 5, y + 5))

        draw.text(
            (x + 5, y + thumb_h),
            Path(path).name[:42],
            fill="black"
        )

    draw.text(
        (5, y + thumb_h + 20),
        f"Pair {idx + 1} | Distance: {row.hamming_distance}",
        fill="red"
    )

sheet.save(OUTPUT / "train_test_candidate_contact_sheet.jpg")

cross.to_csv(
    OUTPUT / "train_test_candidates.csv", index=False
)

print("Saved contact sheet:",
      OUTPUT / "train_test_candidate_contact_sheet.jpg")