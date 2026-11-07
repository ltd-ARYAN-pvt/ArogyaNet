from pathlib import Path
from itertools import combinations
import pandas as pd
import imagehash
from PIL import Image
from tqdm import tqdm

DATASET = Path("data/processed/primary/images")
OUTPUT = Path("reports/part1")
OUTPUT.mkdir(parents=True, exist_ok=True)

# Collect image paths
image_paths = []

for split in ["train", "test", "val"]:
    for label in ["NORMAL", "PNEUMONIA"]:
        folder = DATASET / split / label
        image_paths.extend(
            (path, split, label)
            for path in folder.iterdir()
            if path.is_file()
            and path.suffix.lower() in [".jpeg", ".jpg", ".png"]
        )

print("Total images:", len(image_paths))

# Calculate perceptual hashes
records = []

for path, split, label in tqdm(image_paths, desc="Hashing images"):
    try:
        with Image.open(path) as img:
            phash = imagehash.phash(img.convert("L"))

        records.append({
            "path": str(path),
            "filename": path.name,
            "split": split,
            "label": label,
            "phash": str(phash)
        })

    except Exception as error:
        print("Error:", path, error)

df = pd.DataFrame(records)
df.to_csv(OUTPUT / "perceptual_hashes.csv", index=False)

# Compare pairs
pairs = []

for (i, row1), (j, row2) in tqdm(
    combinations(df.iterrows(), 2),
    total=len(df) * (len(df) - 1) // 2,
    desc="Comparing hashes"
):
    hash1 = imagehash.hex_to_hash(row1["phash"])
    hash2 = imagehash.hex_to_hash(row2["phash"])

    distance = hash1 - hash2

    if distance <= 5:
        pairs.append({
            "image_1": row1["path"],
            "image_2": row2["path"],
            "split_1": row1["split"],
            "split_2": row2["split"],
            "label_1": row1["label"],
            "label_2": row2["label"],
            "hamming_distance": distance
        })

pairs_df = pd.DataFrame(pairs)
pairs_df.to_csv(OUTPUT / "near_duplicate_pairs.csv", index=False)

print("\nTotal hashed images:", len(df))
print("Near-duplicate pairs (distance <= 5):", len(pairs_df))

if not pairs_df.empty:
    print("\nPairs by split combination:")
    print(
        pairs_df.groupby(["split_1", "split_2"]).size()
    )

    print("\nFive closest pairs:")
    print(
        pairs_df.sort_values("hamming_distance").head(5).to_string(index=False)
    )

print("\nSaved perceptual hashes and duplicate pairs.")
