import numpy as np
from pathlib import Path
from collections import Counter
import pandas as pd
from PIL import Image, UnidentifiedImageError

DATASET = Path("data/processed/primary/images")
OUTPUT = Path("reports/part1")
OUTPUT.mkdir(parents=True, exist_ok=True)

records = []
corrupt_files = []

for split in ["train", "test", "val"]:
    for label in ["NORMAL", "PNEUMONIA"]:
        folder = DATASET / split / label

        for path in folder.iterdir():
            if not path.is_file() or path.name.startswith("."):
                continue

            if path.suffix.lower() not in [".jpeg", ".jpg", ".png"]:
                continue

            try:
                with Image.open(path) as img:
                    gray = np.asarray(img.convert("L"), dtype=np.float32)
                    mean_brightness = float(gray.mean())
                    brightness_std = float(gray.std())
                    width, height = img.size
                    records.append({
                        "filename": path.name,
                        "split": split,
                        "label": label,
                        "format": img.format,
                        "width": width,
                        "height": height,
                        "aspect_ratio": width / height,
                        "mode": img.mode,
                        "file_size_kb": path.stat().st_size / 1024,
                        "mean_brightness": mean_brightness,
                        "brightness_std": brightness_std,
                    })

            except (UnidentifiedImageError, OSError) as error:
                corrupt_files.append({
                    "filename": str(path),
                    "error": str(error)
                })

df = pd.DataFrame(records)

df.to_csv(OUTPUT / "image_inventory.csv", index=False)
pd.DataFrame(corrupt_files).to_csv(
    OUTPUT / "corrupt_files.csv", index=False
)

print("\nIMAGE COUNTS")
print(df.groupby(["split", "label"]).size())

print("\nIMAGE DIMENSION SUMMARY")
print(df[["width", "height", "aspect_ratio"]].describe())

print("\nIMAGE MODE DISTRIBUTION")
print(df["mode"].value_counts())

print("\nIMAGE FORMAT DISTRIBUTION")
print(df["format"].value_counts())

print("\nFILE SIZE SUMMARY (KB)")
print(df["file_size_kb"].describe())

print("\nCORRUPT FILES:", len(corrupt_files))
print("\nTOTAL VALID IMAGES:", len(df))
print("\nSaved inventory:", OUTPUT / "image_inventory.csv")