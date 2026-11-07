from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path("data/processed/primary/images")
OUTPUT = Path("reports/part1")
OUTPUT.mkdir(parents=True, exist_ok=True)

records = []

for path in ROOT.rglob("*"):
    if not path.is_file():
        continue

    if path.suffix.lower() not in [".jpeg", ".jpg", ".png"]:
        continue

    with Image.open(path) as img:
        if img.mode != "RGB":
            continue

        arr = np.asarray(img, dtype=np.int16)

        r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]

        records.append({
            "filename": path.name,
            "split": path.parts[-3],
            "label": path.parts[-2],
            "R_G_mean_abs_diff": np.abs(r - g).mean(),
            "G_B_mean_abs_diff": np.abs(g - b).mean(),
            "R_B_mean_abs_diff": np.abs(r - b).mean(),
            "channels_identical": np.array_equal(r, g)
                and np.array_equal(g, b)
        })

df = pd.DataFrame(records)

df.to_csv(OUTPUT / "rgb_channel_audit.csv", index=False)

print("RGB images:", len(df))

if len(df):
    print("\nChannel difference statistics:")
    print(df[[
        "R_G_mean_abs_diff",
        "G_B_mean_abs_diff",
        "R_B_mean_abs_diff"
    ]].describe())

    print("\nIdentical RGB channels:")
    print(df["channels_identical"].value_counts())

    print("\nSplit/class distribution:")
    print(pd.crosstab([df["split"], df["label"]], df["channels_identical"]))