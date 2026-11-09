
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import ks_2samp

ROOT = Path(__file__).resolve().parents[3]

SPLITS = {
    "train": ROOT / "data/processed/primary/images/train",
    "test": ROOT / "data/processed/primary/images/test",
}

CLASSES = ["NORMAL", "PNEUMONIA"]

def inspect_image(path, split, class_name):

    with Image.open(path) as img:

        gray = img.convert("L")
        arr = np.asarray(gray, dtype=np.float32)

        width, height = img.size

        return {
            "split": split,
            "class": class_name,
            "filename": path.name,
            "width": width,
            "height": height,
            "aspect_ratio": width / height,
            "brightness_mean": float(arr.mean()),
            "brightness_std": float(arr.std()),
            "file_size_kb": path.stat().st_size / 1024,
        }


def main():

    records = []

    for split, root in SPLITS.items():

        for class_name in CLASSES:

            folder = root / class_name

            for path in sorted(folder.glob("*.jpeg")):

                records.append(
                    inspect_image(path, split, class_name)
                )

    df = pd.DataFrame(records)

    output_dir = ROOT / "experiments/version_2/dataset_shift"
    output_dir.mkdir(parents=True, exist_ok=True)

    df.to_csv(
        output_dir / "image_level_statistics.csv",
        index=False
    )

    features = [
        "width",
        "height",
        "aspect_ratio",
        "brightness_mean",
        "brightness_std",
        "file_size_kb",
    ]

    print("\nIMAGE COUNTS")
    print(df.groupby(["split", "class"]).size())

    print("\nCLASS-WISE FEATURE SUMMARY")

    summary = (
        df.groupby(["split", "class"])[features]
        .agg(["mean", "std", "median"])
    )

    print(summary.round(2).to_string())

    print("\nTRAIN VS TEST: KS TESTS WITHIN EACH CLASS")

    comparisons = []

    for class_name in CLASSES:

        train_df = df[
            (df["split"] == "train") &
            (df["class"] == class_name)
        ]

        test_df = df[
            (df["split"] == "test") &
            (df["class"] == class_name)
        ]

        for feature in features:

            result = ks_2samp(
                train_df[feature],
                test_df[feature]
            )

            comparisons.append({
                "class": class_name,
                "feature": feature,
                "train_median": train_df[feature].median(),
                "test_median": test_df[feature].median(),
                "ks_statistic": result.statistic,
                "p_value": result.pvalue,
            })

    comparison_df = pd.DataFrame(comparisons)

    comparison_df.to_csv(
        output_dir / "train_test_ks_comparison.csv",
        index=False
    )

    print(comparison_df.round(5).to_string(index=False))

    print("\nSaved audit files to:", output_dir)


if __name__ == "__main__":
    main()
