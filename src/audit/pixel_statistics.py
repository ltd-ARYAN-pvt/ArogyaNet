from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# Paths
INPUT = Path("reports/part1/image_inventory.csv")
OUTPUT = Path("reports/part1")
FIGURES = Path("figures/part1")

FIGURES.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INPUT)

# Check required columns
required = ["label", "split", "mean_brightness", "file_size_kb"]
missing = [col for col in required if col not in df.columns]

if missing:
    raise ValueError(f"Missing columns in inventory: {missing}")

# Class-wise numerical statistics
summary = df.groupby("label").agg(
    image_count=("filename", "count"),
    mean_brightness=("mean_brightness", "mean"),
    brightness_std=("mean_brightness", "std"),
    mean_file_size_kb=("file_size_kb", "mean"),
    median_file_size_kb=("file_size_kb", "median"),
    mean_width=("width", "mean"),
    mean_height=("height", "mean"),
    mean_aspect_ratio=("aspect_ratio", "mean")
)

summary.to_csv(OUTPUT / "class_pixel_statistics.csv")

print("\nCLASS-WISE PIXEL STATISTICS")
print(summary.round(3))

# 1. Brightness histogram
plt.figure(figsize=(9, 5))

for label in ["NORMAL", "PNEUMONIA"]:
    values = df.loc[
        df["label"] == label, "mean_brightness"
    ]
    plt.hist(values, bins=40, alpha=0.55, density=True, label=label)

plt.xlabel("Mean grayscale intensity (0-255)")
plt.ylabel("Density")
plt.title("Image Brightness Distribution by Class")
plt.legend()
plt.tight_layout()
plt.savefig(FIGURES / "brightness_distribution.png", dpi=200)
plt.close()

# 2. File size histogram
plt.figure(figsize=(9, 5))

for label in ["NORMAL", "PNEUMONIA"]:
    values = df.loc[
        df["label"] == label, "file_size_kb"
    ]
    plt.hist(values, bins=40, alpha=0.55, density=True, label=label)

plt.xlabel("File size (KB)")
plt.ylabel("Density")
plt.title("Image File Size Distribution by Class")
plt.legend()
plt.tight_layout()
plt.savefig(FIGURES / "file_size_distribution.png", dpi=200)
plt.close()

# 3. Brightness by split and class
split_summary = df.groupby(["split", "label"]).agg(
    count=("filename", "count"),
    mean_brightness=("mean_brightness", "mean"),
    std_brightness=("mean_brightness", "std"),
    median_file_size_kb=("file_size_kb", "median")
)

split_summary.to_csv(OUTPUT / "split_class_pixel_statistics.csv")

print("\nSPLIT-WISE CLASS STATISTICS")
print(split_summary.round(3))

print("\nSaved:")
print(OUTPUT / "class_pixel_statistics.csv")
print(OUTPUT / "split_class_pixel_statistics.csv")
print(FIGURES / "brightness_distribution.png")
print(FIGURES / "file_size_distribution.png")


# 4. Image mode distribution by class and split
mode_summary = pd.crosstab(
    [df["split"], df["label"]],
    df["mode"]
)

mode_summary.to_csv(OUTPUT / "image_mode_by_class.csv")

print("\nIMAGE MODE BY SPLIT AND CLASS")
print(mode_summary)

# 5. Dimension and aspect-ratio histograms
for column, title, xlabel, filename in [
    ("width", "Image Width Distribution", "Width (pixels)", "width_distribution.png"),
    ("height", "Image Height Distribution", "Height (pixels)", "height_distribution.png"),
    ("aspect_ratio", "Image Aspect Ratio Distribution", "Width / Height", "aspect_ratio_distribution.png")
]:
    plt.figure(figsize=(9, 5))

    for label in ["NORMAL", "PNEUMONIA"]:
        values = df.loc[df["label"] == label, column]
        plt.hist(
            values,
            bins=40,
            alpha=0.55,
            density=True,
            label=label
        )

    plt.xlabel(xlabel)
    plt.ylabel("Density")
    plt.title(title + " by Class")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES / filename, dpi=200)
    plt.close()

print("\nAdditional plots and mode statistics saved.")