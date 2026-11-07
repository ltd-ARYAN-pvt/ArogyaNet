from pathlib import Path
import pandas as pd
from PIL import Image, ImageDraw

OUT = Path("reports/part1")
pairs = pd.read_csv(OUT / "train_test_candidates.csv")
pairs = pairs.sort_values("hamming_distance").head(5)

W, H = 500, 420
sheet = Image.new("RGB", (W * 2, H * 5), "white")
draw = ImageDraw.Draw(sheet)

for i, row in enumerate(pairs.itertuples(index=False)):
    for col, path in enumerate([row.image_1, row.image_2]):
        with Image.open(path) as img:
            img = img.convert("L")
            img.thumbnail((W - 20, H - 80))
            img = img.convert("RGB")

        x = col * W
        y = i * H
        sheet.paste(img, (x + 10, y + 10))

        draw.text(
            (x + 10, y + H - 60),
            Path(path).name[:55],
            fill="black"
        )

    draw.text(
        (10, i * H + H - 25),
        f"Pair {i+1} | pHash distance: {row.hamming_distance}",
        fill="red"
    )

sheet.save(OUT / "worst_five_train_test_pairs.jpg", quality=95)
print("Saved:", OUT / "worst_five_train_test_pairs.jpg")
