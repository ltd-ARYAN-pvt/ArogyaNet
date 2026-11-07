from pathlib import Path
import pandas as pd
from PIL import Image, ImageDraw

OUTPUT = Path("reports/part1")
pairs = pd.read_csv(OUTPUT / "train_test_candidates.csv")

# Larger thumbnails for visual comparison
W, H = 450, 350
LABEL_H = 90
ROW_H = H + LABEL_H

for batch_start in range(0, len(pairs), 5):
    batch = pairs.iloc[batch_start:batch_start + 5]

    sheet = Image.new(
        "RGB",
        (W * 2, ROW_H * len(batch)),
        "white"
    )
    draw = ImageDraw.Draw(sheet)

    for local_idx, row in enumerate(batch.itertuples(index=False)):
        y = local_idx * ROW_H

        for col, path in enumerate([row.image_1, row.image_2]):
            with Image.open(path) as img:
                thumb = img.convert("RGB")
                thumb.thumbnail((W - 10, H - 10))

            x = col * W
            sheet.paste(thumb, (x + 5, y + 5))

            draw.text(
                (x + 5, y + H),
                Path(path).name[:55],
                fill="black"
            )

        draw.text(
            (5, y + H + 55),
            f"Pair {batch_start + local_idx + 1} | "
            f"pHash distance: {row.hamming_distance}",
            fill="red"
        )

    output_file = OUTPUT / f"cross_split_batch_{batch_start // 5 + 1}.jpg"
    sheet.save(output_file, quality=95)
    print("Saved:", output_file)