from pathlib import Path
import random
import numpy as np
import pandas as pd
import tensorflow as tf

from PIL import Image
from tqdm.auto import tqdm

from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    confusion_matrix,
    classification_report
)

# ==========================================
# 1. REPRODUCIBILITY
# ==========================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)

# ==========================================
# 2. PATHS AND CONFIGURATION
# ==========================================

ROOT = Path("data/processed/primary/images/train")
OUTPUT = Path("reports/part1")
OUTPUT.mkdir(parents=True, exist_ok=True)

IMG_SIZE = (128, 128)

class_map = {
    "NORMAL": 0,
    "PNEUMONIA": 1
}

# ==========================================
# 3. COLLECT IMAGE PATHS
# ==========================================

image_files = []

for class_name, label in class_map.items():

    folder = ROOT / class_name

    for path in sorted(folder.iterdir()):

        if path.is_file() and path.suffix.lower() in [
            ".jpeg", ".jpg", ".png"
        ]:
            image_files.append((path, label))

print("Total images:", len(image_files))

# ==========================================
# 4. HOG FEATURE EXTRACTION
# ==========================================

features = []
labels = []
failed_files = []

for path, label in tqdm(
    image_files,
    desc="Extracting HOG features",
    unit="image"
):

    try:

        with Image.open(path) as img:

            img = img.convert("L")
            img = img.resize(
                IMG_SIZE,
                Image.Resampling.LANCZOS
            )

            image = np.asarray(
                img,
                dtype=np.float32
            ) / 255.0

        feature_vector = hog(
            image,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            block_norm="L2-Hys",
            transform_sqrt=True,
            feature_vector=True
        )

        features.append(feature_vector)
        labels.append(label)

    except Exception as error:

        failed_files.append({
            "path": str(path),
            "error": str(error)
        })

X = np.asarray(features, dtype=np.float32)
y = np.asarray(labels, dtype=np.int32)

print("\nHOG feature matrix:", X.shape)
print("Labels:", y.shape)
print("Class counts:", np.bincount(y))

if failed_files:

    pd.DataFrame(failed_files).to_csv(
        OUTPUT / "hog_failed_images.csv",
        index=False
    )

# ==========================================
# 5. STRATIFIED TRAIN/VALIDATION SPLIT
# ==========================================

indices = np.arange(len(y))

train_idx, val_idx = train_test_split(
    indices,
    test_size=0.20,
    random_state=SEED,
    stratify=y
)

X_train, X_val = X[train_idx], X[val_idx]
y_train, y_val = y[train_idx], y[val_idx]

print("\nTraining samples:", len(y_train))
print("Validation samples:", len(y_val))

# ==========================================
# 6. LOGISTIC REGRESSION PIPELINE
# ==========================================

model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(
        max_iter=2000,
        class_weight=None,
        random_state=SEED
    ))
])

# ==========================================
# 7. TRAIN
# ==========================================

print("\nTraining Logistic Regression...")

model.fit(X_train, y_train)

print("Training completed.")

# ==========================================
# 8. PREDICTION
# ==========================================

probabilities = model.predict_proba(X_val)[:, 1]

predictions = (probabilities >= 0.5).astype(int)

# ==========================================
# 9. EVALUATION
# ==========================================

results = {
    "model": "HOG + Logistic Regression",
    "seed": SEED,
    "image_size": "128x128",
    "feature_count": X.shape[1],
    "accuracy": accuracy_score(y_val, predictions),
    "balanced_accuracy": balanced_accuracy_score(
        y_val, predictions
    ),
    "precision": precision_score(
        y_val, predictions, zero_division=0
    ),
    "recall": recall_score(
        y_val, predictions, zero_division=0
    ),
    "f1": f1_score(
        y_val, predictions, zero_division=0
    ),
    "pr_auc": average_precision_score(
        y_val, probabilities
    )
}

print("\nHOG BASELINE RESULTS")

for key, value in results.items():
    print(f"{key}: {value}")

print("\nCONFUSION MATRIX")
print(confusion_matrix(y_val, predictions))

print("\nCLASSIFICATION REPORT")
print(classification_report(
    y_val,
    predictions,
    target_names=["NORMAL", "PNEUMONIA"],
    zero_division=0
))

# ==========================================
# 10. SAVE RESULTS
# ==========================================

pd.DataFrame([results]).to_csv(
    OUTPUT / "hog_baseline_results.csv",
    index=False
)

import joblib

joblib.dump(
    model,
    OUTPUT / "hog_logistic_model.joblib"
)

print("\nHOG baseline results and model saved.")
