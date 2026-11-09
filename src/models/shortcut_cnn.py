
from pathlib import Path
import random
import numpy as np
import pandas as pd
import tensorflow as tf

from PIL import Image
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    classification_report,
    confusion_matrix
)

# -------------------------
# Reproducibility
# -------------------------
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)

for gpu in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(gpu, True)

# -------------------------
# Paths and configuration
# -------------------------
ROOT = Path("data/processed/primary/images/train")
OUTPUT = Path("reports/part1")
OUTPUT.mkdir(parents=True, exist_ok=True)

IMG_SIZE = (16, 16)

class_map = {
    "NORMAL": 0,
    "PNEUMONIA": 1
}

# -------------------------
# Collect image paths
# -------------------------
image_files = []

for class_name, label in class_map.items():
    folder = ROOT / class_name

    for path in sorted(folder.iterdir()):
        if path.is_file() and path.suffix.lower() in [
            ".jpeg", ".jpg", ".png"
        ]:
            image_files.append((path, label))

# -------------------------
# Load and downsample images
# -------------------------
images = []
labels = []
paths = []
failed_files = []

for path, label in tqdm(
    image_files,
    desc="Loading and resizing images",
    unit="image"
):
    try:
        with Image.open(path) as img:
            img = img.convert("L")
            img = img.resize(
                IMG_SIZE,
                Image.Resampling.LANCZOS
            )
            arr = np.asarray(img, dtype=np.float32) / 255.0

        images.append(arr)
        labels.append(label)
        paths.append(str(path))

    except Exception as error:
        failed_files.append({
            "path": str(path),
            "error": str(error)
        })

if failed_files:
    pd.DataFrame(failed_files).to_csv(
        OUTPUT / "shortcut_failed_images.csv",
        index=False
    )

X = np.array(images, dtype=np.float32)[..., np.newaxis]
y = np.array(labels, dtype=np.int32)

print("\nDataset shape:", X.shape)
print("Labels shape:", y.shape)
print("Class counts:", np.bincount(y))
print("Failed images:", len(failed_files))

# -------------------------
# Stratified train/validation split
# -------------------------
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

# -------------------------
# Small CNN
# -------------------------
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(16, 16, 1)),

    tf.keras.layers.Conv2D(
        16, 3, padding="same", activation="relu"
    ),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(
        32, 3, padding="same", activation="relu"
    ),
    tf.keras.layers.GlobalAveragePooling2D(),

    tf.keras.layers.Dense(16, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# -------------------------
# tqdm training callback
# -------------------------
class TqdmTrainingProgress(tf.keras.callbacks.Callback):

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_bar = tqdm(
            total=self.params["steps"],
            desc=f"Epoch {epoch + 1}/{self.params['epochs']}",
            unit="batch",
            leave=False
        )

    def on_train_batch_end(self, batch, logs=None):
        self.epoch_bar.update(1)

        if logs:
            self.epoch_bar.set_postfix(
                loss=f"{logs.get('loss', 0):.4f}",
                accuracy=f"{logs.get('accuracy', 0):.4f}"
            )

    def on_epoch_end(self, epoch, logs=None):
        self.epoch_bar.close()

        logs = logs or {}

        tqdm.write(
            f"Epoch {epoch + 1}: "
            f"loss={logs.get('loss', 0):.4f}, "
            f"accuracy={logs.get('accuracy', 0):.4f}, "
            f"val_loss={logs.get('val_loss', 0):.4f}, "
            f"val_accuracy={logs.get('val_accuracy', 0):.4f}"
        )

# -------------------------
# Train
# -------------------------
history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=30,
    batch_size=32,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=4,
            restore_best_weights=True
        ),
        TqdmTrainingProgress()
    ],
    verbose=0
)

# -------------------------
# Evaluate
# -------------------------
probability_batches = []

for start in tqdm(
    range(0, len(X_val), 32),
    desc="Predicting validation images",
    unit="batch"
):
    batch = X_val[start:start + 32]
    probabilities_batch = model(batch, training=False).numpy().ravel()
    probability_batches.extend(probabilities_batch)

probabilities = np.array(probability_batches)
predictions = (probabilities >= 0.5).astype(int)

results = {
    "seed": SEED,
    "input_size": "16x16",
    "accuracy": accuracy_score(y_val, predictions),
    "balanced_accuracy": balanced_accuracy_score(y_val, predictions),
    "precision": precision_score(
        y_val, predictions, zero_division=0
    ),
    "recall": recall_score(
        y_val, predictions, zero_division=0
    ),
    "f1": f1_score(
        y_val, predictions, zero_division=0
    ),
    "pr_auc": average_precision_score(y_val, probabilities)
}

print("\nSHORTCUT MODEL RESULTS")

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
# MAJORITY CLASS BASELINE
# ==========================================

# Determine majority class from training labels
majority_class = int(np.bincount(y_train).argmax())

# Predict the same class for every validation sample
majority_pred = np.full(
    shape=len(y_val),
    fill_value=majority_class
)

# Metrics
majority_results = {
    "model": "Majority Class Baseline",
    "majority_class": majority_class,
    "accuracy": accuracy_score(y_val, majority_pred),
    "balanced_accuracy": balanced_accuracy_score(
        y_val, majority_pred
    ),
    "precision": precision_score(
        y_val, majority_pred, zero_division=0
    ),
    "recall": recall_score(
        y_val, majority_pred, zero_division=0
    ),
    "f1": f1_score(
        y_val, majority_pred, zero_division=0
    ),
    "pr_auc": average_precision_score(
        y_val, majority_pred
    ),
}

print("\nMAJORITY CLASS BASELINE")

for key, value in majority_results.items():
    print(f"{key}: {value}")

# Save results
pd.DataFrame([majority_results]).to_csv(
    OUTPUT / "majority_baseline_results.csv",
    index=False
)

# -------------------------
# Save results
# -------------------------
pd.DataFrame([results]).to_csv(
    OUTPUT / "shortcut_16x16_results.csv",
    index=False
)

pd.DataFrame(history.history).to_csv(
    OUTPUT / "shortcut_16x16_training_history.csv",
    index=False
)

model.save(OUTPUT / "shortcut_16x16_model.keras")

print("\nResults and model saved successfully.")