from pathlib import Path
import random
import gc

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
    confusion_matrix,
    classification_report
)

# ==========================================
# 1. CONFIGURATION
# ==========================================

ROOT = Path("data/processed/primary/images/train")
OUTPUT = Path("reports/part1/transfer_learning")
OUTPUT.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 16
IMG_SIZE = (224, 224)
SEEDS = [42, 123, 2026]

HEAD_EPOCHS = 5
FINE_TUNE_EPOCHS = 5
UNFREEZE_LAST = 30

class_map = {
    "NORMAL": 0,
    "PNEUMONIA": 1
}

# ==========================================
# 2. REPRODUCIBILITY AND GPU
# ==========================================

for gpu in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(gpu, True)

print("TensorFlow:", tf.__version__)
print("GPUs:", tf.config.list_physical_devices("GPU"))

# ==========================================
# 3. COLLECT IMAGE PATHS ONLY
# ==========================================

image_paths = []
labels = []
failed_files = []

for class_name, label in class_map.items():

    folder = ROOT / class_name

    for path in sorted(folder.iterdir()):

        if path.is_file() and path.suffix.lower() in [
            ".jpeg", ".jpg", ".png"
        ]:
            image_paths.append(str(path))
            labels.append(label)

image_paths = np.asarray(image_paths)
y = np.asarray(labels, dtype=np.int32)

print("Total images:", len(image_paths))
print("Class counts:", np.bincount(y))

# ==========================================
# 4. STREAMING DATASET PIPELINE
# ==========================================

def load_image(path, label):

    image_bytes = tf.io.read_file(path)

    image = tf.io.decode_image(
        image_bytes,
        channels=3,
        expand_animations=False
    )

    image.set_shape([None, None, 3])

    image = tf.image.resize(
        image,
        IMG_SIZE
    )

    # Keep pixel values in 0-255 range.
    # MobileNetV3 include_preprocessing=True
    # handles its own input rescaling.

    return image, label


def make_dataset(paths, labels, training=False):

    ds = tf.data.Dataset.from_tensor_slices(
        (paths, labels)
    )

    if training:

        ds = ds.shuffle(
            buffer_size=len(labels),
            seed=SEED,
            reshuffle_each_iteration=True
        )

    ds = ds.map(
        load_image,
        num_parallel_calls=2
    )

    ds = ds.batch(BATCH_SIZE)

    ds = ds.prefetch(1)

    return ds

# ==========================================
# 5. MODEL BUILDER
# ==========================================

def build_model():

    backbone = tf.keras.applications.MobileNetV3Small(
        input_shape=(224, 224, 3),
        alpha=1.0,
        include_top=False,
        weights="imagenet",
        include_preprocessing=True
    )

    backbone.trainable = False

    inputs = tf.keras.Input(
        shape=(224, 224, 3)
    )

    x = backbone(inputs, training=False)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    x = tf.keras.layers.Dropout(0.3)(x)

    outputs = tf.keras.layers.Dense(
        1,
        activation="sigmoid"
    )(x)

    model = tf.keras.Model(inputs, outputs)

    return model, backbone

# ==========================================
# 6. TQDM CALLBACK
# ==========================================

class TqdmProgress(tf.keras.callbacks.Callback):

    def on_epoch_begin(self, epoch, logs=None):

        steps = self.params.get("steps")

        self.bar = tqdm(
            total=steps,
            desc=f"Epoch {epoch + 1}",
            unit="batch",
            leave=False
        )

    def on_train_batch_end(self, batch, logs=None):

        self.bar.update(1)

        if logs:

            self.bar.set_postfix(
                loss=f"{logs.get('loss', 0):.4f}"
            )

    def on_epoch_end(self, epoch, logs=None):

        self.bar.close()

        logs = logs or {}

        tqdm.write(
            f"Epoch {epoch + 1} | "
            f"loss={logs.get('loss', 0):.4f} | "
            f"val_loss={logs.get('val_loss', 0):.4f} | "
            f"val_accuracy={logs.get('val_accuracy', 0):.4f}"
        )

# ==========================================
# 7. RUN THREE SEEDS
# ==========================================

all_results = []

for SEED in SEEDS:

    print("\n" + "=" * 55)
    print(f"STARTING SEED: {SEED}")
    print("=" * 55)

    random.seed(SEED)
    np.random.seed(SEED)
    tf.keras.utils.set_random_seed(SEED)

    # Stratified split
    indices = np.arange(len(y))

    train_idx, val_idx = train_test_split(
        indices,
        test_size=0.20,
        random_state=SEED,
        stratify=y
    )

    train_ds = make_dataset(
        image_paths[train_idx],
        y[train_idx],
        training=True
    )

    val_ds = make_dataset(
        image_paths[val_idx],
        y[val_idx]
    )

    y_val = y[val_idx]

    print("Training samples:", len(train_idx))
    print("Validation samples:", len(val_idx))

    # Build model
    model, backbone = build_model()

    # --------------------------------------
    # STAGE 1: TRAIN CLASSIFICATION HEAD
    # --------------------------------------

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=1e-3
        ),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    print("\nStage 1: Training classification head")

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=HEAD_EPOCHS,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=2,
                restore_best_weights=True
            ),
            TqdmProgress()
        ],
        verbose=0
    )

    # --------------------------------------
    # STAGE 2: FINE-TUNING
    # --------------------------------------

    print("\nStage 2: Fine-tuning backbone")

    backbone.trainable = True

    # Freeze all but last 30 layers
    for layer in backbone.layers[:-UNFREEZE_LAST]:
        layer.trainable = False

    # Keep BatchNorm frozen for stable transfer
    for layer in backbone.layers:
        if isinstance(
            layer,
            tf.keras.layers.BatchNormalization
        ):
            layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=1e-5
        ),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=FINE_TUNE_EPOCHS,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=2,
                restore_best_weights=True
            ),
            TqdmProgress()
        ],
        verbose=0
    )

    # --------------------------------------
    # EVALUATION
    # --------------------------------------

    probabilities = model.predict(
        val_ds,
        verbose=0
    ).ravel()

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    result = {
        "seed": SEED,
        "accuracy": accuracy_score(
            y_val, predictions
        ),
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

    all_results.append(result)

    print(f"\nRESULTS FOR SEED {SEED}")

    for key, value in result.items():
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

    # Save model for this seed
    model.save(
        OUTPUT / f"mobilenetv3_seed_{SEED}.keras"
    )

    del model, backbone, train_ds, val_ds
    gc.collect()
    tf.keras.backend.clear_session()

# ==========================================
# 8. SAVE THREE-SEED RESULTS
# ==========================================

results_df = pd.DataFrame(all_results)

results_df.to_csv(
    OUTPUT / "mobilenetv3_three_seed_results.csv",
    index=False
)

metric_columns = [
    "accuracy",
    "balanced_accuracy",
    "precision",
    "recall",
    "f1",
    "pr_auc"
]

summary = pd.DataFrame({
    "metric": metric_columns,
    "mean": [
        results_df[col].mean()
        for col in metric_columns
    ],
    "std": [
        results_df[col].std(ddof=1)
        for col in metric_columns
    ]
})

summary.to_csv(
    OUTPUT / "mobilenetv3_mean_std.csv",
    index=False
)

print("\nTHREE-SEED SUMMARY")
print(summary.to_string(index=False))

print("\nAll transfer-learning experiments completed.")