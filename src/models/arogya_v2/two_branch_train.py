
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from PIL import Image
from tqdm.auto import tqdm

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    confusion_matrix,
)

from two_branch_model import build_two_branch_model


# ==================================================
# CONFIGURATION
# ==================================================

ROOT = Path(__file__).resolve().parents[3]

DATA_DIR = ROOT / "data/processed/primary/images/train"

OUTPUT_DIR = ROOT / "experiments/version_2/two_branch"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
SCALER_DIR = OUTPUT_DIR / "scalers"

for folder in [OUTPUT_DIR, CHECKPOINT_DIR, SCALER_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 123, 2026]

IMAGE_SIZE = (128, 128)
BATCH_SIZE = 16
EPOCHS = 30


# ==================================================
# REPRODUCIBILITY
# ==================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


# ==================================================
# COLLECT IMAGE PATHS
# NORMAL = 0, PNEUMONIA = 1
# ==================================================

def collect_paths():

    paths, labels = [], []

    for class_name, label in [
        ("NORMAL", 0),
        ("PNEUMONIA", 1)
    ]:

        files = sorted(
            (DATA_DIR / class_name).glob("*.jpeg")
        )

        paths.extend(str(p) for p in files)
        labels.extend([label] * len(files))

    return np.array(paths), np.array(labels, dtype=np.int32)


# ==================================================
# EXTRACT ORIGINAL IMAGE METADATA
# ==================================================

def extract_metadata(paths):

    features = []

    for path in tqdm(paths, desc="Extracting metadata"):

        with Image.open(path) as img:

            gray = img.convert("L")

            width, height = gray.size

            mean_intensity = (
                np.asarray(gray, dtype=np.float32).mean() / 255.0
            )

            aspect_ratio = width / height

            log_area = np.log1p(width * height)

            features.append([
                mean_intensity,
                aspect_ratio,
                log_area
            ])

    return np.asarray(features, dtype=np.float32)


# ==================================================
# TF.DATA PIPELINE
# ==================================================

def preprocess_image(path, metadata, label):

    image = tf.io.read_file(path)
    image = tf.io.decode_jpeg(image, channels=1)

    image = tf.image.resize(image, IMAGE_SIZE)
    image = tf.cast(image, tf.float32) / 255.0

    inputs = {
        "xray_image": image,
        "image_metadata": metadata
    }

    return inputs, tf.cast(label, tf.float32)


def make_dataset(paths, metadata, labels, training=False, seed=42):

    ds = tf.data.Dataset.from_tensor_slices(
        (paths, metadata, labels)
    )

    if training:

        ds = ds.shuffle(
            buffer_size=len(paths),
            seed=seed,
            reshuffle_each_iteration=True
        )

    ds = ds.map(
        preprocess_image,
        num_parallel_calls=2
    )

    return ds.batch(BATCH_SIZE).prefetch(1)


# ==================================================
# TRAIN ONE SEED
# ==================================================

def train_one_seed(
    seed,
    paths,
    labels,
    metadata
):

    set_seed(seed)

    # Same stratified 80:20 split as A3
    (
        train_paths,
        val_paths,
        train_labels,
        val_labels,
        train_metadata,
        val_metadata
    ) = train_test_split(
        paths,
        labels,
        metadata,
        test_size=0.20,
        stratify=labels,
        random_state=seed
    )

    # Fit scaler ONLY on training metadata
    scaler = StandardScaler()

    train_metadata_scaled = scaler.fit_transform(
        train_metadata
    ).astype(np.float32)

    val_metadata_scaled = scaler.transform(
        val_metadata
    ).astype(np.float32)

    # Save scaler parameters for reproducibility
    np.savez(
        SCALER_DIR / f"metadata_scaler_seed_{seed}.npz",
        mean=scaler.mean_,
        scale=scaler.scale_
    )

    train_ds = make_dataset(
        train_paths,
        train_metadata_scaled,
        train_labels,
        training=True,
        seed=seed
    )

    val_ds = make_dataset(
        val_paths,
        val_metadata_scaled,
        val_labels
    )

    model = build_two_branch_model()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=1e-3
        ),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.AUC(
                curve="PR",
                name="pr_auc"
            )
        ]
    )

    checkpoint = (
        CHECKPOINT_DIR /
        f"two_branch_seed_{seed}.keras"
    )

    callbacks = [

        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint),
            monitor="val_pr_auc",
            mode="max",
            save_best_only=True,
            verbose=0
        ),

        tf.keras.callbacks.EarlyStopping(
            monitor="val_pr_auc",
            mode="max",
            patience=5,
            restore_best_weights=True,
            verbose=0
        ),

        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=0
        )
    ]

    print(f"\nTraining two-branch model | Seed {seed}")

    start_time = time.perf_counter()

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )

    training_time = time.perf_counter() - start_time

    # Load best checkpoint
    model = tf.keras.models.load_model(
        checkpoint,
        compile=False
    )

    probabilities = model.predict(
        val_ds,
        verbose=0
    ).ravel()

    predictions = (probabilities >= 0.5).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        val_labels,
        predictions,
        labels=[0, 1]
    ).ravel()

    best_epoch = (
        int(np.argmax(history.history["val_pr_auc"])) + 1
    )

    result = {
        "variant": "A3_two_branch",
        "seed": seed,
        "parameters": model.count_params(),

        "accuracy": accuracy_score(
            val_labels, predictions
        ),

        "balanced_accuracy": balanced_accuracy_score(
            val_labels, predictions
        ),

        "precision": precision_score(
            val_labels, predictions, zero_division=0
        ),

        "recall": recall_score(
            val_labels, predictions, zero_division=0
        ),

        "f1": f1_score(
            val_labels, predictions, zero_division=0
        ),

        "pr_auc": average_precision_score(
            val_labels, probabilities
        ),

        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),

        "best_epoch": best_epoch,
        "training_time_seconds": training_time
    }

    pd.DataFrame(history.history).to_csv(
        OUTPUT_DIR / f"two_branch_seed_{seed}_history.csv",
        index=False
    )

    print("\nSeed results:", result)

    del model, history, train_ds, val_ds
    tf.keras.backend.clear_session()

    return result


# ==================================================
# MAIN
# ==================================================

if __name__ == "__main__":

    print("TensorFlow:", tf.__version__)
    print("GPU:", tf.config.list_physical_devices("GPU"))

    paths, labels = collect_paths()

    print("\nTotal images:", len(paths))
    print("NORMAL:", int(np.sum(labels == 0)))
    print("PNEUMONIA:", int(np.sum(labels == 1)))

    print("\nExtracting metadata for all images...")

    metadata = extract_metadata(paths)

    all_results = []

    for seed in SEEDS:

        result = train_one_seed(
            seed,
            paths,
            labels,
            metadata
        )

        all_results.append(result)

        pd.DataFrame(all_results).to_csv(
            OUTPUT_DIR / "two_branch_results.csv",
            index=False
        )

    results_df = pd.DataFrame(all_results)

    metrics = [
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "f1",
        "pr_auc",
        "training_time_seconds"
    ]

    summary = results_df[metrics].agg(["mean", "std"]).T

    summary.to_csv(
        OUTPUT_DIR / "two_branch_summary.csv"
    )

    print("\n" + "=" * 65)
    print("TWO-BRANCH EXPERIMENT SUMMARY")
    print("=" * 65)

    print(summary.round(5).to_string())

    print("\nTWO-BRANCH EXPERIMENT COMPLETED!")
