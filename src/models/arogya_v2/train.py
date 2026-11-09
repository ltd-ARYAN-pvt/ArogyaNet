import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import random
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    confusion_matrix,
)

from model import build_arogyanet
from tqdm.auto import tqdm


# --------------------------------------------------
# Configuration
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[3]

DATA_DIR = ROOT / "data/processed/primary/images/train"
RESULT_DIR = ROOT / "experiments/version_2/results"
CHECKPOINT_DIR = ROOT / "experiments/version_2/checkpoints"

SEEDS = [42, 123, 2026]
IMAGE_SIZE = (128, 128)
BATCH_SIZE = 16
EPOCHS = 30

RESULT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

class TQDMProgressBar(tf.keras.callbacks.Callback):

    def on_train_begin(self, logs=None):
        self.epoch_bar = tqdm(
            total=self.params["epochs"],
            desc="Training epochs",
            unit="epoch"
        )

    def on_epoch_begin(self, epoch, logs=None):
        self.batch_bar = tqdm(
            total=self.params["steps"],
            desc=f"Epoch {epoch + 1}",
            unit="batch",
            leave=False
        )

    def on_train_batch_end(self, batch, logs=None):
        self.batch_bar.update(1)

        if logs:
            self.batch_bar.set_postfix(
                loss=f"{logs.get('loss', 0):.4f}"
            )

    def on_epoch_end(self, epoch, logs=None):
        self.batch_bar.close()

        logs = logs or {}

        self.epoch_bar.update(1)
        self.epoch_bar.set_postfix(
            loss=f"{logs.get('loss', 0):.4f}",
            val_loss=f"{logs.get('val_loss', 0):.4f}",
            val_pr_auc=f"{logs.get('val_pr_auc', 0):.4f}"
        )

    def on_train_end(self, logs=None):
        self.epoch_bar.close()


# --------------------------------------------------
# Reproducibility
# --------------------------------------------------

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


# --------------------------------------------------
# Dataset collection
# NORMAL = 0, PNEUMONIA = 1
# --------------------------------------------------

def collect_paths():

    paths = []
    labels = []

    for class_name, label in [("NORMAL", 0), ("PNEUMONIA", 1)]:

        class_dir = DATA_DIR / class_name

        files = sorted(class_dir.glob("*.jpeg"))

        paths.extend([str(p) for p in files])
        labels.extend([label] * len(files))

    return np.array(paths), np.array(labels)


# --------------------------------------------------
# Streaming TensorFlow data pipeline
# --------------------------------------------------

def preprocess_image(path, label):

    image = tf.io.read_file(path)

    image = tf.io.decode_jpeg(
        image,
        channels=1
    )

    image = tf.image.resize(
        image,
        IMAGE_SIZE
    )

    image = tf.cast(image, tf.float32) / 255.0

    return image, tf.cast(label, tf.float32)


def make_dataset(paths, labels, training=False, seed=42):

    dataset = tf.data.Dataset.from_tensor_slices(
        (paths, labels)
    )

    if training:
        dataset = dataset.shuffle(
            buffer_size=len(paths),
            seed=seed,
            reshuffle_each_iteration=True
        )

    dataset = dataset.map(
        preprocess_image,
        num_parallel_calls=2
    )

    dataset = dataset.batch(BATCH_SIZE)

    dataset = dataset.prefetch(1)

    return dataset


# --------------------------------------------------
# Train and evaluate one seed
# --------------------------------------------------

def train_one_seed(paths, labels, seed):

    print("\n" + "=" * 60)
    print(f"TRAINING AROGYANET WIDE | SEED {seed}")
    print("=" * 60)

    set_seed(seed)

    train_paths, val_paths, train_labels, val_labels = (
        train_test_split(
            paths,
            labels,
            test_size=0.20,
            stratify=labels,
            random_state=seed
        )
    )

    print("Training images:", len(train_paths))
    print("Validation images:", len(val_paths))

    train_ds = make_dataset(
        train_paths,
        train_labels,
        training=True,
        seed=seed
    )

    val_ds = make_dataset(
        val_paths,
        val_labels,
        training=False
    )

    model = build_arogyanet()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(
                curve="PR",
                name="pr_auc"
            )
        ]
    )

    checkpoint_path = (
        CHECKPOINT_DIR / f"arogyanet_wide_seed_{seed}.keras"
    )

    callbacks = [
        TQDMProgressBar(),

        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
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
            verbose=1
        ),

        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1
        )
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=0
    )

    # Save training history
    pd.DataFrame(history.history).to_csv(
        RESULT_DIR / f"history_seed_{seed}.csv",
        index=False
    )

    # Evaluate restored best weights
    probabilities = model.predict(
        val_ds,
        verbose=1
    ).ravel()

    predictions = (probabilities >= 0.5).astype(int)

    accuracy = accuracy_score(val_labels, predictions)
    balanced_acc = balanced_accuracy_score(
        val_labels, predictions
    )
    precision = precision_score(
        val_labels, predictions, zero_division=0
    )
    recall = recall_score(
        val_labels, predictions, zero_division=0
    )
    f1 = f1_score(
        val_labels, predictions, zero_division=0
    )
    pr_auc = average_precision_score(
        val_labels, probabilities
    )

    tn, fp, fn, tp = confusion_matrix(
        val_labels, predictions, labels=[0, 1]
    ).ravel()

    result = {
        "seed": seed,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "pr_auc": pr_auc,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "best_epoch": int(
            np.argmax(history.history["val_pr_auc"]) + 1
        ),
        "parameters": model.count_params()
    }

    print("\nValidation results:")
    for key, value in result.items():
        print(f"{key}: {value}")

    # Save best model
    model.save(checkpoint_path)

    return result


# --------------------------------------------------
# Main
# --------------------------------------------------

if __name__ == "__main__":

    print("TensorFlow version:", tf.__version__)
    print("GPUs:", tf.config.list_physical_devices("GPU"))

    paths, labels = collect_paths()

    print("\nTotal training-pool images:", len(paths))
    print("NORMAL:", int(np.sum(labels == 0)))
    print("PNEUMONIA:", int(np.sum(labels == 1)))

    results = []

    for seed in SEEDS:

        result = train_one_seed(
            paths,
            labels,
            seed
        )

        results.append(result)

        pd.DataFrame(results).to_csv(
            RESULT_DIR / "arogyanet_wide_seed_results.csv",
            index=False
        )

    results_df = pd.DataFrame(results)

    metrics = [
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "f1",
        "pr_auc"
    ]

    summary = pd.DataFrame({
        "metric": metrics,
        "mean": [results_df[m].mean() for m in metrics],
        "std": [results_df[m].std(ddof=1) for m in metrics]
    })

    summary.to_csv(
        RESULT_DIR / "arogyanet_wide_mean_std.csv",
        index=False
    )

    print("\nTHREE-SEED SUMMARY")
    print(summary.to_string(index=False))

    print("\nTRAINING COMPLETE!")