
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

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
)

from ablation_models import build_ablation_model


# ==================================================
# CONFIGURATION
# ==================================================

ROOT = Path(__file__).resolve().parents[3]

DATA_DIR = ROOT / "data/processed/primary/images/train"

RESULT_DIR = ROOT / "experiments/version_2/ablation_results"
CHECKPOINT_DIR = ROOT / "experiments/version_2/ablation_checkpoints"

VARIANTS = ["A0", "A1", "A2", "A3", "A4"]
SEEDS = [42, 123, 2026]

IMAGE_SIZE = (128, 128)
BATCH_SIZE = 16
EPOCHS = 30

RESULT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


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

        class_dir = DATA_DIR / class_name

        files = sorted(class_dir.glob("*.jpeg"))

        paths.extend(str(p) for p in files)
        labels.extend([label] * len(files))

    return np.array(paths), np.array(labels)


# ==================================================
# STREAMING DATA PIPELINE
# ==================================================

def preprocess_image(path, label):

    image = tf.io.read_file(path)
    image = tf.io.decode_jpeg(image, channels=1)

    image = tf.image.resize(image, IMAGE_SIZE)
    image = tf.cast(image, tf.float32) / 255.0

    return image, tf.cast(label, tf.float32)


def make_dataset(paths, labels, training=False, seed=42):

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

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

    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(1)

    return ds


# ==================================================
# PROGRESS BAR CALLBACK
# ==================================================

class TQDMProgressBar(tf.keras.callbacks.Callback):

    def on_train_begin(self, logs=None):

        self.epoch_bar = tqdm(
            total=self.params["epochs"],
            desc="Epochs",
            unit="epoch",
            leave=False
        )

    def on_epoch_begin(self, epoch, logs=None):

        steps = self.params.get("steps")

        self.batch_bar = tqdm(
            total=steps,
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


# ==================================================
# TRAIN ONE EXPERIMENT
# ==================================================

def train_experiment(
    variant,
    seed,
    train_paths,
    train_labels,
    val_paths,
    val_labels
):

    set_seed(seed)

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

    model = build_ablation_model(variant)

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
        f"{variant}_seed_{seed}.keras"
    )

    callbacks = [

        TQDMProgressBar(),

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

    start_time = time.perf_counter()

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=0
    )

    training_time = time.perf_counter() - start_time

    # Restore best checkpoint
    model = tf.keras.models.load_model(checkpoint)

    # Predictions
    probabilities = model.predict(
        val_ds,
        verbose=0
    ).ravel()

    predictions = (probabilities >= 0.5).astype(int)

    # Metrics
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
        val_labels,
        predictions,
        labels=[0, 1]
    ).ravel()

    best_epoch = int(
        np.argmax(history.history["val_pr_auc"]) + 1
    )

    result = {
        "variant": variant,
        "seed": seed,
        "parameters": model.count_params(),
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
        "best_epoch": best_epoch,
        "training_time_seconds": training_time
    }

    # Save history
    pd.DataFrame(history.history).to_csv(
        RESULT_DIR / f"{variant}_seed_{seed}_history.csv",
        index=False
    )

    del model, history, train_ds, val_ds
    tf.keras.backend.clear_session()

    return result


# ==================================================
# MAIN EXPERIMENT
# ==================================================

if __name__ == "__main__":

    print("TensorFlow:", tf.__version__)
    print("GPU:", tf.config.list_physical_devices("GPU"))

    paths, labels = collect_paths()

    print("\nTotal images:", len(paths))
    print("NORMAL:", int(np.sum(labels == 0)))
    print("PNEUMONIA:", int(np.sum(labels == 1)))

    all_results = []

    total_runs = len(VARIANTS) * len(SEEDS)

    with tqdm(
        total=total_runs,
        desc="Ablation experiments",
        unit="run"
    ) as experiment_bar:

        for seed in SEEDS:

            # Create split ONCE per seed.
            train_paths, val_paths, train_labels, val_labels = (
                train_test_split(
                    paths,
                    labels,
                    test_size=0.20,
                    stratify=labels,
                    random_state=seed
                )
            )

            print(
                f"\nSeed {seed}: "
                f"Train={len(train_paths)}, "
                f"Validation={len(val_paths)}"
            )

            for variant in VARIANTS:

                experiment_bar.set_postfix(
                    variant=variant,
                    seed=seed
                )

                print(f"\nRunning {variant} | Seed {seed}")

                result = train_experiment(
                    variant,
                    seed,
                    train_paths,
                    train_labels,
                    val_paths,
                    val_labels
                )

                all_results.append(result)

                pd.DataFrame(all_results).to_csv(
                    RESULT_DIR / "ablation_all_results.csv",
                    index=False
                )

                print(
                    f"{variant} | Seed {seed} | "
                    f"Accuracy={result['accuracy']:.4f} | "
                    f"PR-AUC={result['pr_auc']:.4f}"
                )

                experiment_bar.update(1)

    # ==================================================
    # THREE-SEED SUMMARY
    # ==================================================

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

    summary = (
        results_df
        .groupby("variant")[metrics]
        .agg(["mean", "std"])
    )

    summary.to_csv(
        RESULT_DIR / "ablation_mean_std.csv"
    )

    print("\n" + "=" * 60)
    print("FINAL ABLATION SUMMARY")
    print("=" * 60)

    print(summary.round(5).to_string())

    print("\nALL 15 ABLATION EXPERIMENTS COMPLETED!")