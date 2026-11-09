import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import random
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
)


# ==================================================
# CONFIGURATION
# ==================================================

ROOT = Path(__file__).resolve().parents[3]

DATA_DIR = ROOT / "data/processed/primary/images/train"

CHECKPOINT_DIR = (
    ROOT / "experiments/version_2/ablation_checkpoints"
)

OUTPUT_DIR = (
    ROOT / "experiments/version_2/cost_analysis"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

VARIANTS = ["A0", "A3"]
SEEDS = [42, 123, 2026]

IMAGE_SIZE = (128, 128)
BATCH_SIZE = 16

FN_COST = 20
FP_COST = 1

NORMAL_PREVALENCE = 0.90
PNEUMONIA_PREVALENCE = 0.10

THRESHOLDS = np.linspace(0.0, 1.0, 1001)


# ==================================================
# REPRODUCIBILITY
# ==================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


# ==================================================
# COLLECT DATA
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

    return np.array(paths), np.array(labels)


# ==================================================
# VALIDATION DATASET
# ==================================================

def preprocess_image(path, label):

    image = tf.io.read_file(path)
    image = tf.io.decode_jpeg(image, channels=1)
    image = tf.image.resize(image, IMAGE_SIZE)
    image = tf.cast(image, tf.float32) / 255.0

    return image, tf.cast(label, tf.float32)


def make_dataset(paths, labels):

    ds = tf.data.Dataset.from_tensor_slices(
        (paths, labels)
    )

    ds = ds.map(
        preprocess_image,
        num_parallel_calls=2
    )

    ds = ds.batch(BATCH_SIZE).prefetch(1)

    return ds


# ==================================================
# COST FUNCTION
# ==================================================

def calculate_cost(y_true, probabilities, threshold):

    predictions = (probabilities >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1]
    ).ravel()

    # Class-conditional error rates
    fpr = fp / (fp + tn)
    fnr = fn / (fn + tp)

    # Expected cost per child at camp prevalence
    cost_per_child = (
        NORMAL_PREVALENCE * FP_COST * fpr
        +
        PNEUMONIA_PREVALENCE * FN_COST * fnr
    )

    cost_per_1000 = cost_per_child * 1000

    return {
        "threshold": float(threshold),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "fpr": fpr,
        "fnr": fnr,
        "specificity": 1 - fpr,
        "sensitivity": 1 - fnr,
        "accuracy": accuracy_score(y_true, predictions),
        "balanced_accuracy": balanced_accuracy_score(
            y_true, predictions
        ),
        "precision": precision_score(
            y_true, predictions, zero_division=0
        ),
        "recall": recall_score(
            y_true, predictions, zero_division=0
        ),
        "f1": f1_score(
            y_true, predictions, zero_division=0
        ),
        "pr_auc": average_precision_score(
            y_true, probabilities
        ),
        "expected_cost_per_1000": cost_per_1000
    }


# ==================================================
# THRESHOLD SEARCH
# ==================================================

def find_optimal_threshold(y_true, probabilities):

    curve = []

    for threshold in THRESHOLDS:

        result = calculate_cost(
            y_true,
            probabilities,
            threshold
        )

        curve.append(result)

    curve_df = pd.DataFrame(curve)

    # Select minimum expected cost
    best_index = curve_df[
        "expected_cost_per_1000"
    ].idxmin()

    best_result = curve_df.loc[best_index].to_dict()

    return best_result, curve_df


# ==================================================
# MAIN
# ==================================================

if __name__ == "__main__":

    print("TensorFlow:", tf.__version__)
    print("GPU:", tf.config.list_physical_devices("GPU"))

    paths, labels = collect_paths()

    print("\nTotal images:", len(paths))

    all_results = []

    total_runs = len(VARIANTS) * len(SEEDS)

    with tqdm(
        total=total_runs,
        desc="Cost analysis",
        unit="model"
    ) as progress:

        for seed in SEEDS:

            set_seed(seed)

            # Recreate exactly the ablation validation split
            train_paths, val_paths, train_labels, val_labels = (
                train_test_split(
                    paths,
                    labels,
                    test_size=0.20,
                    stratify=labels,
                    random_state=seed
                )
            )

            val_ds = make_dataset(
                val_paths,
                val_labels
            )

            for variant in VARIANTS:

                progress.set_postfix(
                    variant=variant,
                    seed=seed
                )

                checkpoint = (
                    CHECKPOINT_DIR /
                    f"{variant}_seed_{seed}.keras"
                )

                if not checkpoint.exists():
                    raise FileNotFoundError(checkpoint)

                model = tf.keras.models.load_model(
                    checkpoint,
                    compile=False
                )

                probabilities = model.predict(
                    val_ds,
                    verbose=0
                ).ravel()

                # Default threshold
                default_result = calculate_cost(
                    val_labels,
                    probabilities,
                    0.5
                )

                # Cost-minimizing threshold
                best_result, curve_df = find_optimal_threshold(
                    val_labels,
                    probabilities
                )

                best_result.update({
                    "variant": variant,
                    "seed": seed,
                    "parameters": model.count_params(),
                    "default_cost_per_1000":
                        default_result["expected_cost_per_1000"],
                    "default_fn": default_result["fn"],
                    "default_fp": default_result["fp"]
                })

                all_results.append(best_result)

                curve_df.to_csv(
                    OUTPUT_DIR /
                    f"{variant}_seed_{seed}_threshold_curve.csv",
                    index=False
                )

                print(
                    f"\n{variant} | Seed {seed}"
                    f"\nDefault cost/1000: "
                    f"{default_result['expected_cost_per_1000']:.2f}"
                    f"\nOptimal threshold: "
                    f"{best_result['threshold']:.3f}"
                    f"\nOptimized cost/1000: "
                    f"{best_result['expected_cost_per_1000']:.2f}"
                    f"\nFN: {best_result['fn']} | "
                    f"FP: {best_result['fp']}"
                )

                pd.DataFrame(all_results).to_csv(
                    OUTPUT_DIR / "threshold_results.csv",
                    index=False
                )

                del model
                tf.keras.backend.clear_session()

                progress.update(1)

    # ==================================================
    # SUMMARY
    # ==================================================

    results_df = pd.DataFrame(all_results)

    results_df.to_csv(
        OUTPUT_DIR / "threshold_results.csv",
        index=False
    )

    summary_metrics = [
        "threshold",
        "expected_cost_per_1000",
        "default_cost_per_1000",
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "f1",
        "pr_auc",
        "sensitivity",
        "specificity"
    ]

    summary = (
        results_df
        .groupby("variant")[summary_metrics]
        .agg(["mean", "std"])
    )

    summary.to_csv(
        OUTPUT_DIR / "threshold_mean_std.csv"
    )

    print("\n" + "=" * 65)
    print("COST-SENSITIVE THRESHOLD SUMMARY")
    print("=" * 65)

    print(summary.round(4).to_string())

    print("\nALWAYS NORMAL baseline cost per 1000: 2000")
    print("ALWAYS PNEUMONIA baseline cost per 1000: 900")

    print("\nCOST ANALYSIS COMPLETED!")