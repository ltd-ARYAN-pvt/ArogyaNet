
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import random
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    roc_auc_score,
    confusion_matrix,
    brier_score_loss,
    log_loss,
)

# ==================================================
# CONFIGURATION
# ==================================================

ROOT = Path(__file__).resolve().parents[3]

TRAIN_DIR = ROOT / "data/processed/primary/images/train"
TEST_DIR = ROOT / "data/processed/primary/images/test"

CHECKPOINT_DIR = ROOT / "experiments/version_2/ablation_checkpoints"

THRESHOLD_FILE = (
    ROOT / "experiments/version_2/final_threshold_analysis/"
    "final_threshold_results.csv"
)

OUTPUT_DIR = ROOT / "experiments/version_2/final_test_evaluation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

VARIANTS = ["A0", "A3"]
SEEDS = [42, 123, 2026]

IMAGE_SIZE = (128, 128)
BATCH_SIZE = 16

TARGET_PREVALENCE = 0.10
FN_COST = 20
FP_COST = 1


# ==================================================
# REPRODUCIBILITY
# ==================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


# ==================================================
# IMAGE COLLECTION
# NORMAL = 0, PNEUMONIA = 1
# ==================================================

def collect_paths(directory):

    paths, labels = [], []

    for class_name, label in [
        ("NORMAL", 0),
        ("PNEUMONIA", 1)
    ]:

        files = sorted(directory.joinpath(class_name).glob("*.jpeg"))

        paths.extend(str(p) for p in files)
        labels.extend([label] * len(files))

    return np.array(paths), np.array(labels, dtype=np.int32)


# ==================================================
# PREPROCESSING
# ==================================================

def preprocess_image(path, label):

    image = tf.io.read_file(path)
    image = tf.io.decode_jpeg(image, channels=1)
    image = tf.image.resize(image, IMAGE_SIZE)
    image = tf.cast(image, tf.float32) / 255.0

    return image, tf.cast(label, tf.float32)


def make_dataset(paths, labels):

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    ds = ds.map(
        preprocess_image,
        num_parallel_calls=2
    )

    return ds.batch(BATCH_SIZE).prefetch(1)


# ==================================================
# PLATT CALIBRATION
# ==================================================

def to_logits(probabilities):

    p = np.clip(probabilities, 1e-7, 1 - 1e-7)

    return np.log(p / (1 - p)).reshape(-1, 1)


def fit_platt(y_cal, p_cal):

    calibrator = LogisticRegression(
        C=1e6,
        solver="lbfgs",
        max_iter=2000
    )

    calibrator.fit(to_logits(p_cal), y_cal)

    return calibrator


def apply_platt(calibrator, probabilities):

    return calibrator.predict_proba(
        to_logits(probabilities)
    )[:, 1]


# ==================================================
# PRIOR CORRECTION
# ==================================================

def correct_prior(probabilities, source_prior):

    p = np.clip(probabilities, 1e-7, 1 - 1e-7)

    source_odds = source_prior / (1 - source_prior)

    target_odds = (
        TARGET_PREVALENCE / (1 - TARGET_PREVALENCE)
    )

    multiplier = target_odds / source_odds

    corrected_odds = (p / (1 - p)) * multiplier

    return corrected_odds / (1 + corrected_odds)


# ==================================================
# TARGET PREVALENCE WEIGHTS
# ==================================================

def make_target_weights(y_true, source_prior):

    return np.where(
        y_true == 1,
        TARGET_PREVALENCE / source_prior,
        (1 - TARGET_PREVALENCE) / (1 - source_prior)
    )


# ==================================================
# TEST METRICS
# ==================================================

def evaluate_test(y_true, probabilities, threshold, source_prior):

    predictions = (probabilities >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1]
    ).ravel()

    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)

    fpr = fp / (fp + tn)
    fnr = fn / (fn + tp)

    # Expected cost under 90:10 camp prevalence
    expected_cost = 1000 * (
        (1 - TARGET_PREVALENCE) * FP_COST * fpr
        +
        TARGET_PREVALENCE * FN_COST * fnr
    )

    weights = make_target_weights(y_true, source_prior)

    return {
        "threshold": float(threshold),

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
        "roc_auc": roc_auc_score(
            y_true, probabilities
        ),

        "sensitivity": sensitivity,
        "specificity": specificity,

        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),

        "camp_expected_cost_per_1000": expected_cost,

        "target_weighted_brier": brier_score_loss(
            y_true,
            probabilities,
            sample_weight=weights
        ),

        "target_weighted_log_loss": log_loss(
            y_true,
            probabilities,
            labels=[0, 1],
            sample_weight=weights
        )
    }


# ==================================================
# MAIN LOCKED TEST EVALUATION
# ==================================================

if __name__ == "__main__":

    print("=" * 65)
    print("AROGYANET — FINAL LOCKED TEST EVALUATION")
    print("=" * 65)

    print("TensorFlow:", tf.__version__)
    print("GPU:", tf.config.list_physical_devices("GPU"))

    train_paths, train_labels = collect_paths(TRAIN_DIR)
    test_paths, test_labels = collect_paths(TEST_DIR)

    print("\nTraining pool:", len(train_paths))
    print("Locked test images:", len(test_paths))

    print(
        "Test NORMAL:",
        int(np.sum(test_labels == 0))
    )

    print(
        "Test PNEUMONIA:",
        int(np.sum(test_labels == 1))
    )

    assert len(test_paths) == 624, (
        f"Expected 624 test images, found {len(test_paths)}"
    )

    assert np.sum(test_labels == 0) == 234
    assert np.sum(test_labels == 1) == 390

    # Load thresholds selected using validation data only
    threshold_df = pd.read_csv(THRESHOLD_FILE)

    expected_rows = len(VARIANTS) * len(SEEDS) * 2

    assert len(threshold_df) == expected_rows, (
        f"Expected {expected_rows} threshold rows, "
        f"found {len(threshold_df)}"
    )

    threshold_lookup = threshold_df.set_index(
        ["variant", "seed", "method"]
    )

    all_results = []

    total_runs = len(VARIANTS) * len(SEEDS)

    with tqdm(
        total=total_runs,
        desc="Locked test evaluation",
        unit="model"
    ) as progress:

        for seed in SEEDS:

            set_seed(seed)

            # Recreate exact original ablation split
            _, val_paths, _, val_labels = train_test_split(
                train_paths,
                train_labels,
                test_size=0.20,
                stratify=train_labels,
                random_state=seed
            )

            # Recreate exact calibration subset
            cal_paths, _, cal_labels, _ = train_test_split(
                val_paths,
                val_labels,
                test_size=0.50,
                stratify=val_labels,
                random_state=seed
            )

            source_prior = float(np.mean(cal_labels))

            cal_ds = make_dataset(cal_paths, cal_labels)

            print(
                f"\nSeed {seed} | "
                f"Calibration images={len(cal_paths)} | "
                f"Source prevalence={source_prior:.4f}"
            )

            # Same locked test data for every model
            test_ds = make_dataset(test_paths, test_labels)

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

                # Calibration predictions
                p_cal_raw = model.predict(
                    cal_ds,
                    verbose=0
                ).ravel()

                # One inference pass on locked test images
                p_test_raw = model.predict(
                    test_ds,
                    verbose=0
                ).ravel()

                # Fit calibrator on validation calibration subset
                calibrator = fit_platt(
                    cal_labels,
                    p_cal_raw
                )

                # Calibrated test probabilities
                p_test_platt = apply_platt(
                    calibrator,
                    p_test_raw
                )

                # Prior-corrected probabilities
                p_test_corrected = correct_prior(
                    p_test_platt,
                    source_prior
                )

                methods = {
                    "platt_source": p_test_platt,
                    "prior_corrected": p_test_corrected
                }

                for method, probabilities in methods.items():

                    key = (variant, seed, method)

                    if key not in threshold_lookup.index:
                        raise KeyError(
                            f"Missing validation threshold: {key}"
                        )

                    # Threshold already selected on validation tuning data
                    selected_threshold = float(
                        threshold_lookup.loc[
                            key, "selected_threshold"
                        ]
                    )

                    metrics = evaluate_test(
                        test_labels,
                        probabilities,
                        selected_threshold,
                        source_prior
                    )

                    row = {
                        "variant": variant,
                        "seed": seed,
                        "method": method,
                        "parameters": model.count_params(),
                        "source_prevalence": source_prior,
                        "target_prevalence": TARGET_PREVALENCE,
                        **metrics
                    }

                    all_results.append(row)

                    print(
                        f"\n{variant} | Seed {seed} | {method}"
                        f"\nThreshold: {selected_threshold:.4f}"
                        f"\nSensitivity: {metrics['sensitivity']:.4f}"
                        f"\nSpecificity: {metrics['specificity']:.4f}"
                        f"\nCamp cost / 1000: "
                        f"{metrics['camp_expected_cost_per_1000']:.2f}"
                    )

                del model, calibrator
                tf.keras.backend.clear_session()

                progress.update(1)

    # ==================================================
    # SAVE INDIVIDUAL RESULTS
    # ==================================================

    results_df = pd.DataFrame(all_results)

    results_path = OUTPUT_DIR / "final_test_results.csv"

    results_df.to_csv(results_path, index=False)

    # ==================================================
    # MEAN ± STANDARD DEVIATION
    # ==================================================

    metrics = [
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "f1",
        "pr_auc",
        "roc_auc",
        "sensitivity",
        "specificity",
        "camp_expected_cost_per_1000",
        "target_weighted_brier",
        "target_weighted_log_loss"
    ]

    summary = (
        results_df
        .groupby(["variant", "method"])[metrics]
        .agg(["mean", "std"])
    )

    summary_path = OUTPUT_DIR / "final_test_summary.csv"

    summary.to_csv(summary_path)

    print("\n" + "=" * 75)
    print("FINAL LOCKED TEST SUMMARY")
    print("=" * 75)

    print(summary.round(4).to_string())

    print("\nSaved:", results_path)
    print("Saved:", summary_path)

    print("\nFINAL TEST EVALUATION COMPLETED!")