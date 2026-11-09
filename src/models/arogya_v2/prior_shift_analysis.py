
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
    brier_score_loss,
    log_loss,
    confusion_matrix,
)

# ==================================================
# CONFIGURATION
# ==================================================

ROOT = Path(__file__).resolve().parents[3]

DATA_DIR = ROOT / "data/processed/primary/images/train"
CHECKPOINT_DIR = ROOT / "experiments/version_2/ablation_checkpoints"
OUTPUT_DIR = ROOT / "experiments/version_2/prior_shift"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

VARIANTS = ["A0", "A3"]
SEEDS = [42, 123, 2026]

IMAGE_SIZE = (128, 128)
BATCH_SIZE = 16

TARGET_PREVALENCE = 0.10
FN_COST = 20
FP_COST = 1

THRESHOLDS = np.linspace(0.001, 0.999, 999)


# ==================================================
# REPRODUCIBILITY
# ==================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


# ==================================================
# DATA COLLECTION
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
# DATA PIPELINE
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

    return ds.batch(BATCH_SIZE).prefetch(1)


# ==================================================
# PLATT SCALING
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

    calibrator.fit(
        to_logits(p_cal),
        y_cal
    )

    return calibrator


def apply_platt(calibrator, probabilities):

    return calibrator.predict_proba(
        to_logits(probabilities)
    )[:, 1]


# ==================================================
# PRIOR PROBABILITY CORRECTION
# ==================================================

def correct_prior(probabilities, source_prior, target_prior):

    p = np.clip(
        probabilities,
        1e-7,
        1 - 1e-7
    )

    source_odds = source_prior / (1 - source_prior)
    target_odds = target_prior / (1 - target_prior)

    odds_multiplier = target_odds / source_odds

    corrected_odds = (
        (p / (1 - p)) * odds_multiplier
    )

    corrected = corrected_odds / (1 + corrected_odds)

    return corrected


# ==================================================
# TARGET-PREVALENCE SAMPLE WEIGHTS
# ==================================================

def make_target_weights(y_true, source_prior, target_prior):

    weights = np.where(
        y_true == 1,
        target_prior / source_prior,
        (1 - target_prior) / (1 - source_prior)
    )

    return weights


# ==================================================
# WEIGHTED ECE
# ==================================================

def weighted_ece(
    y_true,
    probabilities,
    sample_weights,
    n_bins=10
):

    edges = np.linspace(0, 1, n_bins + 1)

    total_weight = np.sum(sample_weights)
    ece = 0.0

    for i in range(n_bins):

        if i == n_bins - 1:
            mask = (
                (probabilities >= edges[i])
                & (probabilities <= edges[i + 1])
            )
        else:
            mask = (
                (probabilities >= edges[i])
                & (probabilities < edges[i + 1])
            )

        if not np.any(mask):
            continue

        w = sample_weights[mask]
        bin_weight = np.sum(w)

        confidence = np.average(
            probabilities[mask],
            weights=w
        )

        observed = np.average(
            y_true[mask],
            weights=w
        )

        ece += (
            bin_weight / total_weight
        ) * abs(confidence - observed)

    return ece


# ==================================================
# WEIGHTED CALIBRATION METRICS
# ==================================================

def calibration_metrics(y_true, probabilities, weights):

    return {
        "weighted_brier": brier_score_loss(
            y_true,
            probabilities,
            sample_weight=weights
        ),
        "weighted_log_loss": log_loss(
            y_true,
            probabilities,
            labels=[0, 1],
            sample_weight=weights
        ),
        "weighted_ece": weighted_ece(
            y_true,
            probabilities,
            weights
        )
    }


# ==================================================
# COST EVALUATION
# ==================================================

def evaluate_cost(y_true, probabilities, threshold):

    predictions = (
        probabilities >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1]
    ).ravel()

    fpr = fp / (fp + tn)
    fnr = fn / (fn + tp)

    expected_cost = (
        (1 - TARGET_PREVALENCE) * FP_COST * fpr
        +
        TARGET_PREVALENCE * FN_COST * fnr
    )

    return {
        "threshold": float(threshold),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "fpr": fpr,
        "fnr": fnr,
        "sensitivity": 1 - fnr,
        "specificity": 1 - fpr,
        "expected_cost_per_1000": expected_cost * 1000
    }


def find_best_threshold(y_true, probabilities):

    results = [
        evaluate_cost(y_true, probabilities, t)
        for t in THRESHOLDS
    ]

    curve = pd.DataFrame(results)

    best = curve.loc[
        curve["expected_cost_per_1000"].idxmin()
    ].to_dict()

    return best, curve


# ==================================================
# MAIN
# ==================================================

if __name__ == "__main__":

    print("TensorFlow:", tf.__version__)
    print("GPU:", tf.config.list_physical_devices("GPU"))

    paths, labels = collect_paths()

    all_results = []

    with tqdm(
        total=len(VARIANTS) * len(SEEDS),
        desc="Prior shift experiments",
        unit="model"
    ) as progress:

        for seed in SEEDS:

            set_seed(seed)

            # Same original split used in ablation experiment
            _, val_paths, _, val_labels = train_test_split(
                paths,
                labels,
                test_size=0.20,
                stratify=labels,
                random_state=seed
            )

            # Same calibration/evaluation split as calibration analysis
            cal_paths, eval_paths, cal_labels, eval_labels = (
                train_test_split(
                    val_paths,
                    val_labels,
                    test_size=0.50,
                    stratify=val_labels,
                    random_state=seed
                )
            )

            source_prior = float(np.mean(cal_labels))

            weights = make_target_weights(
                eval_labels,
                source_prior,
                TARGET_PREVALENCE
            )

            cal_ds = make_dataset(cal_paths, cal_labels)
            eval_ds = make_dataset(eval_paths, eval_labels)

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

                p_cal_raw = model.predict(
                    cal_ds,
                    verbose=0
                ).ravel()

                p_eval_raw = model.predict(
                    eval_ds,
                    verbose=0
                ).ravel()

                # Fit Platt scaler using calibration subset only
                calibrator = fit_platt(
                    cal_labels,
                    p_cal_raw
                )

                p_eval_calibrated = apply_platt(
                    calibrator,
                    p_eval_raw
                )

                # Correct prior from source to camp prevalence
                p_eval_corrected = correct_prior(
                    p_eval_calibrated,
                    source_prior,
                    TARGET_PREVALENCE
                )

                # Weighted calibration metrics
                for method, probabilities in [
                    ("platt_source", p_eval_calibrated),
                    ("prior_corrected", p_eval_corrected)
                ]:

                    metrics = calibration_metrics(
                        eval_labels,
                        probabilities,
                        weights
                    )

                    all_results.append({
                        "variant": variant,
                        "seed": seed,
                        "method": method,
                        "source_prevalence": source_prior,
                        "target_prevalence": TARGET_PREVALENCE,
                        **metrics
                    })

                # Threshold optimization on corrected probabilities
                best, curve = find_best_threshold(
                    eval_labels,
                    p_eval_corrected
                )

                best.update({
                    "variant": variant,
                    "seed": seed,
                    "source_prevalence": source_prior
                })

                curve.to_csv(
                    OUTPUT_DIR /
                    f"{variant}_seed_{seed}_cost_curve.csv",
                    index=False
                )

                pd.DataFrame(all_results).to_csv(
                    OUTPUT_DIR / "prior_calibration_results.csv",
                    index=False
                )

                pd.DataFrame([best]).to_csv(
                    OUTPUT_DIR /
                    f"{variant}_seed_{seed}_best_threshold.csv",
                    index=False
                )

                print(
                    f"\n{variant} | Seed {seed}"
                    f"\nSource prevalence: {source_prior:.4f}"
                    f"\nTarget prevalence: {TARGET_PREVALENCE:.4f}"
                    f"\nCorrected threshold: {best['threshold']:.3f}"
                    f"\nExpected cost/1000: "
                    f"{best['expected_cost_per_1000']:.2f}"
                    f"\nFN: {best['fn']} | FP: {best['fp']}"
                )

                del model, calibrator
                tf.keras.backend.clear_session()

                progress.update(1)

    # ==================================================
    # FINAL SUMMARY
    # ==================================================

    results_df = pd.DataFrame(all_results)

    summary = (
        results_df
        .groupby(["variant", "method"])[
            [
                "weighted_brier",
                "weighted_log_loss",
                "weighted_ece"
            ]
        ]
        .agg(["mean", "std"])
    )

    summary.to_csv(
        OUTPUT_DIR / "prior_calibration_summary.csv"
    )

    print("\n" + "=" * 65)
    print("PRIOR SHIFT CALIBRATION SUMMARY")
    print("=" * 65)

    print(summary.round(5).to_string())

    print("\nPRIOR SHIFT ANALYSIS COMPLETED!")
