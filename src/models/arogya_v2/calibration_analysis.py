import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import random
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    brier_score_loss,
    log_loss,
    average_precision_score,
    roc_auc_score,
)
from sklearn.calibration import calibration_curve


# ==================================================
# CONFIGURATION
# ==================================================

ROOT = Path(__file__).resolve().parents[3]

DATA_DIR = ROOT / "data/processed/primary/images/train"

CHECKPOINT_DIR = (
    ROOT / "experiments/version_2/ablation_checkpoints"
)

OUTPUT_DIR = (
    ROOT / "experiments/version_2/calibration"
)

FIGURE_DIR = OUTPUT_DIR / "figures"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

VARIANTS = ["A0", "A3"]
SEEDS = [42, 123, 2026]

IMAGE_SIZE = (128, 128)
BATCH_SIZE = 16
N_BINS = 10


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
# IMAGE PREPROCESSING
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

def probability_to_logit(probabilities):

    probabilities = np.clip(
        probabilities, 1e-7, 1 - 1e-7
    )

    return np.log(
        probabilities / (1 - probabilities)
    ).reshape(-1, 1)


def fit_platt_scaler(y_cal, p_cal):

    logits = probability_to_logit(p_cal)

    calibrator = LogisticRegression(
        C=1e6,
        solver="lbfgs",
        max_iter=2000
    )

    calibrator.fit(logits, y_cal)

    return calibrator


def apply_platt_scaler(calibrator, probabilities):

    logits = probability_to_logit(probabilities)

    return calibrator.predict_proba(logits)[:, 1]


# ==================================================
# EXPECTED CALIBRATION ERROR
# ==================================================

def calculate_ece(y_true, probabilities, n_bins=10):

    bin_edges = np.linspace(0, 1, n_bins + 1)

    ece = 0.0
    bin_results = []

    for i in range(n_bins):

        lower = bin_edges[i]
        upper = bin_edges[i + 1]

        if i == n_bins - 1:
            mask = (
                (probabilities >= lower)
                & (probabilities <= upper)
            )
        else:
            mask = (
                (probabilities >= lower)
                & (probabilities < upper)
            )

        if np.sum(mask) == 0:
            continue

        confidence = np.mean(probabilities[mask])
        observed = np.mean(y_true[mask])
        count = int(np.sum(mask))

        ece += (
            count / len(y_true)
        ) * abs(confidence - observed)

        bin_results.append({
            "bin_lower": lower,
            "bin_upper": upper,
            "count": count,
            "mean_confidence": confidence,
            "observed_frequency": observed
        })

    return ece, pd.DataFrame(bin_results)


# ==================================================
# CALIBRATION METRICS
# ==================================================

def evaluate_probabilities(y_true, probabilities):

    ece, bins = calculate_ece(
        y_true,
        probabilities,
        n_bins=N_BINS
    )

    metrics = {
        "brier_score": brier_score_loss(
            y_true, probabilities
        ),
        "log_loss": log_loss(
            y_true, probabilities,
            labels=[0, 1]
        ),
        "ece": ece,
        "pr_auc": average_precision_score(
            y_true, probabilities
        ),
        "roc_auc": roc_auc_score(
            y_true, probabilities
        )
    }

    return metrics, bins


# ==================================================
# RELIABILITY DIAGRAM
# ==================================================

def save_reliability_plot(
    y_true,
    raw_probabilities,
    calibrated_probabilities,
    title,
    save_path
):

    raw_fraction, raw_mean = calibration_curve(
        y_true,
        raw_probabilities,
        n_bins=N_BINS,
        strategy="uniform"
    )

    cal_fraction, cal_mean = calibration_curve(
        y_true,
        calibrated_probabilities,
        n_bins=N_BINS,
        strategy="uniform"
    )

    plt.figure(figsize=(7, 6))

    plt.plot(
        [0, 1], [0, 1],
        linestyle="--",
        label="Perfect calibration"
    )

    plt.plot(
        raw_mean,
        raw_fraction,
        marker="o",
        label="Original probabilities"
    )

    plt.plot(
        cal_mean,
        cal_fraction,
        marker="s",
        label="Platt calibrated"
    )

    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed pneumonia frequency")
    plt.title(title)

    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(save_path, dpi=200)
    plt.close()


# ==================================================
# MAIN CALIBRATION EXPERIMENT
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
        desc="Calibration experiments",
        unit="model"
    ) as progress:

        for seed in SEEDS:

            set_seed(seed)

            # Recreate the exact ablation validation split
            train_paths, val_paths, train_labels, val_labels = (
                train_test_split(
                    paths,
                    labels,
                    test_size=0.20,
                    stratify=labels,
                    random_state=seed
                )
            )

            # Split validation into calibration and evaluation.
            # Neither part is the original locked test set.
            cal_paths, eval_paths, cal_labels, eval_labels = (
                train_test_split(
                    val_paths,
                    val_labels,
                    test_size=0.50,
                    stratify=val_labels,
                    random_state=seed
                )
            )

            eval_ds = make_dataset(
                eval_paths,
                eval_labels
            )

            print(
                f"\nSeed {seed}: "
                f"Calibration={len(cal_paths)}, "
                f"Evaluation={len(eval_paths)}"
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

                # Calibration predictions
                cal_ds = make_dataset(
                    cal_paths,
                    cal_labels
                )

                p_cal = model.predict(
                    cal_ds,
                    verbose=0
                ).ravel()

                # Independent evaluation predictions
                p_eval = model.predict(
                    eval_ds,
                    verbose=0
                ).ravel()

                # Fit calibration only on calibration subset
                calibrator = fit_platt_scaler(
                    cal_labels,
                    p_cal
                )

                p_calibrated = apply_platt_scaler(
                    calibrator,
                    p_eval
                )

                # Evaluate before and after calibration
                raw_metrics, raw_bins = evaluate_probabilities(
                    eval_labels,
                    p_eval
                )

                calibrated_metrics, calibrated_bins = (
                    evaluate_probabilities(
                        eval_labels,
                        p_calibrated
                    )
                )

                # Save reliability diagram
                save_reliability_plot(
                    eval_labels,
                    p_eval,
                    p_calibrated,
                    f"{variant} | Seed {seed}",
                    FIGURE_DIR /
                    f"{variant}_seed_{seed}_reliability.png"
                )

                raw_bins.to_csv(
                    OUTPUT_DIR /
                    f"{variant}_seed_{seed}_raw_bins.csv",
                    index=False
                )

                calibrated_bins.to_csv(
                    OUTPUT_DIR /
                    f"{variant}_seed_{seed}_calibrated_bins.csv",
                    index=False
                )

                for method, metrics in [
                    ("raw", raw_metrics),
                    ("platt", calibrated_metrics)
                ]:

                    row = {
                        "variant": variant,
                        "seed": seed,
                        "method": method,
                        **metrics
                    }

                    all_results.append(row)

                print(
                    f"\n{variant} | Seed {seed}"
                    f"\nRaw Brier: "
                    f"{raw_metrics['brier_score']:.5f}"
                    f"\nPlatt Brier: "
                    f"{calibrated_metrics['brier_score']:.5f}"
                    f"\nRaw ECE: "
                    f"{raw_metrics['ece']:.5f}"
                    f"\nPlatt ECE: "
                    f"{calibrated_metrics['ece']:.5f}"
                )

                pd.DataFrame(all_results).to_csv(
                    OUTPUT_DIR / "calibration_results.csv",
                    index=False
                )

                del model, calibrator
                tf.keras.backend.clear_session()

                progress.update(1)

    # ==================================================
    # SUMMARY
    # ==================================================

    results_df = pd.DataFrame(all_results)

    metrics = [
        "brier_score",
        "log_loss",
        "ece",
        "pr_auc",
        "roc_auc"
    ]

    summary = (
        results_df
        .groupby(["variant", "method"])[metrics]
        .agg(["mean", "std"])
    )

    summary.to_csv(
        OUTPUT_DIR / "calibration_mean_std.csv"
    )

    print("\n" + "=" * 65)
    print("CALIBRATION SUMMARY")
    print("=" * 65)

    print(summary.round(5).to_string())

    print("\nCALIBRATION ANALYSIS COMPLETED!")
