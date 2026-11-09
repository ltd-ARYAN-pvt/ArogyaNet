
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
from sklearn.metrics import confusion_matrix


# ==================================================
# CONFIGURATION
# ==================================================

ROOT = Path(__file__).resolve().parents[3]

DATA_DIR = ROOT / "data/processed/primary/images/train"

CHECKPOINT_DIR = (
    ROOT / "experiments/version_2/ablation_checkpoints"
)

OUTPUT_DIR = (
    ROOT / "experiments/version_2/final_threshold_analysis"
)

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

    return np.array(paths), np.array(labels)


# ==================================================
# IMAGE PIPELINE
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

    p = np.clip(
        probabilities,
        1e-7,
        1 - 1e-7
    )

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
# PRIOR CORRECTION
# ==================================================

def correct_prior(
    probabilities,
    source_prior,
    target_prior=TARGET_PREVALENCE
):

    p = np.clip(
        probabilities,
        1e-7,
        1 - 1e-7
    )

    source_odds = source_prior / (1 - source_prior)
    target_odds = target_prior / (1 - target_prior)

    multiplier = target_odds / source_odds

    corrected_odds = (
        p / (1 - p)
    ) * multiplier

    return corrected_odds / (1 + corrected_odds)


# ==================================================
# COST FUNCTION
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


def select_threshold(y_true, probabilities):

    tuning_results = []

    for threshold in THRESHOLDS:

        result = evaluate_cost(
            y_true,
            probabilities,
            threshold
        )

        tuning_results.append(result)

    curve = pd.DataFrame(tuning_results)

    best_index = curve[
        "expected_cost_per_1000"
    ].idxmin()

    best = curve.loc[best_index].to_dict()

    return best, curve


# ==================================================
# MAIN EXPERIMENT
# ==================================================

if __name__ == "__main__":

    print("TensorFlow:", tf.__version__)
    print("GPU:", tf.config.list_physical_devices("GPU"))

    paths, labels = collect_paths()

    all_results = []

    total_runs = len(VARIANTS) * len(SEEDS)

    with tqdm(
        total=total_runs,
        desc="Final threshold comparison",
        unit="model"
    ) as progress:

        for seed in SEEDS:

            set_seed(seed)

            # Recreate original ablation validation split
            _, val_paths, _, val_labels = train_test_split(
                paths,
                labels,
                test_size=0.20,
                stratify=labels,
                random_state=seed
            )

            # First: 50% calibration, 50% remaining
            cal_paths, remaining_paths, cal_labels, remaining_labels = (
                train_test_split(
                    val_paths,
                    val_labels,
                    test_size=0.50,
                    stratify=val_labels,
                    random_state=seed
                )
            )

            # Second: remaining split into 25% tuning, 25% evaluation
            tune_paths, eval_paths, tune_labels, eval_labels = (
                train_test_split(
                    remaining_paths,
                    remaining_labels,
                    test_size=0.50,
                    stratify=remaining_labels,
                    random_state=seed
                )
            )

            source_prior = float(np.mean(cal_labels))

            print(
                f"\nSeed {seed}: "
                f"Calibration={len(cal_paths)}, "
                f"Tuning={len(tune_paths)}, "
                f"Evaluation={len(eval_paths)}, "
                f"Source prior={source_prior:.4f}"
            )

            cal_ds = make_dataset(cal_paths, cal_labels)
            tune_ds = make_dataset(tune_paths, tune_labels)
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

                # Generate predictions
                p_cal_raw = model.predict(
                    cal_ds,
                    verbose=0
                ).ravel()

                p_tune_raw = model.predict(
                    tune_ds,
                    verbose=0
                ).ravel()

                p_eval_raw = model.predict(
                    eval_ds,
                    verbose=0
                ).ravel()

                # Fit Platt only on calibration subset
                calibrator = fit_platt(
                    cal_labels,
                    p_cal_raw
                )

                p_tune_platt = apply_platt(
                    calibrator,
                    p_tune_raw
                )

                p_eval_platt = apply_platt(
                    calibrator,
                    p_eval_raw
                )

                # Apply target prior correction
                p_tune_corrected = correct_prior(
                    p_tune_platt,
                    source_prior
                )

                p_eval_corrected = correct_prior(
                    p_eval_platt,
                    source_prior
                )

                methods = {
                    "platt_source": (
                        p_tune_platt,
                        p_eval_platt
                    ),
                    "prior_corrected": (
                        p_tune_corrected,
                        p_eval_corrected
                    )
                }

                for method, (
                    tune_probabilities,
                    eval_probabilities
                ) in methods.items():

                    # Select threshold on tuning subset only
                    best, curve = select_threshold(
                        tune_labels,
                        tune_probabilities
                    )

                    selected_threshold = best["threshold"]

                    # Evaluate selected threshold on separate subset
                    independent_result = evaluate_cost(
                        eval_labels,
                        eval_probabilities,
                        selected_threshold
                    )

                    # Also evaluate default 0.5 threshold
                    default_result = evaluate_cost(
                        eval_labels,
                        eval_probabilities,
                        0.5
                    )

                    result = {
                        "variant": variant,
                        "seed": seed,
                        "method": method,
                        "parameters": model.count_params(),
                        "source_prevalence": source_prior,
                        "target_prevalence": TARGET_PREVALENCE,
                        "selected_threshold": selected_threshold,
                        "tuning_cost_per_1000":
                            best["expected_cost_per_1000"],
                        "evaluation_cost_per_1000":
                            independent_result["expected_cost_per_1000"],
                        "default_cost_per_1000":
                            default_result["expected_cost_per_1000"],
                        "tn": independent_result["tn"],
                        "fp": independent_result["fp"],
                        "fn": independent_result["fn"],
                        "tp": independent_result["tp"],
                        "sensitivity": independent_result["sensitivity"],
                        "specificity": independent_result["specificity"]
                    }

                    all_results.append(result)

                    curve.to_csv(
                        OUTPUT_DIR /
                        f"{variant}_{method}_seed_{seed}_tuning_curve.csv",
                        index=False
                    )

                    pd.DataFrame(all_results).to_csv(
                        OUTPUT_DIR / "final_threshold_results.csv",
                        index=False
                    )

                    print(
                        f"\n{variant} | {method} | Seed {seed}"
                        f"\nSelected threshold: {selected_threshold:.3f}"
                        f"\nTuning cost/1000: "
                        f"{result['tuning_cost_per_1000']:.2f}"
                        f"\nIndependent evaluation cost/1000: "
                        f"{result['evaluation_cost_per_1000']:.2f}"
                        f"\nDefault evaluation cost/1000: "
                        f"{result['default_cost_per_1000']:.2f}"
                        f"\nFN={result['fn']} FP={result['fp']}"
                    )

                del model, calibrator
                tf.keras.backend.clear_session()

                progress.update(1)

    # ==================================================
    # SUMMARY
    # ==================================================

    results_df = pd.DataFrame(all_results)

    summary_metrics = [
        "selected_threshold",
        "tuning_cost_per_1000",
        "evaluation_cost_per_1000",
        "default_cost_per_1000",
        "sensitivity",
        "specificity"
    ]

    summary = (
        results_df
        .groupby(["variant", "method"])[summary_metrics]
        .agg(["mean", "std"])
    )

    summary.to_csv(
        OUTPUT_DIR / "final_threshold_summary.csv"
    )

    print("\n" + "=" * 70)
    print("FINAL COST-SENSITIVE THRESHOLD SUMMARY")
    print("=" * 70)

    print(summary.round(4).to_string())

    print("\nFINAL THRESHOLD ANALYSIS COMPLETED!")
