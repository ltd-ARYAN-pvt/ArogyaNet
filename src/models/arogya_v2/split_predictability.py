
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

ROOT = Path(__file__).resolve().parents[3]

DATA_FILE = (
    ROOT / "experiments/version_2/dataset_shift/"
    "image_level_statistics.csv"
)

OUTPUT_DIR = ROOT / "experiments/version_2/dataset_shift"

FEATURES = [
    "width",
    "height",
    "aspect_ratio",
    "brightness_mean",
    "brightness_std",
    "file_size_kb",
]

def main():

    df = pd.read_csv(DATA_FILE)

    df["is_test"] = (df["split"] == "test").astype(int)

    results = []

    for class_name in ["NORMAL", "PNEUMONIA"]:

        subset = df[df["class"] == class_name].copy()

        X = subset[FEATURES]
        y = subset["is_test"]

        cv = StratifiedKFold(
            n_splits=5,
            shuffle=True,
            random_state=42
        )

        models = {
            "Logistic Regression": make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced"
                )
            ),

            "Random Forest": RandomForestClassifier(
                n_estimators=300,
                max_depth=6,
                min_samples_leaf=5,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1
            )
        }

        print(f"\n{'=' * 60}")
        print(f"CLASS: {class_name}")
        print("Train images:", int((y == 0).sum()))
        print("Test images:", int((y == 1).sum()))

        for model_name, model in models.items():

            scores = cross_validate(
                model,
                X,
                y,
                cv=cv,
                scoring={
                    "auc": "roc_auc",
                    "balanced_accuracy": "balanced_accuracy"
                },
                n_jobs=-1
            )

            auc = scores["test_auc"]
            balanced_acc = scores["test_balanced_accuracy"]

            results.append({
                "class": class_name,
                "model": model_name,
                "auc_mean": auc.mean(),
                "auc_std": auc.std(),
                "balanced_accuracy_mean": balanced_acc.mean(),
                "balanced_accuracy_std": balanced_acc.std()
            })

            print(f"\n{model_name}")
            print(f"Split AUC: {auc.mean():.4f} ± {auc.std():.4f}")
            print(
                "Balanced accuracy:",
                f"{balanced_acc.mean():.4f} ± {balanced_acc.std():.4f}"
            )

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        OUTPUT_DIR / "split_predictability_results.csv",
        index=False
    )

    print("\nSaved split predictability results.")


if __name__ == "__main__":
    main()