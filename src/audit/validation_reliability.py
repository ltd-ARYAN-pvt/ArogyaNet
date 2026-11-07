from pathlib import Path
import pandas as pd
from scipy.stats import beta

OUTPUT = Path("reports/part1")
OUTPUT.mkdir(parents=True, exist_ok=True)

# Original validation set
n = 16
correct = 16
errors = n - correct
observed_accuracy = correct / n

# Exact two-sided Clopper-Pearson confidence interval
alpha = 0.05

if correct == n:
    lower = (alpha / 2) ** (1 / n)
    upper = 1.0

elif correct == 0:
    lower = 0.0
    upper = 1 - (alpha / 2) ** (1 / n)

else:
    lower = beta.ppf(alpha / 2, correct, errors + 1)
    upper = beta.ppf(1 - alpha / 2, correct + 1, errors)

print("Original validation set")
print("Total images:", n)
print("Correct predictions:", correct)
print("Observed accuracy:", f"{observed_accuracy:.2%}")
print("95% exact CI:", f"{lower:.2%} - {upper:.2%}")

# Proposed evaluation strategy
inventory = pd.read_csv(OUTPUT / "image_inventory.csv")

train_pool = inventory[inventory["split"] == "train"]

print("\nProposed evaluation strategy")
print("Development pool:", len(train_pool))
print("\nClass distribution:")
print(train_pool["label"].value_counts())

print("\nOriginal untouched test set:")
test_pool = inventory[inventory["split"] == "test"]
print("Total:", len(test_pool))
print(test_pool["label"].value_counts())

# Save results
results = pd.DataFrame([{
    "validation_images": n,
    "correct_predictions": correct,
    "observed_accuracy": observed_accuracy,
    "ci_95_lower": lower,
    "ci_95_upper": upper
}])

results.to_csv(
    OUTPUT / "validation_reliability.csv",
    index=False
)

print("\nSaved validation_reliability.csv")