# ArogyaNet 🩺
### Author - Aryan Pandey

**Lightweight AI for Pediatric Pneumonia Detection**

ArogyaNet is a machine learning research project focused on developing an efficient deep learning system for detecting pneumonia from pediatric chest X-ray images.

The project explores the complete ML workflow — from dataset auditing and leakage investigation to traditional machine learning, transfer learning, custom CNN architecture design, and eventual offline deployment.

The long-term objective is to develop a compact, reliable model suitable for resource-constrained healthcare environments.

---

## 🎯 Project Objectives

- Investigate dataset quality, class imbalance, image characteristics, and duplicate images.
- Identify potential data leakage and dataset-specific shortcuts.
- Establish traditional ML and deep learning baselines.
- Explore transfer learning using pretrained CNN architectures.
- Design a custom lightweight CNN with fewer than 2 million parameters.
- Evaluate model performance beyond accuracy using clinically relevant metrics.
- Work towards offline CPU deployment with strict size and latency constraints.

## 📊 Dataset

**Primary Dataset:** Chest X-Ray Images (Pneumonia)

The dataset contains pediatric chest X-ray images divided into:

- NORMAL
- PNEUMONIA

| Split      |    NORMAL | PNEUMONIA |     Total |
| ---------- | --------: | --------: | --------: |
| Train      |     1,341 |     3,875 |     5,216 |
| Test       |       234 |       390 |       624 |
| Validation |         8 |         8 |        16 |
| **Total**  | **1,583** | **4,273** | **5,856** |

The locally extracted dataset contains 5,856 images, compared with 5,863 reported in the dataset description.

**Note:** Dataset files are excluded from this repository because of their size and distribution considerations.

## 🔬 Version 1 — Dataset Audit & Baselines

Version 1 covers Tasks 1–6 of the practical assignment.

### 1. Dataset Audit

- Image inventory and class distribution.
- Image dimensions, aspect ratios, file sizes, and pixel statistics.
- Brightness distribution analysis.
- Image mode and RGB channel investigation.

### 2. Duplicate & Leakage Investigation

- Exact duplicate detection using SHA-256.
- Near-duplicate detection using perceptual hashing.
- Cross-split duplicate candidate investigation.
- Visual inspection of selected near-duplicate candidates.

### 3. Validation Reliability

- Exact confidence interval analysis for the original 16-image validation set.
- Adoption of stratified train/validation evaluation for baseline experiments.
- Original test split retained for future final evaluation.

### 4. Shortcut Learning Investigation

- Deliberately downsampled 16×16 CNN.
- Comparison against majority-class baseline.
- Investigation of class-correlated signals surviving extreme downsampling.

### 5. Traditional Machine Learning Baselines

- HOG + Logistic Regression.
- LBP + Logistic Regression.
- Comparison using multiple classification metrics.

### 6. Transfer Learning

- MobileNetV3Small pretrained on ImageNet.
- Two-stage training: frozen feature extractor followed by fine-tuning.
- Three experimental seeds: 42, 123, and 2026.
- Mean and standard deviation reporting.

## 📈 Version 1 — Experimental Results

Internal validation results obtained using the stratified 80:20 split of the original training pool.

| Model                     |           Accuracy |
| ------------------------- | -----------------: |
| Majority Class Baseline   |             74.33% |
| 16×16 Shortcut CNN        |             91.76% |
| LBP + Logistic Regression |             87.36% |
| HOG + Logistic Regression |             97.41% |
| MobileNetV3Small          | **97.38% ± 0.31%** |

### MobileNetV3Small — Three-Seed Summary

| Metric            |      Mean ± SD |
| ----------------- | -------------: |
| Accuracy          | 97.38% ± 0.31% |
| Balanced Accuracy | 96.08% ± 0.08% |
| Precision         | 97.75% ± 0.27% |
| Recall            | 98.75% ± 0.71% |
| F1-score          | 98.25% ± 0.22% |
| PR-AUC            | 99.81% ± 0.05% |

**Important:** These are internal validation results, not independent clinical performance estimates. The three seeds use different stratified validation splits. Patient-level independence and generalisation require further investigation.

## 🏗️ Project Structure

```text
ArogyaNet/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── external/
│
├── src/
│   ├── audit/
│   ├── preprocessing/
│   ├── models/
│   ├── training/
│   ├── evaluation/
│   └── deployment/
│
├── experiments/
│   ├── version_1/
│   ├── version_2/
│   ├── version_3/
│   └── version_4/
│
├── reports/
│   └── part1/
│
├── figures/
│   └── part1/
│
├── checkpoints/
│
├── scripts/
│   └── activate_env.sh
│
├── pyproject.toml
├── uv.lock
├── .gitignore
└── README.md
```

## ⚙️ Tech Stack

- **Language:** Python
- **Deep Learning:** TensorFlow / Keras
- **Traditional ML:** Scikit-learn
- **Computer Vision:** OpenCV, Pillow, scikit-image
- **Data Processing:** NumPy, Pandas
- **Experimentation:** tqdm, MLflow, Optuna
- **Environment Management:** uv
- **Development Environment:** VS Code + WSL2 Ubuntu

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/ltd-ARYAN-pvt/ArogyaNet.git
cd ArogyaNet
```

### 2. Install dependencies

```bash
uv sync
```

### 3. Activate the environment

```bash
source .venv/bin/activate
```

### 4. Configure GPU libraries (WSL2)

If required by your TensorFlow environment:

```bash
export LD_LIBRARY_PATH="$(uv run python -c 'import nvidia.cusolver, pathlib; print(pathlib.Path(next(iter(nvidia.cusolver.__path__))) / 'lib')'):/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

Alternatively, use the project activation script if configured.

### 5. Run experiments

```bash
python src/audit/dataset_inventory.py
python src/models/shortcut_cnn.py
python src/models/hog_baseline.py
python src/models/lbp_baseline.py
python src/models/transfer_learning.py
```

Place the primary dataset in the expected project data directory before running the experiments.

## 🛣️ Roadmap

### Version 1 — Dataset Audit & Baselines

- [x] Dataset inventory and pixel statistics
- [x] Duplicate and leakage investigation
- [x] Validation reliability analysis
- [x] 16×16 shortcut experiment
- [x] Majority, HOG, and LBP baselines
- [x] MobileNetV3Small transfer learning

### Version 2 — ArogyaNet Architecture

- [x] Design a custom CNN with fewer than 2M parameters
- [x] Implement depthwise separable convolutions
- [x] Explore residual connections
- [x] Implement Squeeze-and-Excitation attention
- [x] Conduct architecture ablations
- [x] Evaluate performance across three seeds

### Version 3 — Robustness & Model Optimization

- [x] Address class imbalance and asymmetric error costs
- [ ] Cross-validation and threshold optimization
- [ ] Regularization and ensemble experiments
- [ ] Feature-space analysis
- [ ] External dataset evaluation and distribution-shift investigation

### Version 4 — Deployment & Final Evaluation

- [ ] Model compression and optimization
- [ ] CPU inference benchmarking
- [ ] Offline prediction pipeline
- [ ] Interactive demonstration interface
- [ ] Robustness and failure-case testing
- [ ] Final technical report

## ⚠️ Intended Use

ArogyaNet is an educational and research project. It is not a clinically validated medical device and must not be used as a substitute for professional medical diagnosis.

## 👨‍💻 Project Status

**Current Stage:** Version 2 completed — preparing Version 3.

The project is being developed incrementally with an emphasis on reproducibility, transparent evaluation, model efficiency, and responsible AI for healthcare.

---

*Built as part of an applied machine learning research project.*
