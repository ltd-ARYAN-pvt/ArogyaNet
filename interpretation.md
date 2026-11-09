SHA-256 analysis identified 30 groups of exact duplicate files, comprising 34 duplicate pairs. All identified groups were confined to either the training or test split, with no exact duplicates detected across training, testing, or validation partitions. Perceptual hashing identified 26 train–test candidate pairs requiring further visual verification. Therefore, exact cross-split duplication was not observed, but potential near-duplicate overlap remains under investigation.

### Task 4: Shortcut Learning Investigation — 16×16 Resolution

A deliberately constrained CNN was trained using chest X-ray images downsampled to only 16×16 pixels. Despite the severe loss of spatial detail, the model achieved 89.08% accuracy, 80.69% balanced accuracy, and 0.988 PR-AUC.

The confusion matrix showed 98 false positives among 268 NORMAL images and 16 false negatives among 776 PNEUMONIA images. Pneumonia recall was 97.94%, whereas NORMAL recall was only 63.43%.

These results indicate that substantial class-discriminative information remains available even at extremely low resolution. This raises concerns that the model may exploit coarse image characteristics or dataset-specific acquisition patterns rather than relying exclusively on clinically meaningful lung abnormalities.

However, this experiment alone cannot establish the presence of shortcut learning. Further investigation is required to identify which image characteristics contribute to the predictions. The result should therefore be treated as a diagnostic warning, not proof of shortcut exploitation.

### Task 4: Shortcut Learning Investigation

A CNN trained on images downsampled to 16×16 pixels achieved 91.76% accuracy and 85.54% balanced accuracy. Its performance was compared against a majority-class baseline, which achieved 74.33% accuracy and 50% balanced accuracy by predicting PNEUMONIA for every sample.

The low-resolution CNN improved accuracy by 17.43 percentage points over the majority baseline. This demonstrates that substantial class-discriminative information survives extreme image downsampling.

However, the result raises concerns about possible reliance on coarse image characteristics, acquisition patterns, or dataset-specific artifacts rather than exclusively clinically meaningful abnormalities. Further experiments are necessary to identify the source of this information. High performance at low resolution is a warning signal, not definitive proof of shortcut learning.

### Task 5A: HOG + Logistic Regression Baseline

The HOG feature extraction pipeline converted 128×128 grayscale chest X-ray images into 8,100-dimensional feature vectors. These features were standardized and classified using Logistic Regression.

The model achieved 97.41% accuracy, 96.79% balanced accuracy, 98.45% precision, 98.07% recall, and 98.26% F1-score on the stratified validation subset.

The confusion matrix contained 256 true negatives, 12 false positives, 15 false negatives, and 761 true positives.

The HOG baseline substantially outperformed the 16×16 CNN and majority-class baseline in this experiment. This suggests that gradient-orientation and structural information provides highly discriminative signals in the dataset.

However, this result must be interpreted cautiously because the random validation split may contain related images, and patient-level independence has not yet been established. The result does not demonstrate clinical generalisation or prove that the model relies on clinically meaningful features.

### Task 5B: LBP + Logistic Regression Baseline

The LBP feature extraction method was applied to 128×128 grayscale chest X-ray images using a radius of 1, 8 sampling points, and the uniform pattern method. A global normalized histogram containing 10 features per image was generated and classified using Logistic Regression.

The model achieved 87.36% accuracy, 84.04% balanced accuracy, 92.04% precision, 90.85% recall, and 91.44% F1-score.

Compared with HOG, LBP performed substantially worse across all reported metrics. A possible explanation is that the global LBP histogram captures local texture frequencies but discards their spatial locations. HOG, in contrast, retains more spatially organized gradient-orientation information.

These findings suggest that edge and structural representations were more discriminative than the global texture representation in this experiment. However, the comparison is specific to the selected feature configurations and does not establish that HOG is universally superior to LBP.