# BrainScope-XAI accuracy improvement prompt pack

## Clear objective

Use the current MRI images to run honest, controlled preprocessing, augmentation, and training experiments that may improve classification performance. Treat accuracy above 99% as an aspiration, not a promised result. Do not force, fabricate, or selectively report a score. The agent must stop when the predefined validation-led experiment loop is complete, summarize the best result, and wait for the user to paste the next stage prompt.

The user will paste every stage prompt manually. **Never automatically start the next stage.** A loop means repeat the experiments named inside the current stage, compare them, record the best validation result, and stop at that stage’s boundary.

## Current project facts to verify before acting

- The repository has 6,400 2D image files: 896 Mild, 64 Moderate, 3,200 Non Demented, and 2,240 Very Mild.
- The current split is 4,480 train, 960 validation, and 960 test. Preserve the 70/15/15 ratio and every existing image’s split membership exactly.
- The split manifest and several scripts still contain old C:\HEHAW\BrainX paths. Images are now under this repository’s Dataset folder. Repair path resolution only; never regenerate or reassign split rows.
- The existing “baseline” training loader applies RandomRotation up to 10 degrees to all training conditions. It is a raw-intensity plus rotation baseline, not a no-augmentation baseline.
- Saved results are baseline 906/960 = 94.38% accuracy and 95.30% macro-F1; CLAHE 913/960 = 95.10% accuracy and 94.61% macro-F1; HE 751/960 = 78.23% accuracy and 79.76% macro-F1. These are saved, image-level results; the underlying run is not reproducible until data paths and experiment records are repaired.
- The existing test results have already been inspected and CLAHE was chosen after comparing them. Do not describe another run on the same test images as an untouched confirmation.
- Grad-CAM files and code already exist under src/explainability and results/gradcam. Leave them alone. This plan is for accuracy, preprocessing, and augmentation only.

## Non-negotiable experiment rules

1. Preserve the current 70/15/15 split assignments exactly. Do not regenerate, rebalance, or move records.
2. Run model fitting on train only. Use validation to select epochs and candidate recipes. Keep test results hidden during every experiment loop. The current test has already been consulted, so disclose that limitation in the final report.
3. Use multiple training seeds to measure stochastic variation. A seed makes a run traceable; it does not guarantee a high score. Report each seed and the mean/spread. Do not select the luckiest run as the headline.
4. Use a bounded list of candidate experiments. Continue the current stage’s loop until its candidates are completed or the stated early-stop rule fires. Never invent more transformations to chase 99.99%.
5. Always report accuracy, macro-F1, balanced accuracy, per-class precision/recall/F1, confusion matrix, and counts. Accuracy alone can hide failure on the 64-image Moderate class.
6. Never delete difficult images, alter labels, copy augmented images into validation/test, use test data for calibration/tuning, or hide failed experiments.
7. Keep existing checkpoints and results. Write new run artifacts to versioned folders under results/experiments/.
8. ResNet18 remains the home-system model unless the user explicitly changes that decision.
9. At each stage’s end, summarize code/results, report the best validation-selected candidate and limitations, then stop and wait for the next manually supplied prompt.

## Stage order

1. [Audit data and repair paths without changing the split](01_data_and_split_audit.md)
2. [Make training reproducible and establish baselines](02_reproducible_training_baseline.md)
3. [Run the preprocessing experiment loop](03_preprocessing_ablation.md)
4. [Run the augmentation experiment loop](04_augmentation_ablation.md)
5. [Compare imbalance and training choices](05_imbalance_and_training_stability.md)
6. [Optimize the combined pipeline on validation](06_validation_led_pipeline_optimization.md)
7. [Evaluate the frozen pipeline and report accuracy](07_final_accuracy_evaluation.md)

## Where the agent should get evidence

First inspect the repository’s README, code, split CSV, predictions, reports, histories, and checkpoints. Then inspect the user-provided notes and the pinned chat “Review papers and plan slides.” Verify paper claims from the original PDF, not from the notes alone.

Use primary papers and official documentation for technical choices:

- MRI split leakage study: https://www.nature.com/articles/s41598-021-01681-w
- CLAIM 2024 medical-imaging reporting guide: https://pubs.rsna.org/doi/10.1148/ryai.240300
- Official PyTorch transfer-learning tutorial: https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
- Torchvision model and pretrained-weight docs: https://docs.pytorch.org/vision/stable/models.html
- OpenCV CLAHE API: https://docs.opencv.org/4.x/d6/db6/classcv_1_1CLAHE.html
- PyTorch image transforms: https://docs.pytorch.org/vision/stable/transforms.html
- Official scikit-learn classification metrics: https://scikit-learn.org/stable/modules/model_evaluation.html

Do not copy a high accuracy from another paper as a target expectation. Different datasets and split units are not directly comparable.