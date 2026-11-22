# Stage 2 prompt — make training reproducible and establish baselines

Copy this prompt after Stage 1 is reviewed:

---

Do only Stage 2. Do not start preprocessing or augmentation searches and do not automatically move to Stage 3. Keep the exact 70/15/15 manifest and memberships from Stage 1. Do not calculate or use new test metrics.

## Goal

Create a reproducible local training path and establish honest baselines so later preprocessing and augmentation changes can be compared fairly.

## Read and preserve

Inspect Stage 1’s audit, current training code, model code, dataset loader, checkpoints, histories, prediction files, and the stated hardware. Use the existing local home system; do not send data to a cloud training service. Keep ResNet18 as the main 2D architecture.

The old baseline checkpoint used original intensity plus random rotation up to 10 degrees. Preserve that name/history. Add a separate raw_no_aug condition so “no augmentation” has an accurate meaning.

## Training pipeline requirements

1. Replace machine-specific paths with a single configurable project/data root. Make train, validation, and eventual evaluation use the same path-resolution logic.
2. Create one model factory using Torchvision ResNet18 with ImageNet pretrained weights and a four-output classifier head. Do not change backbone between conditions.
3. Use 224×224 inputs initially. Decode grayscale and adapt consistently for the pretrained model. Use identical resizing interpolation and normalization in training, validation, and inference.
4. Keep validation transforms deterministic. Augmentation must be switched on only for a training condition and must run online after splitting.
5. Fix class order in one shared definition. Derive loss weights from train counts only.
6. Add traceable stochastic training: record Python, NumPy, PyTorch, CUDA, DataLoader, and initialization seeds. Use three predeclared seeds such as 17, 42, and 2026. A seed is for comparison and traceability, not a guarantee of a particular accuracy. Report each run and aggregate results.
7. Save a versioned run manifest containing code version, dataset-manifest checksum, split counts, model, weights, preprocessing, augmentation, optimizer, learning rate, batch size, class weights, seed, software versions, best epoch, and checkpoint path.
8. Keep every old checkpoint and result. Write new outputs to results/experiments/<run_id>/.

## Baseline conditions

Train and clearly label:
- raw_no_aug: original grayscale with only shared resize and normalization;
- raw_rot10_legacy: original grayscale with the existing random rotation up to ±10°.

The legacy condition answers whether the earlier baseline behavior can be reproduced. The no-augmentation condition is the clean baseline for future augmentation comparisons.

## Baseline training loop

For each condition and each seed:

1. Start from the same pretrained weight version. Replace the final classifier layer for four classes.
2. Begin with the repository’s historical recipe: batch size 16, Adam learning rate 1e-4, train-only class-weighted cross-entropy, and up to eight epochs.
3. Record train and validation loss, accuracy, macro-F1, balanced accuracy, and per-class recall each epoch. Select checkpoints by validation macro-F1.
4. If the validation curve shows under-training or instability, run a small, predeclared training-only grid on raw_no_aug: learning rates 5e-5, 1e-4, and 2e-4 with the same optimizer family and a maximum of 20 epochs, early stopping patience 5. Select one common recipe by mean validation macro-F1 and freeze it before later arms.
5. Repeat seeds; report the mean and standard deviation, not only the best seed. Do not use the test split for early stopping, candidate selection, or threshold setting.

If the home GPU runs out of memory, lower batch size to 8 and use that setting for every future comparison. Do not switch to a remote machine or larger architecture to chase accuracy.

## Loop stop rule

Within a run, stop after five epochs without a validation macro-F1 improvement, or at the stated epoch maximum. Across the predeclared learning-rate grid, stop after the grid is complete or two successive settings fail to improve the best mean validation score. Do not expand the grid indefinitely.

## Deliverables

Provide the reusable trainer/config, baseline checkpoints per seed, histories, validation predictions, comparison table, and a short reproducibility report. Confirm raw_no_aug contains no random transformations. Do not open or score the test predictions. End and wait for the next manually pasted prompt.

## Official references

- PyTorch transfer learning: https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
- Torchvision pretrained models: https://docs.pytorch.org/vision/stable/models.html
- Project training code and historical result files are the evidence for the old recipe.