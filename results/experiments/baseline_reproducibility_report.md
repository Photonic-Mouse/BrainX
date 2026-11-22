# Stage 2 Baseline Reproducibility Report

**Execution Date:** 2026-11-22  
**Platform / Hardware:** Local Home System — NVIDIA GeForce RTX 3050 Laptop GPU (4.0 GB VRAM, Compute 8.6, CUDA 12.4)  
**PyTorch Version:** 2.6.0+cu124  
**Torchvision Version:** 0.21.0+cu124  
**Git Commit:** `104e3f67f020c76bfd15bf1ac44c656fe78d474e`  
**Dataset Manifest Checksum:** `d11d3a62e955e4631bffd409285f6ce966059c39ecaf1f1f4bbd6d4ed0b133ab`  

---

## 1. Objective & Scope

The goal of Stage 2 is to create a fully reproducible local training pipeline and establish honest baselines across multiple random seeds, enabling rigorous comparisons for subsequent preprocessing and augmentation stages.

**Key Guardrails Followed:**
- **Zero Test Leakage:** The held-out test split ($N = 960$) was **strictly kept blinded**. No test images were loaded, evaluated, or scored during any phase of training, early stopping, or checkpoint selection.
- **Split & Membership Fidelity:** Preserved the exact 70/15/15 stratified manifest (4,480 train / 960 validation).
- **Home Hardware Execution:** Enabled the local NVIDIA GeForce RTX 3050 Laptop GPU via PyTorch CUDA 12.4. Zero data was sent to external cloud services.
- **Traceable Seeds:** Evaluated across three predeclared seeds: `17`, `42`, and `2026`.

---

## 2. Experimental Setup & Baseline Conditions

Two distinct baseline conditions were trained using the identical backbone (`ResNet18` with ImageNet weights `ResNet18_Weights.DEFAULT`, custom 4-class linear head `nn.Linear(512, 4)`):

### Condition 1: `raw_no_aug` (Clean Baseline)
- **Preprocessing:** Original intensity (8-bit grayscale converted to 3-channel RGB).
- **Training Transformations:** `Resize((224, 224))`, `ToTensor()`, ImageNet `Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])`.
- **Explicit Confirmation:** **Contains ZERO random transformations**. No rotations, flips, crops, or distortions.
- **Validation Transformations:** Identical deterministic resize and normalization.

### Condition 2: `raw_rot10_legacy` (Legacy Replication Baseline)
- **Preprocessing:** Original intensity.
- **Training Transformations:** Identical to above plus `transforms.RandomRotation(10)` (the transformation hardcoded in the repository's original `train.py`).
- **Validation Transformations:** Identical deterministic resize and normalization.

### Shared Hyperparameters (Frozen Historical Recipe)
- **Optimizer:** Adam ($\beta_1 = 0.9, \beta_2 = 0.999$, $\epsilon = 10^{-8}$)
- **Learning Rate:** $1 \times 10^{-4}$
- **Batch Size:** 16 (peak VRAM: 1.6 GB, well within the 4.0 GB budget)
- **Class Weights:** Derived strictly from training set counts:
  $$w_{\text{Mild}} = 1.786, \quad w_{\text{Moderate}} = 24.889, \quad w_{\text{Non}} = 0.500, \quad w_{\text{VeryMild}} = 0.714$$
- **Max Epochs:** 8 (Early stopping patience: 5 on validation Macro-F1)
- **Epoch Training Duration:** $\sim 28.5$ seconds per epoch on the RTX 3050.

---

## 3. Individual Run Results (Validation Split, $N = 960$)

| Condition | Seed | Best Epoch | Val Accuracy | Val Balanced Acc | Val Macro F1 | Val Weighted F1 | Moderate Recall (N=10) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **raw_no_aug** | 17 | 5 | 96.35% | 96.12% | **96.75%** | 96.34% | 100.0% (10/10) |
| **raw_no_aug** | 42 | 6 | 95.10% | 96.75% | **95.87%** | 95.12% | 100.0% (10/10) |
| **raw_no_aug** | 2026 | 4 | 95.00% | 96.18% | **96.01%** | 94.98% | 100.0% (10/10) |
| **raw_rot10_legacy** | 17 | 8 | 98.23% | 98.26% | **98.64%** | 98.23% | 100.0% (10/10) |
| **raw_rot10_legacy** | 42 | 5 | 93.75% | 93.11% | **94.18%** | 93.71% | 90.0% (9/10) |
| **raw_rot10_legacy** | 2026 | 7 | 96.46% | 97.24% | **96.03%** | 96.45% | 100.0% (10/10) |

---

## 4. Aggregated Statistical Performance (Mean $\pm$ Standard Deviation)

| Metric | `raw_no_aug` (Clean Baseline) | `raw_rot10_legacy` (Legacy Baseline) | Delta ($\Delta$) |
| :--- | :---: | :---: | :---: |
| **Validation Accuracy** | $95.49\% \pm 0.75\%$ | $96.15\% \pm 2.26\%$ | $+0.66\%$ |
| **Validation Balanced Accuracy** | $96.35\% \pm 0.35\%$ | $96.21\% \pm 2.73\%$ | $-0.14\%$ |
| **Validation Macro F1** | **$96.21\% \pm 0.47\%$** | **$96.28\% \pm 2.24\%$** | $+0.07\%$ |
| **Validation Weighted F1** | $95.48\% \pm 0.75\%$ | $96.13\% \pm 2.28\%$ | $+0.65\%$ |
| **Moderate Class Recall** | **$100.0\% \pm 0.00\%$** | $96.67\% \pm 5.77\%$ | $-3.33\%$ |

---

## 5. Key Empirical Observations

1. **Clean Baseline Reproducibility (`raw_no_aug`):**
   * Achieving a mean validation Macro-F1 of **$96.21\% \pm 0.47\%$** without any data augmentation establishes a solid and highly repeatable benchmark.
   * Variance across random initialization seeds is remarkably low ($\sigma = 0.47\%$), proving the stability of the transfer-learning backbone on this cohort.
   * The rare 10-sample validation *Moderate Demented* class achieved $100\%$ sensitivity across all three seeds ($10/10$), demonstrating that class-weighted cross-entropy successfully neutralizes majority-class bias.

2. **Legacy Rotation Impact (`raw_rot10_legacy`):**
   * Replicates the repository's original training behavior. While it achieved an exceptional run on Seed 17 (98.64% Macro-F1), it exhibited **nearly 5 times higher variance across seeds** ($\sigma = 2.24\%$ vs. $0.47\%$), with Seed 42 falling to $94.18\%$.
   * On Seed 42, Moderate recall dropped to $90\%$ (1 false negative misclassified as Mild).

3. **Convergence Dynamics:**
   * Both conditions converge rapidly within $4 - 8$ epochs. Training loss drops from $\sim 0.82$ in epoch 1 to $\sim 0.05$ in epoch 8.
   * No under-training was observed; the historical hyperparameter recipe (Adam, $\text{lr} = 10^{-4}$, batch size 16) is well-calibrated and frozen as the standard execution recipe for subsequent stages.

---

## 6. Generated Stage 2 Artifacts

All run outputs have been organized into versioned experiment directories:
- Reusable training pipeline: [`src/training/stage2_baseline.py`](file:///c:/SHARAN%20PROJECTS/BRAINSCOPE-XAI/src/training/stage2_baseline.py)
- Run artifacts (checkpoints, histories, validation predictions, and manifests):
  - `results/experiments/raw_no_aug_seed17/`
  - `results/experiments/raw_no_aug_seed42/`
  - `results/experiments/raw_no_aug_seed2026/`
  - `results/experiments/raw_rot10_legacy_seed17/`
  - `results/experiments/raw_rot10_legacy_seed42/`
  - `results/experiments/raw_rot10_legacy_seed2026/`
- Aggregated comparison data:
  - [`results/experiments/baseline_comparison_summary.csv`](file:///c:/SHARAN%20PROJECTS/BRAINSCOPE-XAI/results/experiments/baseline_comparison_summary.csv)
  - [`results/experiments/baseline_aggregates.csv`](file:///c:/SHARAN%20PROJECTS/BRAINSCOPE-XAI/results/experiments/baseline_aggregates.csv)

---

## 7. Next-Stage Prerequisites (Stage 3)

With reproducible, multi-seed baselines firmly established:
- [x] Clean zero-augmentation baseline quantified: **$95.49\% \pm 0.75\%$ accuracy / $96.21\% \pm 0.47\%$ Macro-F1**.
- [x] Legacy rotation baseline quantified: **$96.15\% \pm 2.26\%$ accuracy / $96.28\% \pm 2.24\%$ Macro-F1**.
- [x] Training recipe frozen: ResNet18, Adam, $\text{lr} = 10^{-4}$, batch size 16, 8 epochs, seeds `17, 42, 2026`.
- [x] Test split strictly unread and blinded.

*Stage 2 complete. Ready and waiting for the user's Stage 3 prompt.*
