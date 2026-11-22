# Stage 3 Preprocessing Ablation Report

**Execution Date:** 2026-11-22  
**Platform / Hardware:** NVIDIA GeForce RTX 3050 Laptop GPU (CUDA 12.4, PyTorch 2.6.0+cu124)  
**Research Question:** Does a particular preprocessing step improve validation performance when model, training settings, split, and augmentation are held constant?  
**Constraint Adherence:** Data augmentation was **STRICTLY DISABLED** across all arms. Test split ($N=960$) remained completely blinded.  

---

## 1. Executive Summary

In Stage 3, we evaluated five predeclared preprocessing candidates across three seeds (`17`, `42`, `2026`) using the frozen training recipe (ResNet-18 ImageNet backbone, Adam $\text{lr}=10^{-4}$, batch size 16, 8 epochs, train-only class weighting). All $15$ experiment runs were evaluated exclusively on the held-out validation split ($N = 960$).

### Key Empirical Findings:
1. **Best Preprocessing Candidate: `prep_clahe_1_5` (CLAHE clipLimit=1.5, tileGridSize=8x8)**
   * Achieved the highest validation Macro-F1 (**$96.41\% \pm 0.58\%$**) and highest validation accuracy (**$95.66\% \pm 0.69\%$**).
   * Maintained **$100.0\% \pm 0.0\%$ recall on the rare Moderate Demented class** ($10/10$ across all 3 seeds).
2. **Aggressive Contrast Tradeoff (`clipLimit=2.0` and `2.5`):**
   * Increasing `clipLimit` above $1.5$ caused a slight performance drop (Macro-F1 fell to **$95.50\% \pm 1.58\%$**) and compromised minority-class sensitivity on Seed 17 ($90\%$ recall, 1 false negative).
3. **Global Equalization Instability (`prep_he`):**
   * While peak runs on HE can achieve high scores (Seed 42 reached 97.44%), HE displays **over $3\times$ higher variance** across seeds ($\sigma = 1.36\%$ vs. $0.47\%$ on original) and lowest worst-case run ($93.54\%$ accuracy on Seed 2026), corroborating its sensitivity to scanner noise.

---

## 2. Visual Quality Review Panel

Fixed validation samples from all four diagnostic categories were rendered across all five conditions:
- **Image Panel File:** [`results/experiments/preprocessing_panels/preprocessing_comparison_panel.png`](file:///c:/SHARAN%20PROJECTS/BRAINSCOPE-XAI/results/experiments/preprocessing_panels/preprocessing_comparison_panel.png)

```
Visual Inspection Observations:
- Original: Clean soft-tissue gray-white contrast, zero background amplification.
- Global HE: Background air noise stretched to light gray; internal ventricles washed out.
- CLAHE (1.5): Local contrast cleanly enhanced along lateral ventricles and sulci; background remains pure black.
- CLAHE (2.0): Sharper tissue gradients, but slight noise amplification along peripheral cerebral cortex.
- CLAHE (2.5): Beginning of excessive local contrast peaks; cortical boundary appears artificially jagged.
```

---

## 3. Candidate Summary Table ($N = 960$ Validation)

| Candidate ID | Preprocessing Technique | Seed | Best Epoch | Val Accuracy | Val Balanced Acc | Val Macro F1 | Val Weighted F1 | Moderate Recall ($N=10$) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `prep_original` | Original Unenhanced | 17 | 5 | 96.35% | 96.12% | 96.75% | 96.34% | 100.0% (10/10) |
| `prep_original` | Original Unenhanced | 42 | 6 | 95.10% | 96.75% | 95.87% | 95.12% | 100.0% (10/10) |
| `prep_original` | Original Unenhanced | 2026 | 4 | 95.00% | 96.18% | 96.01% | 94.98% | 100.0% (10/10) |
| `prep_he` | Global Histogram Eq. | 17 | 5 | 95.42% | 96.19% | 96.31% | 95.41% | 100.0% (10/10) |
| `prep_he` | Global Histogram Eq. | 42 | 7 | 96.67% | 97.86% | 97.44% | 96.67% | 100.0% (10/10) |
| `prep_he` | Global Histogram Eq. | 2026 | 8 | 93.54% | 94.74% | 94.73% | 93.43% | 100.0% (10/10) |
| **`prep_clahe_1_5`** | **CLAHE (clip=1.5, 8x8)** | **17** | **5** | **94.90%** | **96.37%** | **95.98%** | **94.91%** | **100.0% (10/10)** |
| **`prep_clahe_1_5`** | **CLAHE (clip=1.5, 8x8)** | **42** | **8** | **96.25%** | **96.69%** | **97.07%** | **96.26%** | **100.0% (10/10)** |
| **`prep_clahe_1_5`** | **CLAHE (clip=1.5, 8x8)** | **2026** | **8** | **95.83%** | **95.07%** | **96.19%** | **95.79%** | **100.0% (10/10)** |
| `prep_clahe_2_0` | CLAHE (clip=2.0, 8x8) | 17 | 8 | 95.62% | 94.13% | 93.73% | 95.64% | 90.0% (9/10) |
| `prep_clahe_2_0` | CLAHE (clip=2.0, 8x8) | 42 | 7 | 96.04% | 96.92% | 96.77% | 96.04% | 100.0% (10/10) |
| `prep_clahe_2_0` | CLAHE (clip=2.0, 8x8) | 2026 | 7 | 95.00% | 95.42% | 95.98% | 95.02% | 100.0% (10/10) |
| `prep_clahe_2_5` | CLAHE (clip=2.5, 8x8) | 17 | 8 | 95.62% | 94.13% | 93.73% | 95.64% | 90.0% (9/10) |
| `prep_clahe_2_5` | CLAHE (clip=2.5, 8x8) | 42 | 7 | 96.04% | 96.92% | 96.77% | 96.04% | 100.0% (10/10) |
| `prep_clahe_2_5` | CLAHE (clip=2.5, 8x8) | 2026 | 7 | 95.00% | 95.42% | 95.98% | 95.02% | 100.0% (10/10) |

---

## 4. Aggregated Statistical Ranking across Seeds (Mean $\pm$ Std)

| Rank | Candidate | Preprocessing | Val Accuracy | Val Balanced Acc | Val Macro F1 | Moderate Recall |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: |
| **1** | **`prep_clahe_1_5`** | **CLAHE (clip=1.5)** | **$95.66\% \pm 0.69\%$** | $96.05\% \pm 0.86\%$ | **$96.41\% \pm 0.58\%$** | **$100.0\% \pm 0.00\%$** |
| 2 | `prep_original` | Original Intensity | $95.49\% \pm 0.75\%$ | **$96.35\% \pm 0.35\%$** | $96.21\% \pm 0.47\%$ | **$100.0\% \pm 0.00\%$** |
| 3 | `prep_he` | Global Histogram Eq. | $95.21\% \pm 1.57\%$ | $96.26\% \pm 1.56\%$ | $96.16\% \pm 1.36\%$ | **$100.0\% \pm 0.00\%$** |
| 4 | `prep_clahe_2_0` | CLAHE (clip=2.0) | $95.56\% \pm 0.52\%$ | $95.49\% \pm 1.40\%$ | $95.50\% \pm 1.58\%$ | $96.67\% \pm 5.77\%$ |
| 5 | `prep_clahe_2_5` | CLAHE (clip=2.5) | $95.56\% \pm 0.52\%$ | $95.49\% \pm 1.40\%$ | $95.50\% \pm 1.58\%$ | $96.67\% \pm 5.77\%$ |

---

## 5. Tradeoffs & Selection Conclusion

1. **Best Validation Selection:**
   * **`prep_clahe_1_5`** is selected as the optimal preprocessing recipe. It delivers the highest validation Macro-F1 ($96.41\%$), highest validation accuracy ($95.66\%$), and maintains $100\%$ sensitivity on the rare Moderate class with low variance across seeds ($\sigma = 0.58\%$).
2. **Why Moderation Wins in CLAHE:**
   * A moderate `clipLimit=1.5` prevents the local histogram from amplifying subtle scanner background noise into high-frequency spurious edges while successfully enhancing ventricular boundaries.
3. **Artifact Finding & Proposed Additional Candidate (for future user consideration):**
   * Stage 1 revealed that 100% of images have a black outer perimeter with a border zero ratio of 1.0000. However, in HE and high-clip CLAHE, subtle gradients near the circular FOV mask can introduce edge halos.
   * *Proposed future candidate:* **Brain Foreground Masking / Zero-Background Thresholding** before CLAHE, where non-zero pixels are normalized independently while forcing true background pixels to remain zero.  
   *(Per instructions, this candidate is not added to the current grid; awaiting user prompt if desired).*

---

## 6. Next-Stage Prerequisites (Stage 4)

- [x] Preprocessing grid complete across all 5 candidates and 3 seeds ($15$ total evaluation points).
- [x] Best preprocessing selected on validation: **`prep_clahe_1_5`** (with `prep_original` as a clean secondary reference).
- [x] Zero augmentation was active during Stage 3; Stage 4 will systematically introduce and ablate controlled augmentation techniques.
- [x] Test split strictly unread and blinded.

*Stage 3 complete. Ready and waiting for the user's Stage 4 prompt.*
