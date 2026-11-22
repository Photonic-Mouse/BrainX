# BrainScope-XAI Data & Split Audit Report (Stage 1)

**Execution Date:** 2026-11-22  
**Audit Scope:** Data verification, manifest integrity, stale path repair, duplicate/near-duplicate detection, and split preservation without model training.  
**Repository Working Directory:** `C:\SHARAN PROJECTS\BRAINSCOPE-XAI`  

---

## 1. Executive Summary

This audit establishes an empirical baseline for the BrainScope-XAI project. All 6,400 MRI image files on disk were audited, mapped to manifest records, analyzed for physical properties, checked for duplicates, and verified against saved research metrics.

**Key Audit Outcomes:**
- **Exact Data Preservation:** All 6,400 images across the 4 diagnostic categories exist and match the manifest with 100% agreement.
- **Split Preservation:** The 70/15/15 split (4,480 Train / 960 Validation / 960 Test) is preserved exactly. Zero rows, split assignments, or labels were modified.
- **Stable Identity Checksum:** The SHA-256 checksum over stable identity fields (`filename|class_name|label|split`) was computed before and after path repair:
  - `85a5d0b86ac2669f55f1c443d6e345782da2bc7af087f335334f6768eb841825` (Identical Before and After).
- **Safe Path Repair:** Stale paths pointing to `C:\HEHAW\BrainX\Dataset\...` were updated in `Dataset/splits/splits.csv` and source scripts to repository-relative paths (`Dataset/<class_name>/<filename>`). All 6,400 repaired paths resolve to valid files on disk.
- **Leakage & Independence Finding:** 0 exact byte-level SHA-256 duplicates exist. However, 543 perceptual near-duplicate groups cross split boundaries, and no patient/subject IDs exist in metadata or filenames. This indicates 2D slice-level splitting with potential patient overlap (per *Yagis et al., Nature Scientific Reports 2021*).

---

## 2. Verification of Prior Claims

| Claim / Statement | Source | Audit Finding | Status |
| :--- | :--- | :--- | :---: |
| **Cohort size: 6,400 images across 4 classes** | README.md, `inspect_dataset.py` | 896 Mild, 64 Moderate, 3200 Non Demented, 2240 Very Mild confirmed on disk. | **Verified** |
| **Split: 4,480 train (70%), 960 val (15%), 960 test (15%)** | README.md, `splits.csv` | Exact row counts match: train=4480, val=960, test=960. | **Verified** |
| **Saved Baseline: 94.38% acc, 95.30% macro-F1** | `results/baseline/` | 906/960 = 94.375% accuracy; text report states 0.9437, macro-F1 0.9530. | **Verified** |
| **Saved CLAHE: 95.10% acc, 94.61% macro-F1** | `results/clahe/` | 913/960 = 95.104% accuracy; text report states 0.9510, macro-F1 0.9461. | **Verified** |
| **Saved HE: 78.23% acc, 79.76% macro-F1** | `results/he/` | 751/960 = 78.229% accuracy; text report states 0.7823, macro-F1 0.7976. | **Verified** |
| **Baseline training is "no-augmentation"** | Project notes | `src/data/dataset.py` line 34 applies `transforms.RandomRotation(10)` during training. It is raw-intensity plus rotation, not no-augmentation. | **Contradicted** |
| **Subject-level clinical independence verified** | Project notes | Filenames are generic sequential (`mild_1.jpg`); 0 EXIF tags exist. Subject IDs cannot be evidenced. | **Contradicted** / **Not Evidenced** |
| **External notes / Pinned chat "Review papers and plan slides"** | Prompt instructions | No external note files or slides exist in the repository; context was evaluated from prompt instructions and source code. | **Not Evidenced** |

---

## 3. Required Data Checks

### Check 1: Actual Image Counts vs. Manifest
Every class folder in `Dataset/` was enumerated on disk and compared to expected totals:

| Class Folder Name | Label ID | Expected Count | Disk Count | Manifest Count | Delta |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `Mild_Demented` | 0 | 896 | 896 | 896 | 0 |
| `Moderate_Demented` | 1 | 64 | 64 | 64 | 0 |
| `Non_Demented` | 2 | 3,200 | 3,200 | 3,200 | 0 |
| `Very_Mild_Demented` | 3 | 2,240 | 2,240 | 2,240 | 0 |
| **Total Cohort** | — | **6,400** | **6,400** | **6,400** | **0** |

### Check 2: Row Counts by Fixed Split and Class
The split manifest was queried for exact class distribution per subset:

| Split Subset | Mild (0) | Moderate (1) | Non Demented (2) | Very Mild (3) | Subset Total | Expected Total | Match Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Train** | 627 | 45 | 2,240 | 1,568 | **4,480** | 4,480 | **Match (100%)** |
| **Validation** | 134 | 10 | 480 | 336 | **960** | 960 | **Match (100%)** |
| **Test** | 135 | 9 | 480 | 336 | **960** | 960 | **Match (100%)** |
| **Cohort Total**| **896** | **64** | **3,200** | **2,240** | **6,400** | **6,400** | **Match (100%)** |

*Class Imbalance Ratio:* Moderate Demented represents only 1.0% of the entire dataset (45 train, 10 val, 9 test), while Non Demented represents 50.0%.

### Check 3: Row Agreement and Resolution
Each manifest record was checked against:
1. `class_name` $\leftrightarrow$ `label` mapping: $0 \rightarrow$ Mild, $1 \rightarrow$ Moderate, $2 \rightarrow$ Non Demented, $3 \rightarrow$ Very Mild.
2. File existence on disk at `Dataset/<class_name>/<filename>`.
- **Total resolution and mapping errors:** **0** (all 6,400 records agree).

---

## 4. Stale Path Inventory & Repair Summary

The previous development environment hardcoded `C:\HEHAW\BrainX`. Below is the complete inventory of stale paths identified and repaired:

| Source File | Stale Absolute Path Found | Repaired Dynamic Resolution |
| :--- | :--- | :--- |
| `Dataset/splits/splits.csv` | `C:\HEHAW\BrainX\Dataset\...` (6,400 rows) | `Dataset/<class_name>/<filename>` |
| `src/data/dataset.py` | Required absolute path on disk | Dynamic relative path resolution from `self.csv_path` or cwd |
| `src/data/create_splits.py` | `C:\HEHAW\BrainX\Dataset` & `dataset\splits` | `REPO_ROOT / "Dataset"` & `DATASET_PATH / "splits"` |
| `src/data/inspect_dataset.py` | `C:\HEHAW\BrainX\Dataset` & `sample_images.png` | `REPO_ROOT / "Dataset"` & `REPO_ROOT / "sample_images.png"` |
| `src/data/test_dataset.py` | `C:\HEHAW\BrainX\dataset\splits\splits.csv` | `REPO_ROOT / "Dataset" / "splits" / "splits.csv"` |
| `src/training/train.py` | `C:\HEHAW\BrainX\dataset\splits\...` & `checkpoints` | Dynamic `REPO_ROOT` relative paths |
| `src/training/train_clahe.py` | `C:\HEHAW\BrainX\dataset\splits\...` & `checkpoints` | Dynamic `REPO_ROOT` relative paths |
| `src/training/train_he.py` | `C:\HEHAW\BrainX\dataset\splits\...` & `checkpoints` | Dynamic `REPO_ROOT` relative paths |
| `src/evaluation/evaluate.py` | `C:\HEHAW\BrainX\dataset\splits\...` & `checkpoints` | Dynamic `REPO_ROOT` relative paths |
| `src/evaluation/evaluate_clahe.py`| `C:\HEHAW\BrainX\dataset\splits\...` & `checkpoints` | Dynamic `REPO_ROOT` relative paths |
| `src/evaluation/evaluate_he.py` | `C:\HEHAW\BrainX\dataset\splits\...` & `checkpoints` | Dynamic `REPO_ROOT` relative paths |
| `src/preprocessing/compare_preprocessing.py` | `C:\HEHAW\BrainX\Dataset` & `results\preprocessing` | Dynamic `REPO_ROOT` relative paths |
| `src/explainability/visualize_gradcam.py` | `C:\HEHAW\BrainX\...` (lines 21, 22, 26, 31, 36) | *Untouched per instructions (leave Grad-CAM alone)* |

### Stable Identity Checksum Verification
- **Before Repair:** `85a5d0b86ac2669f55f1c443d6e345782da2bc7af087f335334f6768eb841825`
- **After Repair:**  `85a5d0b86ac2669f55f1c443d6e345782da2bc7af087f335334f6768eb841825`
- **Match:** **Identical (100% verified)**. Split memberships, labels, class names, and row sequence are completely unaltered.

---

## 5. Physical Image Properties & Artifact Audit

All 6,400 image files were inspected using PIL and NumPy:

- **Dimensions:** $128 \times 128$ pixels across all 6,400 images (100%).
- **Color Mode:** 8-bit Grayscale (`L`) across all 6,400 images (100%).
- **Corrupted Images:** 0 unreadable or corrupted files.
- **Intensity Statistics:**
  - Minimum intensity: 0 (black background across all images).
  - Maximum intensity: 255.
  - Mean image intensity across cohort: $54.21 \pm 8.45$.
- **Border Artifacts:**
  - Border zero ratio (5-pixel outer perimeter): **1.0000** (all images possess completely black boundary margins from circular scanner FOV masking).
- **Text / Watermarks:** No burned-in patient names, timestamps, orientation markers ("L"/"R"), or text annotations were detected.

---

## 6. Duplicate & Cross-Split Contamination Audit

### Exact File Duplicates (SHA-256)
- **Exact duplicate groups:** **0**. No two files share identical SHA-256 hashes.

### Perceptual Near-Duplicates (dHash, 64-bit gradient)
- **Perceptual near-duplicate clusters:** **543 clusters** contain image slices that cross between `train`, `validation`, and `test` splits.
- **Root Cause Analysis:** In 2D axial MRI datasets derived from volumetric scans (e.g. OASIS-1), adjacent slices from the same subject display near-identical structural patterns. A slice-level random split scatters adjacent brain slices of the same individual across training and testing.
- **Reference:** *Yagis et al., Nature Scientific Reports (2021)* demonstrated that 2D slice-level random splitting on brain MRI yields inflated test performance due to patient identity leakage.
- **Action Taken:** Per the non-negotiable rules, **no records were moved or deleted**. The existing split is preserved to maintain comparability with saved benchmarks, but this limitation is formally documented.

---

## 7. Metadata & Subject Independence

- **Filename Patterns:**
  - `mild_*.jpg` (895 files) + `mild.jpg` (1 file)
  - `moderate_*.jpg` (63 files) + `moderate.jpg` (1 file)
  - `non_*.jpg` (3199 files) + `non.jpg` (1 file)
  - `verymild_*.jpg` (2239 files) + `verymild.jpg` (1 file)
- **EXIF Metadata:** 0 EXIF or DICOM header tags exist in the JPEG files.
- **Conclusion:** **Subject-level independence cannot be verified** from this dataset copy. Reported test metrics represent 2D image/slice classification accuracy on the benchmark split, not prospective patient-level clinical diagnosis.

---

## 8. Saved Arithmetic Verification

The test prediction logs (`test_predictions.csv`) were re-analyzed to confirm arithmetic consistency with saved classification reports:

1. **Baseline ResNet-18:**
   - Saved report: Test Accuracy = 0.9437, Test Macro F1 = 0.9530.
   - Evaluated from `test_predictions.csv`: $\frac{906}{960} = 0.943750$. Arithmetic verified.
2. **CLAHE ResNet-18:**
   - Saved report: Test Accuracy = 0.9510, Test Macro F1 = 0.9461.
   - Evaluated from `test_predictions.csv`: $\frac{913}{960} = 0.951042$. Arithmetic verified.
3. **HE ResNet-18:**
   - Saved report: Test Accuracy = 0.7823, Test Macro F1 = 0.7976.
   - Evaluated from `test_predictions.csv`: $\frac{751}{960} = 0.782292$. Arithmetic verified.

---

## 9. Machine-Readable Artifact Index

The following deliverables were generated and stored in `results/audit/`:
1. `results/audit/data_audit_report.md` — This comprehensive audit report.
2. `results/audit/file_inventory.csv` — Full 6,400-row inventory with SHA-256 hashes, dHash values, dimensions, intensities, and split tags.
3. `results/audit/duplicate_groups.json` — Detailed mapping of all 543 cross-split perceptual near-duplicate groups.
4. `results/audit/path_repair_summary.json` — Verification log of manifest path repairs and stable identity checksums.
5. `results/audit/run_stage1_audit.py` — Fully reproducible audit script.

---

## 10. Next-Stage Prerequisites (Stage 2)

Before beginning Stage 2 ("Make training reproducible and establish baselines"):
- [x] All 6,400 files resolve unambiguously from `Dataset/splits/splits.csv`.
- [x] PyTorch dataset loader (`BrainMRIDataset`) loads batches with ImageNet normalization and relative paths without errors.
- [x] Stable identity hash confirmed matching before and after path repair.
- [x] `torchvision` installed and operational in the local runtime environment.
- [x] Test set remains blinded; all subsequent model selection decisions will be restricted strictly to validation metrics.

---
*Stage 1 audit complete. Awaiting user prompt for Stage 2.*
