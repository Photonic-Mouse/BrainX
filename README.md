# BrainX — Deep Learning & Explainable AI for Alzheimer's Disease & MCI

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Framework-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

An end-to-end deep learning and visual explainability platform that differentiates four stages of dementia across structural brain MRI scans using **ResNet18** and **Layer-4 Grad-CAM**, evaluating the comparative impact of **Histogram Equalization (HE)** and **Contrast Limited Adaptive Histogram Equalization (CLAHE)** against an unenhanced **Baseline**.

---

## 🔬 Project Overview

- **Target Architecture:** ResNet18 (ImageNet pretrained backbone fine-tuned with class-weighted cross-entropy).
- **Cohort:** 6,400 2D axial structural brain MRI scans.
  - **Train Split (70%):** 4,480 images
  - **Validation Split (15%):** 960 images
  - **Held-Out Test Split (15%):** 960 images
- **Four-Class Classification Target:**
  1. **Mild Demented**
  2. **Moderate Demented**
  3. **Non Demented**
  4. **Very Mild Demented** (Early Mild Cognitive Impairment)
- **Explainability:** Dynamic Gradient-weighted Class Activation Mapping (Grad-CAM) targeting `model.layer4[-1]`.

---

## 📊 Comparative Research Benchmarks

Evaluated on the exact held-out test cohort ($N = 960$ unseen scans):

| Model Architecture | Preprocessing Technique | Test Accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted F1 |
|:---|:---|:---:|:---:|:---:|:---:|:---:|
| **Baseline ResNet18** | Original Unenhanced MRI | 94.38% | 95.37% | **95.23%** | **95.30%** | 94.36% |
| **HE ResNet18** | Global Histogram Equalization | 78.23% | 78.63% | 82.49% | 79.76% | 78.40% |
| **CLAHE ResNet18** | Adaptive Local Contrast (`clipLimit=2.0`) | **95.10%** | **96.68%** | 92.81% | 94.61% | **95.11%** |

### Key Empirical Findings:
1. **CLAHE delivers the highest overall accuracy (95.10%) and highest weighted F1 (95.11%)**, outperforming the unenhanced baseline on Mild, Non Demented, and Very Mild categories.
2. **Global Histogram Equalization severely degrades diagnostic performance (-16.15% drop in accuracy)** by amplifying background MRI scanner noise and flattening soft-tissue gray/white matter boundaries.
3. **Grad-CAM Saliency Correlates with Neuropathology:** Baseline and CLAHE focus consistently on ventricular expansion (lateral ventricles) and temporal lobe / hippocampal atrophy. In contrast, HE generates diffused activations extending to the skull table and background artifacts.

---

## 🚀 Quick Start & Demonstration

### 1-Click Launch (Windows)
Double-click **`START_BRAINX.bat`** in the project root:
- Automatically verifies project root, virtual environment, and model checkpoints.
- Checks port 8000 (prevents duplicate instances if already running).
- Polls backend health status on CUDA/CPU.
- **Pops open your default web browser directly to `http://127.0.0.1:8000`**.

### Manual Command Line Launch
```powershell
# Activate virtual environment
.\venv\Scripts\activate

# Launch application
python run_app.py
```

### Clean Shutdown
- Press `Ctrl + C` in the launcher console, or
- Double-click **`STOP_BRAINX.bat`**.

---

## 🖥️ Interactive Web Dashboard Features

- **Multi-Model Inference:** Upload an arbitrary brain MRI (`.png`, `.jpg`, `.jpeg`) or select from built-in cohort sample chips to run all three models simultaneously.
- **Model Consensus & Agreement Engine:** Identifies unanimous agreement, majority consensus, or borderline diagnostic divergence.
- **Interactive Probability Chart:** Grouped bar chart comparing 4-class softmax scores across all models.
- **3-Way Preprocessing Inspector:** Visual comparison of Original, HE, and CLAHE enhancements.
- **Dynamic Layer-4 Grad-CAM:** 4-column aligned view showing Original MRI, Preprocessed Image, Heatmap, and Overlay.
- **Downloadable Formal PDF Report:** ReportLab-generated research summary report with embedded figures, probability tables, and clinical disclaimers.

---

## 📁 Repository Structure

```text
BrainX/
├── checkpoints/                    # Trained model weights (.pth)
│   ├── resnet18_baseline.pth
│   ├── resnet18_he.pth
│   └── resnet18_clahe.pth
├── dataset/
│   └── splits/
│       └── splits.csv              # Exact stratified train/val/test splits
├── frontend/                       # Interactive Research Dashboard
│   ├── index.html
│   ├── styles.css
│   └── app.js
├── results/                        # Research evaluation outputs
│   ├── baseline/
│   ├── he/
│   ├── clahe/
│   ├── comparison/
│   └── gradcam/
├── src/
│   ├── api/                        # FastAPI application server
│   │   └── app.py
│   ├── data/                       # Dataset loaders and splitting scripts
│   ├── evaluation/                 # Test set evaluation scripts
│   ├── explainability/             # Grad-CAM implementation
│   ├── inference/                  # Pipeline engine & PDF report generator
│   ├── models/                     # ResNet18 architecture definition
│   └── training/                   # Model training scripts
├── launcher.py                     # Pre-flight check & health orchestrator
├── run_app.py                      # Server runner
├── START_BRAINX.bat                # 1-Click Windows Demonstration Launcher
├── STOP_BRAINX.bat                 # 1-Click Shutdown Utility
└── test_system.py                  # Automated 8-point verification test suite
```

---

## ⚖️ Academic Safety & Ethical Disclaimer

This application is an **academic research prototype** created for computational neuroscience and algorithmic comparison. It is **NOT** a certified medical diagnostic device and does **NOT** provide a clinical diagnosis of Alzheimer's Disease or Mild Cognitive Impairment. Model softmax confidence scores reflect internal training calibration and must not be used as clinical diagnostic certainty.
