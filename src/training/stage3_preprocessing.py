import os
import sys
import time
import json
import random
import hashlib
import platform
import subprocess
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix
)

# -------------------------------------------------------------
# 1. SETUP & PATH RESOLUTION
# -------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = REPO_ROOT / "Dataset"
SPLITS_CSV = DATASET_DIR / "splits" / "splits.csv"
EXPERIMENTS_DIR = REPO_ROOT / "results" / "experiments"
PANELS_DIR = EXPERIMENTS_DIR / "preprocessing_panels"
EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
PANELS_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = [
    "Mild_Demented",
    "Moderate_Demented",
    "Non_Demented",
    "Very_Mild_Demented"
]

def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def compute_file_sha256(path: Path) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()

# -------------------------------------------------------------
# 2. FIXED-IMAGE PREPROCESSING COMPARISON PANEL
# -------------------------------------------------------------
def generate_preprocessing_comparison_panel(df: pd.DataFrame):
    print("Generating visual comparison panel for fixed samples...")
    # Select 1 representative validation sample per class
    val_df = df[df["split"] == "validation"]
    selected_samples = []
    for c_idx, c_name in enumerate(CLASS_NAMES):
        c_rows = val_df[val_df["label"] == c_idx]
        selected_samples.append((c_name, c_rows.iloc[0]["image_path"]))

    clahe_15 = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    clahe_20 = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_25 = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))

    fig, axes = plt.subplots(4, 5, figsize=(15, 12))
    col_titles = ["Original", "Global HE", "CLAHE (clip=1.5)", "CLAHE (clip=2.0)", "CLAHE (clip=2.5)"]

    for row_idx, (c_name, rel_p) in enumerate(selected_samples):
        img_path = REPO_ROOT / rel_p
        raw_gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)

        img_orig = raw_gray
        img_he = cv2.equalizeHist(raw_gray)
        img_c15 = clahe_15.apply(raw_gray)
        img_c20 = clahe_20.apply(raw_gray)
        img_c25 = clahe_25.apply(raw_gray)

        images = [img_orig, img_he, img_c15, img_c20, img_c25]

        for col_idx, img in enumerate(images):
            ax = axes[row_idx, col_idx]
            ax.imshow(img, cmap="gray")
            ax.axis("off")
            if row_idx == 0:
                ax.set_title(col_titles[col_idx], fontsize=12, pad=8)
            if col_idx == 0:
                ax.text(-25, 64, c_name.replace("_", "\n"), fontsize=11, fontweight="bold",
                        va="center", ha="right")

    plt.tight_layout()
    panel_path = PANELS_DIR / "preprocessing_comparison_panel.png"
    plt.savefig(panel_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Comparison panel saved to: {panel_path}")

# -------------------------------------------------------------
# 3. HIGH-PERFORMANCE PRE-CACHED DATASET
# -------------------------------------------------------------
class FastCachedDataset(Dataset):
    def __init__(self, records, prep_type, transform):
        """
        records: list of (raw_gray_numpy, label, rel_path)
        prep_type: 'original', 'HE', 'CLAHE_1.5', 'CLAHE_2.0', 'CLAHE_2.5'
        transform: torchvision transforms
        """
        self.records = records
        self.prep_type = prep_type
        self.transform = transform

        if prep_type == "CLAHE_1.5":
            self.clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
        elif prep_type == "CLAHE_2.0":
            self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        elif prep_type == "CLAHE_2.5":
            self.clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        else:
            self.clahe = None

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        raw_gray, label, rel_path = self.records[idx]

        # Apply deterministic preprocessing
        if self.prep_type == "HE":
            proc = cv2.equalizeHist(raw_gray)
        elif self.clahe is not None:
            proc = self.clahe.apply(raw_gray)
        else: # original
            proc = raw_gray

        # Convert to 3-channel RGB PIL Image
        rgb = cv2.cvtColor(proc, cv2.COLOR_GRAY2RGB)
        pil_img = Image.fromarray(rgb)

        if self.transform is not None:
            pil_img = self.transform(pil_img)

        return pil_img, torch.tensor(label, dtype=torch.long), rel_path

# -------------------------------------------------------------
# 4. SINGLE RUN EXECUTION
# -------------------------------------------------------------
def run_preprocessing_experiment(
    condition_name: str,
    prep_type: str,
    seed: int,
    train_cache,
    val_cache,
    train_df,
    val_df,
    batch_size: int = 16,
    lr: float = 1e-4,
    epochs: int = 8,
    patience: int = 5,
    device: torch.device = None
):
    run_id = f"{condition_name}_seed{seed}"
    run_dir = EXPERIMENTS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print(f"  STARTING RUN: {run_id}")
    print(f"  Condition: {condition_name} (Prep: {prep_type}) | Seed: {seed} | Device: {device}")
    print(f"  Augmentation: DISABLED (Clean comparison) | Batch: {batch_size} | LR: {lr}")
    print("=" * 65)

    seed_everything(seed)
    gen = torch.Generator()
    gen.manual_seed(seed)

    # Class weights from TRAIN COUNTS ONLY
    class_counts = train_df["label"].value_counts().reindex(range(4), fill_value=0)
    class_weights = len(train_df) / (4.0 * class_counts.values)
    class_weights_t = torch.tensor(class_weights, dtype=torch.float32).to(device)

    # Transforms: Deterministic for BOTH train and val (NO augmentation)
    shared_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    train_ds = FastCachedDataset(train_cache, prep_type=prep_type, transform=shared_transform)
    val_ds = FastCachedDataset(val_cache, prep_type=prep_type, transform=shared_transform)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        worker_init_fn=seed_worker,
        generator=gen,
        pin_memory=True if device.type == "cuda" else False
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True if device.type == "cuda" else False
    )

    # Model: ResNet18 ImageNet pretrained with 4-class head
    weights = ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, 4)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights_t)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    history = []
    best_val_macro_f1 = -1.0
    best_epoch = -1
    best_ckpt_path = ckpt_dir / "best_model.pth"
    best_val_preds = None
    epochs_no_improve = 0

    for epoch in range(1, epochs + 1):
        t_start = time.time()
        model.train()
        running_train_loss = 0.0

        for imgs, lbls, _ in train_loader:
            imgs = imgs.to(device, non_blocking=True)
            lbls = lbls.to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, lbls)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()

        epoch_train_loss = running_train_loss / len(train_loader)

        # Validation
        model.eval()
        running_val_loss = 0.0
        val_targets = []
        val_preds_list = []
        val_probs_list = []
        val_paths_list = []

        with torch.no_grad():
            for imgs, lbls, paths in val_loader:
                imgs = imgs.to(device, non_blocking=True)
                lbls = lbls.to(device, non_blocking=True)

                outputs = model(imgs)
                loss = criterion(outputs, lbls)
                running_val_loss += loss.item()

                probs = torch.softmax(outputs, dim=1)
                preds = outputs.argmax(dim=1)

                val_targets.extend(lbls.cpu().tolist())
                val_preds_list.extend(preds.cpu().tolist())
                val_probs_list.extend(probs.cpu().tolist())
                val_paths_list.extend(paths)

        epoch_val_loss = running_val_loss / len(val_loader)
        val_acc = accuracy_score(val_targets, val_preds_list)
        val_bal_acc = balanced_accuracy_score(val_targets, val_preds_list)
        val_macro_f1 = f1_score(val_targets, val_preds_list, average="macro", zero_division=0)
        val_weighted_f1 = f1_score(val_targets, val_preds_list, average="weighted", zero_division=0)
        per_class_rec = recall_score(val_targets, val_preds_list, average=None, zero_division=0).tolist()
        per_class_prec = precision_score(val_targets, val_preds_list, average=None, zero_division=0).tolist()
        per_class_f1 = f1_score(val_targets, val_preds_list, average=None, zero_division=0).tolist()
        cm = confusion_matrix(val_targets, val_preds_list, labels=[0, 1, 2, 3]).tolist()

        epoch_sec = time.time() - t_start

        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] ({epoch_sec:.1f}s) | "
            f"Train Loss: {epoch_train_loss:.4f} | "
            f"Val Loss: {epoch_val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f} | "
            f"Val BalAcc: {val_bal_acc:.4f} | "
            f"Val Macro-F1: {val_macro_f1:.4f}"
        )

        epoch_data = {
            "epoch": epoch,
            "train_loss": epoch_train_loss,
            "val_loss": epoch_val_loss,
            "val_accuracy": val_acc,
            "val_balanced_accuracy": val_bal_acc,
            "val_macro_f1": val_macro_f1,
            "val_weighted_f1": val_weighted_f1,
            "per_class_recall": per_class_rec,
            "per_class_precision": per_class_prec,
            "per_class_f1": per_class_f1,
            "confusion_matrix": cm,
            "epoch_duration_sec": epoch_sec
        }
        history.append(epoch_data)

        # Selection strictly on validation macro-F1
        if val_macro_f1 > best_val_macro_f1:
            best_val_macro_f1 = val_macro_f1
            best_epoch = epoch
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_ckpt_path)

            best_val_preds = pd.DataFrame({
                "image_path": val_paths_list,
                "actual": val_targets,
                "predicted": val_preds_list,
                "prob_mild": [p[0] for p in val_probs_list],
                "prob_moderate": [p[1] for p in val_probs_list],
                "prob_non": [p[2] for p in val_probs_list],
                "prob_verymild": [p[3] for p in val_probs_list]
            })
            print(f"  -> Best model saved at epoch {epoch} (Val Macro-F1: {val_macro_f1:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"  -> Early stopping after {patience} epochs without validation improvement.")
                break

    # Save artifacts
    history_df = pd.DataFrame([{
        "epoch": h["epoch"],
        "train_loss": h["train_loss"],
        "val_loss": h["val_loss"],
        "val_accuracy": h["val_accuracy"],
        "val_balanced_accuracy": h["val_balanced_accuracy"],
        "val_macro_f1": h["val_macro_f1"],
        "val_weighted_f1": h["val_weighted_f1"],
        "mild_recall": h["per_class_recall"][0],
        "moderate_recall": h["per_class_recall"][1],
        "non_recall": h["per_class_recall"][2],
        "verymild_recall": h["per_class_recall"][3],
        "duration_sec": h["epoch_duration_sec"]
    } for h in history])
    history_df.to_csv(run_dir / "history.csv", index=False)

    if best_val_preds is not None:
        best_val_preds.to_csv(run_dir / "val_predictions.csv", index=False)

    best_record = next(h for h in history if h["epoch"] == best_epoch)

    manifest = {
        "run_id": run_id,
        "condition_name": condition_name,
        "prep_type": prep_type,
        "seed": seed,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None",
        "splits_manifest_sha256": compute_file_sha256(SPLITS_CSV),
        "model": "ResNet18",
        "weights": "ResNet18_Weights.DEFAULT (ImageNet)",
        "preprocessing": prep_type,
        "augmentation": "none",
        "optimizer": "Adam",
        "learning_rate": lr,
        "batch_size": batch_size,
        "class_weights": class_weights.tolist(),
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_val_macro_f1,
        "best_val_accuracy": best_record["val_accuracy"],
        "best_val_balanced_accuracy": best_record["val_balanced_accuracy"],
        "best_val_weighted_f1": best_record["val_weighted_f1"],
        "best_per_class_recall": {
            "Mild_Demented": best_record["per_class_recall"][0],
            "Moderate_Demented": best_record["per_class_recall"][1],
            "Non_Demented": best_record["per_class_recall"][2],
            "Very_Mild_Demented": best_record["per_class_recall"][3]
        },
        "best_confusion_matrix": best_record["confusion_matrix"],
        "best_checkpoint_path": str(best_ckpt_path.relative_to(REPO_ROOT))
    }

    with open(run_dir / "run_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Run {run_id} completed. Best Val Macro-F1: {best_val_macro_f1:.4f} at epoch {best_epoch}.\n")
    return manifest

# -------------------------------------------------------------
# 5. MAIN STAGE 3 LOOP
# -------------------------------------------------------------
def main():
    SEEDS = [17, 42, 2026]
    CANDIDATES = [
        ("prep_he", "HE"),
        ("prep_clahe_1_5", "CLAHE_1.5"),
        ("prep_clahe_2_0", "CLAHE_2.0"),
        ("prep_clahe_2_5", "CLAHE_2.5")
    ]

    print("================================================================")
    print("  STAGE 3: PREPROCESSING ABLATION EXPERIMENT LOOP               ")
    print("  Candidates: Original, Global HE, CLAHE 1.5, 2.0, 2.5          ")
    print("  Augmentation: DISABLED in ALL arms                            ")
    print("  Seeds: 17, 42, 2026                                           ")
    print("================================================================")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    # Load Manifest
    df = pd.read_csv(SPLITS_CSV)
    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "validation"].copy()

    # Generate fixed-image comparison panels first
    generate_preprocessing_comparison_panel(df)

    # Pre-load raw images into RAM cache (104 MB)
    print("Caching 4,480 train images into memory...")
    t0 = time.time()
    train_cache = []
    for _, row in train_df.iterrows():
        raw_gray = cv2.imread(str(REPO_ROOT / row["image_path"]), cv2.IMREAD_GRAYSCALE)
        train_cache.append((raw_gray, int(row["label"]), row["image_path"]))
    print(f"Train cached in {time.time() - t0:.2f}s.")

    print("Caching 960 validation images into memory...")
    t0 = time.time()
    val_cache = []
    for _, row in val_df.iterrows():
        raw_gray = cv2.imread(str(REPO_ROOT / row["image_path"]), cv2.IMREAD_GRAYSCALE)
        val_cache.append((raw_gray, int(row["label"]), row["image_path"]))
    print(f"Validation cached in {time.time() - t0:.2f}s.")

    all_manifests = []

    # Include original baseline from Stage 2 (raw_no_aug)
    for seed in SEEDS:
        stage2_manifest_path = EXPERIMENTS_DIR / f"raw_no_aug_seed{seed}" / "run_manifest.json"
        if stage2_manifest_path.exists():
            with open(stage2_manifest_path, "r") as f:
                m = json.load(f)
                m["condition_name"] = "prep_original"
                m["prep_type"] = "original"
                all_manifests.append(m)

    # Execute all 4 new candidates across the 3 seeds
    for cond_name, prep_type in CANDIDATES:
        for seed in SEEDS:
            m = run_preprocessing_experiment(
                condition_name=cond_name,
                prep_type=prep_type,
                seed=seed,
                train_cache=train_cache,
                val_cache=val_cache,
                train_df=train_df,
                val_df=val_df,
                batch_size=16,
                lr=1e-4,
                epochs=8,
                patience=5,
                device=device
            )
            all_manifests.append(m)

    # Compile Comparison Table
    records = []
    for m in all_manifests:
        records.append({
            "candidate": m["condition_name"],
            "preprocessing": m["prep_type"],
            "seed": m["seed"],
            "best_epoch": m["best_epoch"],
            "val_accuracy": m["best_val_accuracy"],
            "val_balanced_accuracy": m["best_val_balanced_accuracy"],
            "val_macro_f1": m["best_val_macro_f1"],
            "val_weighted_f1": m["best_val_weighted_f1"],
            "moderate_recall": m["best_per_class_recall"]["Moderate_Demented"],
            "mild_recall": m["best_per_class_recall"]["Mild_Demented"],
            "non_recall": m["best_per_class_recall"]["Non_Demented"],
            "verymild_recall": m["best_per_class_recall"]["Very_Mild_Demented"]
        })

    summary_df = pd.DataFrame(records)
    summary_csv = EXPERIMENTS_DIR / "preprocessing_comparison_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    print("\nCandidate Summary Table:")
    print(summary_df.to_string())

    # Aggregates across seeds
    agg = summary_df.groupby(["candidate", "preprocessing"]).agg({
        "val_accuracy": ["mean", "std"],
        "val_balanced_accuracy": ["mean", "std"],
        "val_macro_f1": ["mean", "std"],
        "val_weighted_f1": ["mean", "std"],
        "moderate_recall": ["mean", "std"]
    }).reset_index()
    agg.to_csv(EXPERIMENTS_DIR / "preprocessing_aggregates.csv", index=False)
    print("\nAggregated Performance by Preprocessing Candidate (Mean +/- Std):")
    print(agg.to_string())

if __name__ == "__main__":
    main()
