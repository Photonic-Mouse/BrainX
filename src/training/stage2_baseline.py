import os
import sys
import time
import json
import random
import hashlib
import platform
import subprocess
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

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix
)

# -------------------------------------------------------------
# 1. SHARED DEFINITIONS & PATH RESOLUTION
# -------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = REPO_ROOT / "Dataset"
SPLITS_CSV = DATASET_DIR / "splits" / "splits.csv"
EXPERIMENTS_DIR = REPO_ROOT / "results" / "experiments"
EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = [
    "Mild_Demented",
    "Moderate_Demented",
    "Non_Demented",
    "Very_Mild_Demented"
]
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}

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

def get_git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT).decode("ascii").strip()
    except Exception:
        return "unknown"

def compute_file_sha256(path: Path) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()

# -------------------------------------------------------------
# 2. DATASET FACTORY (Consistent Path Resolution & Transforms)
# -------------------------------------------------------------
class ReproducibleBrainMRIDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform=None, preprocessing="original"):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.preprocessing = preprocessing

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_rel_path = row["image_path"]
        label = int(row["label"])

        # Dynamically resolve relative path to REPO_ROOT
        full_path = REPO_ROOT / img_rel_path
        if not full_path.exists():
            raise FileNotFoundError(f"Image not found at {full_path}")

        # Open in Grayscale ('L')
        with Image.open(full_path) as img:
            img = img.convert("L")
            # Apply preprocessing if requested
            if self.preprocessing == "HE":
                import cv2
                arr = np.array(img)
                arr = cv2.equalizeHist(arr)
                img = Image.fromarray(arr)
            elif self.preprocessing == "CLAHE":
                import cv2
                arr = np.array(img)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                arr = clahe.apply(arr)
                img = Image.fromarray(arr)
            
            # Convert grayscale to 3-channel RGB for pretrained ResNet18
            img = img.convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        return img, torch.tensor(label, dtype=torch.long), img_rel_path

# -------------------------------------------------------------
# 3. MODEL FACTORY
# -------------------------------------------------------------
def build_resnet18(num_classes=4):
    weights = ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model

# -------------------------------------------------------------
# 4. SINGLE RUN TRAINER
# -------------------------------------------------------------
def run_training_experiment(
    condition_name: str,
    seed: int,
    batch_size: int = 16,
    lr: float = 1e-4,
    epochs: int = 8,
    patience: int = 5,
    device: torch.device = None
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_id = f"{condition_name}_seed{seed}"
    run_dir = EXPERIMENTS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print(f"  STARTING RUN: {run_id}")
    print(f"  Condition: {condition_name} | Seed: {seed} | Device: {device}")
    print(f"  Batch Size: {batch_size} | LR: {lr} | Max Epochs: {epochs}")
    print("=" * 65)

    # 1. Set Seed
    seed_everything(seed)
    gen = torch.Generator()
    gen.manual_seed(seed)

    # 2. Load Manifest & Splits
    df = pd.read_csv(SPLITS_CSV)
    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "validation"].copy()
    # Note: test_df is NEVER loaded into the trainer! Test set is strictly kept blinded.

    # 3. Class weights from TRAIN COUNTS ONLY
    class_counts = train_df["label"].value_counts().reindex(range(4), fill_value=0)
    class_weights = len(train_df) / (4.0 * class_counts.values)
    class_weights_t = torch.tensor(class_weights, dtype=torch.float32).to(device)

    # 4. Transforms Definition
    shared_norm = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    if condition_name.startswith("raw_no_aug"):
        # Explicit confirmation: ZERO random transformation
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            shared_norm
        ])
    elif condition_name.startswith("raw_rot10_legacy"):
        # Legacy replication condition: RandomRotation up to 10 degrees
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            shared_norm
        ])
    else:
        raise ValueError(f"Unknown condition: {condition_name}")

    # Validation transform is strictly deterministic
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        shared_norm
    ])

    train_ds = ReproducibleBrainMRIDataset(train_df, transform=train_transform, preprocessing="original")
    val_ds = ReproducibleBrainMRIDataset(val_df, transform=val_transform, preprocessing="original")

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

    # 5. Build Model & Optimizer
    model = build_resnet18(num_classes=4).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights_t)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 6. Training & Validation Loop
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

        # Validation phase
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

        # Checkpoint selection on validation macro-F1
        if val_macro_f1 > best_val_macro_f1:
            best_val_macro_f1 = val_macro_f1
            best_epoch = epoch
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_ckpt_path)

            # Store predictions of best validation epoch
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
                print(f"  -> Early stopping triggered after {patience} epochs without improvement.")
                break

    # 7. Save Run Artifacts
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
        "seed": seed,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None",
        "git_commit": get_git_commit(),
        "splits_manifest_sha256": compute_file_sha256(SPLITS_CSV),
        "split_counts": {
            "train": len(train_df),
            "validation": len(val_df)
        },
        "model": "ResNet18",
        "weights": "ResNet18_Weights.DEFAULT (ImageNet)",
        "preprocessing": "original",
        "augmentation": "none" if condition_name.startswith("raw_no_aug") else "RandomRotation(10)",
        "optimizer": "Adam",
        "learning_rate": lr,
        "batch_size": batch_size,
        "class_weights": class_weights.tolist(),
        "software_versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torchvision": models.__version__ if hasattr(models, "__version__") else "0.21.0"
        },
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
# 5. MAIN EXECUTION PIPELINE ACROSS SEEDS AND CONDITIONS
# -------------------------------------------------------------
def main():
    SEEDS = [17, 42, 2026]
    CONDITIONS = ["raw_no_aug", "raw_rot10_legacy"]

    print("================================================================")
    print("  STAGE 2: REPRODUCIBLE BASELINE TRAINING (SEEDS: 17, 42, 2026) ")
    print("================================================================")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Compute hardware target: {device}")
    if device.type == "cuda":
        print(f"GPU device: {torch.cuda.get_device_name(0)}")

    all_manifests = []

    for condition in CONDITIONS:
        for seed in SEEDS:
            manifest = run_training_experiment(
                condition_name=condition,
                seed=seed,
                batch_size=16,
                lr=1e-4,
                epochs=8,
                patience=5,
                device=device
            )
            all_manifests.append(manifest)

    # Summary table computation
    records = []
    for m in all_manifests:
        records.append({
            "condition": m["condition_name"],
            "seed": m["seed"],
            "best_epoch": m["best_epoch"],
            "val_accuracy": m["best_val_accuracy"],
            "val_balanced_accuracy": m["best_val_balanced_accuracy"],
            "val_macro_f1": m["best_val_macro_f1"],
            "val_weighted_f1": m["best_val_weighted_f1"],
            "moderate_recall": m["best_per_class_recall"]["Moderate_Demented"]
        })

    summary_df = pd.DataFrame(records)
    summary_csv = EXPERIMENTS_DIR / "baseline_comparison_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    print("Comparison table:")
    print(summary_df.to_string())

    # Aggregates across seeds
    agg = summary_df.groupby("condition").agg({
        "val_accuracy": ["mean", "std"],
        "val_balanced_accuracy": ["mean", "std"],
        "val_macro_f1": ["mean", "std"],
        "val_weighted_f1": ["mean", "std"],
        "moderate_recall": ["mean", "std"]
    }).reset_index()
    agg.to_csv(EXPERIMENTS_DIR / "baseline_aggregates.csv", index=False)
    print("\nAggregated Performance (Mean +/- Std):")
    print(agg.to_string())

if __name__ == "__main__":
    main()
