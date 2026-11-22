import os
import sys
import time
import json
import random
import hashlib
import platform
import subprocess
import shutil
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
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
# 1. SETUP & PATH RESOLUTION
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
# 2. CACHED DATASET (Using Stage 3+4 Winner: CLAHE 1.5 + rot5)
# -------------------------------------------------------------
class ImbalanceCachedDataset(Dataset):
    def __init__(self, records, transform):
        """
        records: list of (raw_gray_numpy, label, rel_path)
        transform: torchvision transforms
        """
        self.records = records
        self.transform = transform
        self.clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        raw_gray, label, rel_path = self.records[idx]

        # Stage 3 Winner: CLAHE 1.5
        proc = self.clahe.apply(raw_gray)
        rgb = cv2.cvtColor(proc, cv2.COLOR_GRAY2RGB)
        pil_img = Image.fromarray(rgb)

        if self.transform is not None:
            pil_img = self.transform(pil_img)

        return pil_img, torch.tensor(label, dtype=torch.long), rel_path

# -------------------------------------------------------------
# 3. SINGLE RUN TRAINER FOR IMBALANCE STRATEGIES
# -------------------------------------------------------------
def run_imbalance_experiment(
    strategy_name: str,
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
    run_id = f"{strategy_name}_seed{seed}"
    run_dir = EXPERIMENTS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print(f"  STARTING RUN: {run_id}")
    print(f"  Strategy: {strategy_name} | Seed: {seed} | Device: {device}")
    print(f"  Base Pipeline: CLAHE 1.5 + RandomRotation(5)")
    print("=" * 65)

    seed_everything(seed)
    gen = torch.Generator()
    gen.manual_seed(seed)

    # 1. Transforms (Stage 4 Winner: rot5 for train, deterministic for val)
    shared_norm = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(5),
        transforms.ToTensor(),
        shared_norm
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        shared_norm
    ])

    train_ds = ImbalanceCachedDataset(train_cache, transform=train_transform)
    val_ds = ImbalanceCachedDataset(val_cache, transform=val_transform)

    # 2. Configure Loss & Sampler according to Strategy
    targets = [int(lbl) for _, lbl, _ in train_cache]
    class_counts = pd.Series(targets).value_counts().reindex(range(4), fill_value=0)

    if strategy_name == "imb_unweighted":
        # Candidate 1: Ordinary Cross-Entropy, standard shuffling
        criterion = nn.CrossEntropyLoss()
        sampler = None
        shuffle = True
        class_weights_desc = "none (unweighted)"

    elif strategy_name == "imb_class_weighted":
        # Candidate 2: Inverse-frequency class weights from train counts only
        class_weights = len(train_df) / (4.0 * class_counts.values)
        weights_t = torch.tensor(class_weights, dtype=torch.float32).to(device)
        criterion = nn.CrossEntropyLoss(weight=weights_t)
        sampler = None
        shuffle = True
        class_weights_desc = class_weights.tolist()

    elif strategy_name == "imb_balanced_sampler":
        # Candidate 3: Balanced training sampler without class weights
        class_weights_for_sampling = 1.0 / class_counts.values
        sample_weights = [class_weights_for_sampling[t] for t in targets]
        sampler = WeightedRandomSampler(
            weights=torch.DoubleTensor(sample_weights),
            num_samples=len(train_df),
            replacement=True,
            generator=gen
        )
        criterion = nn.CrossEntropyLoss() # Unweighted loss! Do not double-correct
        shuffle = False # mutually exclusive with sampler
        class_weights_desc = "none (unweighted, balanced sampler active)"

    else:
        raise ValueError(f"Unknown strategy: {strategy_name}")

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
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

    # Model
    weights = ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, 4)
    model = model.to(device)

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
            f"Val Macro-F1: {val_macro_f1:.4f} | "
            f"Mod Recall: {per_class_rec[1]:.2f}"
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

    # Save history and predictions
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
        "mild_precision": h["per_class_precision"][0],
        "moderate_precision": h["per_class_precision"][1],
        "non_precision": h["per_class_precision"][2],
        "verymild_precision": h["per_class_precision"][3],
        "mild_f1": h["per_class_f1"][0],
        "moderate_f1": h["per_class_f1"][1],
        "non_f1": h["per_class_f1"][2],
        "verymild_f1": h["per_class_f1"][3],
        "duration_sec": h["epoch_duration_sec"]
    } for h in history])
    history_df.to_csv(run_dir / "history.csv", index=False)

    if best_val_preds is not None:
        best_val_preds.to_csv(run_dir / "val_predictions.csv", index=False)

    best_record = next(h for h in history if h["epoch"] == best_epoch)

    manifest = {
        "run_id": run_id,
        "strategy_name": strategy_name,
        "seed": seed,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None",
        "splits_manifest_sha256": compute_file_sha256(SPLITS_CSV),
        "model": "ResNet18",
        "weights": "ResNet18_Weights.DEFAULT (ImageNet)",
        "preprocessing": "CLAHE_1.5",
        "augmentation": "RandomRotation(5)",
        "imbalance_handling": strategy_name,
        "class_weights": class_weights_desc,
        "optimizer": "Adam",
        "learning_rate": lr,
        "batch_size": batch_size,
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
        "best_per_class_precision": {
            "Mild_Demented": best_record["per_class_precision"][0],
            "Moderate_Demented": best_record["per_class_precision"][1],
            "Non_Demented": best_record["per_class_precision"][2],
            "Very_Mild_Demented": best_record["per_class_precision"][3]
        },
        "best_per_class_f1": {
            "Mild_Demented": best_record["per_class_f1"][0],
            "Moderate_Demented": best_record["per_class_f1"][1],
            "Non_Demented": best_record["per_class_f1"][2],
            "Very_Mild_Demented": best_record["per_class_f1"][3]
        },
        "best_confusion_matrix": best_record["confusion_matrix"],
        "best_checkpoint_path": str(best_ckpt_path.relative_to(REPO_ROOT))
    }

    with open(run_dir / "run_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Run {run_id} completed. Best Val Macro-F1: {best_val_macro_f1:.4f} at epoch {best_epoch}.\n")
    return manifest

# -------------------------------------------------------------
# 4. MAIN STAGE 5 EXECUTION
# -------------------------------------------------------------
def main():
    SEEDS = [17, 42, 2026]
    STRATEGIES = [
        "imb_unweighted",
        "imb_balanced_sampler"
    ]

    print("================================================================")
    print("  STAGE 5: CLASS IMBALANCE HANDLING & TRAINING STABILITY LOOP   ")
    print("  Candidates:                                                   ")
    print("    1. Ordinary unweighted Cross-Entropy                        ")
    print("    2. Inverse-frequency class-weighted Cross-Entropy (baseline)")
    print("    3. Balanced WeightedRandomSampler (unweighted loss)        ")
    print("  Pipeline Base: CLAHE 1.5 + RandomRotation(5)                  ")
    print("  Seeds: 17, 42, 2026                                           ")
    print("================================================================")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    df = pd.read_csv(SPLITS_CSV)
    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "validation"].copy()

    # Pre-cache images in memory
    print("Caching 4,480 train images into memory...")
    train_cache = []
    for _, row in train_df.iterrows():
        raw_gray = cv2.imread(str(REPO_ROOT / row["image_path"]), cv2.IMREAD_GRAYSCALE)
        train_cache.append((raw_gray, int(row["label"]), row["image_path"]))

    print("Caching 960 validation images into memory...")
    val_cache = []
    for _, row in val_df.iterrows():
        raw_gray = cv2.imread(str(REPO_ROOT / row["image_path"]), cv2.IMREAD_GRAYSCALE)
        val_cache.append((raw_gray, int(row["label"]), row["image_path"]))

    all_manifests = []

    # Include candidate 2 (imb_class_weighted) from Stage 4 clahe15_rot5 runs
    for seed in SEEDS:
        src_dir = EXPERIMENTS_DIR / f"clahe15_rot5_seed{seed}"
        dst_dir = EXPERIMENTS_DIR / f"imb_class_weighted_seed{seed}"
        if src_dir.exists():
            if not dst_dir.exists():
                shutil.copytree(src_dir, dst_dir)
            p = dst_dir / "run_manifest.json"
            with open(p) as f:
                m = json.load(f)
            m["run_id"] = f"imb_class_weighted_seed{seed}"
            m["strategy_name"] = "imb_class_weighted"
            m["imbalance_handling"] = "imb_class_weighted"
            m["best_checkpoint_path"] = str((dst_dir / "checkpoints" / "best_model.pth").relative_to(REPO_ROOT))

            # Compute precision/F1 if missing
            val_preds_path = dst_dir / "val_predictions.csv"
            if val_preds_path.exists():
                vdf = pd.read_csv(val_preds_path)
                prec = precision_score(vdf["actual"], vdf["predicted"], average=None, zero_division=0).tolist()
                f1s = f1_score(vdf["actual"], vdf["predicted"], average=None, zero_division=0).tolist()
                m["best_per_class_precision"] = {
                    CLASS_NAMES[i]: prec[i] for i in range(4)
                }
                m["best_per_class_f1"] = {
                    CLASS_NAMES[i]: f1s[i] for i in range(4)
                }

            with open(p, "w") as f:
                json.dump(m, f, indent=2)
            all_manifests.append(m)

    # Run Candidate 1 (unweighted) and Candidate 3 (balanced sampler)
    for strat in STRATEGIES:
        for seed in SEEDS:
            m = run_imbalance_experiment(
                strategy_name=strat,
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

    # Summary table
    records = []
    for m in all_manifests:
        records.append({
            "strategy": m["strategy_name"],
            "seed": m["seed"],
            "best_epoch": m["best_epoch"],
            "val_accuracy": m["best_val_accuracy"],
            "val_balanced_accuracy": m["best_val_balanced_accuracy"],
            "val_macro_f1": m["best_val_macro_f1"],
            "val_weighted_f1": m["best_val_weighted_f1"],
            "moderate_recall": m["best_per_class_recall"]["Moderate_Demented"],
            "moderate_precision": m.get("best_per_class_precision", {}).get("Moderate_Demented", 0.0),
            "moderate_f1": m.get("best_per_class_f1", {}).get("Moderate_Demented", 0.0),
            "mild_recall": m["best_per_class_recall"]["Mild_Demented"],
            "non_recall": m["best_per_class_recall"]["Non_Demented"],
            "verymild_recall": m["best_per_class_recall"]["Very_Mild_Demented"]
        })

    summary_df = pd.DataFrame(records)
    summary_csv = EXPERIMENTS_DIR / "imbalance_comparison_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    print("\nImbalance Candidate Summary Table:")
    print(summary_df.to_string())

    # Aggregates across seeds
    agg = summary_df.groupby("strategy").agg({
        "val_accuracy": ["mean", "std"],
        "val_balanced_accuracy": ["mean", "std"],
        "val_macro_f1": ["mean", "std"],
        "val_weighted_f1": ["mean", "std"],
        "moderate_recall": ["mean", "std"],
        "moderate_precision": ["mean", "std"],
        "moderate_f1": ["mean", "std"]
    }).reset_index()
    agg.to_csv(EXPERIMENTS_DIR / "imbalance_aggregates.csv", index=False)
    print("\nAggregated Performance by Imbalance Strategy (Mean +/- Std):")
    print(agg.to_string())

if __name__ == "__main__":
    main()
