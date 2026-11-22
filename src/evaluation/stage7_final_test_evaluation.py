import os
import sys
import time
import json
import random
import hashlib
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image

import torch
import torch.nn as nn
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
from scipy import stats

# -------------------------------------------------------------
# 1. SETUP & PATH RESOLUTION
# -------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = REPO_ROOT / "Dataset"
SPLITS_CSV = DATASET_DIR / "splits" / "splits.csv"
EXPERIMENTS_DIR = REPO_ROOT / "results" / "experiments"
EVAL_DIR = REPO_ROOT / "results" / "evaluation"
EVAL_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = [
    "Mild_Demented",
    "Moderate_Demented",
    "Non_Demented",
    "Very_Mild_Demented"
]

def seed_everything(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def compute_file_sha256(path: Path) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()

def make_opaque_id(rel_path: str, idx: int) -> str:
    h = hashlib.sha256(rel_path.encode("utf-8")).hexdigest()[:8]
    return f"TEST_{idx+1:04d}_{h}"

# -------------------------------------------------------------
# 2. TEST DATASET & INFERENCE TRANSFORMS
# -------------------------------------------------------------
class TestInferenceDataset(Dataset):
    def __init__(self, records, is_clahe: bool = False):
        """
        records: list of (raw_gray_numpy, label, rel_path, opaque_id)
        is_clahe: whether to apply CLAHE 1.5
        """
        self.records = records
        self.is_clahe = is_clahe
        if is_clahe:
            self.clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        raw_gray, label, rel_path, opaque_id = self.records[idx]

        if self.is_clahe:
            proc = self.clahe.apply(raw_gray)
            rgb = cv2.cvtColor(proc, cv2.COLOR_GRAY2RGB)
        else:
            rgb = cv2.cvtColor(raw_gray, cv2.COLOR_GRAY2RGB)

        pil_img = Image.fromarray(rgb)
        tensor_img = self.transform(pil_img)

        return tensor_img, torch.tensor(label, dtype=torch.long), opaque_id, rel_path

# -------------------------------------------------------------
# 3. STATISTICAL HELPERS (CI & McNEMAR)
# -------------------------------------------------------------
def wilson_score_interval(k: int, n: int, confidence: float = 0.95):
    """Computes Wilson score 95% confidence interval for binomial proportion."""
    if n == 0:
        return 0.0, 0.0
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    p = k / n
    denominator = 1 + z**2 / n
    centre_adjusted_probability = p + z**2 / (2 * n)
    adjusted_std = np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n)
    lower = (centre_adjusted_probability - z * adjusted_std) / denominator
    upper = (centre_adjusted_probability + z * adjusted_std) / denominator
    return max(0.0, float(lower)), min(1.0, float(upper))

def bootstrap_metric_ci(y_true, y_pred, metric_fn, n_bootstraps: int = 1000, seed: int = 42):
    """Computes 95% bootstrap confidence interval for arbitrary metric function."""
    rng = np.random.RandomState(seed)
    n = len(y_true)
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    boot_scores = []
    for _ in range(n_bootstraps):
        idxs = rng.choice(n, size=n, replace=True)
        boot_scores.append(metric_fn(y_true[idxs], y_pred[idxs]))
    lower = np.percentile(boot_scores, 2.5)
    upper = np.percentile(boot_scores, 97.5)
    return float(lower), float(upper)

def mcnemar_test_paired(y_true, y_pred_model_a, y_pred_model_b):
    """
    McNemar's test with continuity correction comparing Model A (Frozen) vs Model B (Baseline).
    b = Model A correct, Model B wrong
    c = Model B correct, Model A wrong
    """
    corr_a = (np.array(y_pred_model_a) == np.array(y_true))
    corr_b = (np.array(y_pred_model_b) == np.array(y_true))

    n00 = int(np.sum(corr_a & corr_b))
    n01 = int(np.sum((~corr_a) & corr_b)) # B correct, A wrong (c)
    n10 = int(np.sum(corr_a & (~corr_b))) # A correct, B wrong (b)
    n11 = int(np.sum((~corr_a) & (~corr_b)))

    b = n10
    c = n01
    discordant = b + c

    if discordant == 0:
        chi2 = 0.0
        p_val = 1.0
        odds_ratio = 1.0
        ci_lower, ci_upper = 1.0, 1.0
    else:
        # Edwards continuity correction
        chi2 = float((abs(b - c) - 1.0)**2 / discordant)
        p_val = float(1.0 - stats.chi2.cdf(chi2, df=1))
        
        # Odds ratio = b / c
        if c == 0:
            odds_ratio = float("inf")
            ci_lower, ci_upper = float("nan"), float("nan")
        else:
            odds_ratio = float(b / c)
            se_log_or = np.sqrt(1.0 / max(b, 1) + 1.0 / max(c, 1))
            ci_lower = float(np.exp(np.log(odds_ratio) - 1.96 * se_log_or))
            ci_upper = float(np.exp(np.log(odds_ratio) + 1.96 * se_log_or))

    n_total = len(y_true)
    risk_diff = (b - c) / n_total # Difference in accuracy
    se_rd = np.sqrt(((b + c) - (b - c)**2 / n_total) / (n_total**2))
    rd_ci_lower = float(risk_diff - 1.96 * se_rd)
    rd_ci_upper = float(risk_diff + 1.96 * se_rd)

    return {
        "contingency_table": {
            "both_correct": n00,
            "model_a_correct_model_b_wrong": b,
            "model_b_correct_model_a_wrong": c,
            "both_wrong": n11
        },
        "mcnemar_chi2_statistic": chi2,
        "p_value": p_val,
        "odds_ratio": odds_ratio,
        "odds_ratio_95_ci": [ci_lower, ci_upper],
        "risk_difference_accuracy_gain": float(risk_diff),
        "risk_difference_95_ci": [rd_ci_lower, rd_ci_upper]
    }

# -------------------------------------------------------------
# 4. EVALUATION PIPELINE
# -------------------------------------------------------------
def run_stage7_evaluation():
    print("=" * 70)
    print("  STAGE 7: FROZEN ACCURACY EVALUATION (INTERNAL HOLDOUT N=960)  ")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    manifest_sha256 = compute_file_sha256(SPLITS_CSV)
    print(f"Splits Manifest SHA-256: {manifest_sha256}")

    splits_df = pd.read_csv(SPLITS_CSV)
    test_df = splits_df[splits_df["split"] == "test"].reset_index(drop=True)
    print(f"Loaded Test Cohort: {len(test_df)} images across 4 classes.")
    print("Test Class Distribution:")
    for c_id, c_name in enumerate(CLASS_NAMES):
        count = int(np.sum(test_df["label"] == c_id))
        print(f"  [{c_id}] {c_name:20s}: {count:3d} images ({count / len(test_df) * 100:.1f}%)")

    # Cache raw grayscale test images
    print("\nCaching 960 raw grayscale test images...")
    test_records = []
    for idx, row in test_df.iterrows():
        raw_gray = cv2.imread(str(REPO_ROOT / row["image_path"]), cv2.IMREAD_GRAYSCALE)
        opaque_id = make_opaque_id(row["image_path"], idx)
        test_records.append((raw_gray, int(row["label"]), row["image_path"], opaque_id))

    # Predeclared models to evaluate
    MODELS_TO_EVAL = [
        ("raw_rot5", "Raw Grayscale + Rot(±5°) [Frozen Winner]", False),
        ("raw_no_aug", "Raw Grayscale (No Augmentation) [Baseline Control]", False),
        ("raw_rot10", "Raw Grayscale + Rot(±10°) [Legacy Stress Baseline]", False),
        ("clahe15_rot5", "CLAHE 1.5 + Rot(±5°) [Best Preprocessed Baseline]", True)
    ]
    SEEDS = [17, 42, 2026]

    all_predictions_rows = []
    run_summaries = []

    for model_code, model_desc, is_clahe in MODELS_TO_EVAL:
        ds = TestInferenceDataset(test_records, is_clahe=is_clahe)
        loader = DataLoader(
            ds,
            batch_size=16,
            shuffle=False,
            num_workers=0,
            pin_memory=True if device.type == "cuda" else False
        )

        for s in SEEDS:
            run_id = f"{model_code}_seed{s}"
            ckpt_path = EXPERIMENTS_DIR / run_id / "checkpoints" / "best_model.pth"
            if not ckpt_path.exists():
                raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")

            print(f"\nEvaluating: {model_code} | Seed: {s} | Checkpoint: {ckpt_path.name}")
            
            # Load Model
            model = models.resnet18(weights=None)
            model.fc = nn.Linear(model.fc.in_features, 4)
            state_dict = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(state_dict)
            model = model.to(device)
            model.eval()

            preds_list = []
            probs_list = []
            targets_list = []
            opaque_ids_list = []
            paths_list = []

            with torch.no_grad():
                for imgs, lbls, opaque_ids, paths in loader:
                    imgs = imgs.to(device, non_blocking=True)
                    outputs = model(imgs)
                    probs = torch.softmax(outputs, dim=1)
                    preds = outputs.argmax(dim=1)

                    preds_list.extend(preds.cpu().tolist())
                    probs_list.extend(probs.cpu().tolist())
                    targets_list.extend(lbls.tolist())
                    opaque_ids_list.extend(opaque_ids)
                    paths_list.extend(paths)

            # Store per-image predictions
            for i in range(len(targets_list)):
                all_predictions_rows.append({
                    "model_code": model_code,
                    "model_description": model_desc,
                    "seed": s,
                    "image_id": opaque_ids_list[i],
                    "true_label": targets_list[i],
                    "true_class_name": CLASS_NAMES[targets_list[i]],
                    "predicted_label": preds_list[i],
                    "predicted_class_name": CLASS_NAMES[preds_list[i]],
                    "is_correct": int(targets_list[i] == preds_list[i]),
                    "prob_mild": probs_list[i][0],
                    "prob_moderate": probs_list[i][1],
                    "prob_non": probs_list[i][2],
                    "prob_verymild": probs_list[i][3],
                    "relative_path_hash": hashlib.sha256(paths_list[i].encode()).hexdigest()
                })

            # Calculate Run Metrics
            n_total = len(targets_list)
            n_correct = int(np.sum(np.array(targets_list) == np.array(preds_list)))
            acc = accuracy_score(targets_list, preds_list)
            bal_acc = balanced_accuracy_score(targets_list, preds_list)
            macro_f1 = f1_score(targets_list, preds_list, average="macro", zero_division=0)
            weighted_f1 = f1_score(targets_list, preds_list, average="weighted", zero_division=0)
            
            recs = recall_score(targets_list, preds_list, average=None, zero_division=0)
            precs = precision_score(targets_list, preds_list, average=None, zero_division=0)
            f1s = f1_score(targets_list, preds_list, average=None, zero_division=0)
            cm = confusion_matrix(targets_list, preds_list, labels=[0, 1, 2, 3]).tolist()

            # Wilson CI for Accuracy
            acc_ci_low, acc_ci_high = wilson_score_interval(n_correct, n_total)

            print(
                f"  -> Accuracy: {acc*100:.2f}% ({n_correct}/{n_total}, 95% CI: [{acc_ci_low*100:.2f}%, {acc_ci_high*100:.2f}%]) | "
                f"Macro-F1: {macro_f1*100:.2f}% | BalAcc: {bal_acc*100:.2f}% | "
                f"Mod Recall: {recs[1]*100:.1f}% ({int(recs[1]*9)}/9)"
            )

            run_summaries.append({
                "model_code": model_code,
                "model_description": model_desc,
                "seed": s,
                "total_test_images": n_total,
                "correct_predictions": n_correct,
                "test_accuracy": acc,
                "accuracy_wilson_95_ci_lower": acc_ci_low,
                "accuracy_wilson_95_ci_upper": acc_ci_high,
                "test_balanced_accuracy": bal_acc,
                "test_macro_f1": macro_f1,
                "test_weighted_f1": weighted_f1,
                "moderate_recall": recs[1],
                "moderate_precision": precs[1],
                "moderate_f1": f1s[1],
                "mild_recall": recs[0],
                "mild_precision": precs[0],
                "mild_f1": f1s[0],
                "non_recall": recs[2],
                "non_precision": precs[2],
                "non_f1": f1s[2],
                "verymild_recall": recs[3],
                "verymild_precision": precs[3],
                "verymild_f1": f1s[3],
                "confusion_matrix": json.dumps(cm),
                "checkpoint_evaluated": str(ckpt_path.relative_to(REPO_ROOT))
            })

    # Save per-image predictions CSV
    preds_all_df = pd.DataFrame(all_predictions_rows)
    preds_all_csv = EVAL_DIR / "test_predictions_all_models.csv"
    preds_all_df.to_csv(preds_all_csv, index=False)
    print(f"\nSaved all per-image test predictions: {preds_all_csv} ({len(preds_all_df)} rows)")

    # Save predictions for frozen winner only
    winner_preds_df = preds_all_df[preds_all_df["model_code"] == "raw_rot5"].reset_index(drop=True)
    winner_preds_csv = EVAL_DIR / "test_predictions_frozen_winner.csv"
    winner_preds_df.to_csv(winner_preds_csv, index=False)
    print(f"Saved frozen winner per-image test predictions: {winner_preds_csv} ({len(winner_preds_df)} rows)")

    # Save run summaries CSV
    summaries_df = pd.DataFrame(run_summaries)
    summaries_csv = EVAL_DIR / "final_test_metrics_summary.csv"
    summaries_df.to_csv(summaries_csv, index=False)
    print(f"Saved run summaries: {summaries_csv}")

    # Compute Aggregates across Seeds (Mean ± Std)
    agg_df = summaries_df.groupby(["model_code", "model_description"]).agg({
        "test_accuracy": ["mean", "std"],
        "test_balanced_accuracy": ["mean", "std"],
        "test_macro_f1": ["mean", "std"],
        "test_weighted_f1": ["mean", "std"],
        "moderate_recall": ["mean", "std"],
        "moderate_precision": ["mean", "std"],
        "moderate_f1": ["mean", "std"],
        "mild_recall": ["mean", "std"],
        "non_recall": ["mean", "std"],
        "verymild_recall": ["mean", "std"]
    }).reset_index()

    agg_df.columns = [f"{c[0]}_{c[1]}" if c[1] else c[0] for c in agg_df.columns]
    agg_df = agg_df.sort_values(by="test_macro_f1_mean", ascending=False).reset_index(drop=True)
    
    agg_csv = EVAL_DIR / "final_test_aggregates.csv"
    agg_df.to_csv(agg_csv, index=False)
    print(f"Saved final test aggregates: {agg_csv}")

    print("\n=== FINAL TEST COHORT PERFORMANCE (MEAN ± STD, N=960) ===")
    print(agg_df[[
        "model_code", "test_macro_f1_mean", "test_macro_f1_std",
        "test_accuracy_mean", "test_accuracy_std",
        "test_balanced_accuracy_mean", "moderate_recall_mean"
    ]].to_string())

    # -------------------------------------------------------------
    # 5. DETAILED CONFIDENCE INTERVALS FOR FROZEN WINNER
    # -------------------------------------------------------------
    print("\nComputing 95% Bootstrap Confidence Intervals for Frozen Winner (`raw_rot5`)...")
    ci_results = {}
    for s in SEEDS:
        sub = winner_preds_df[winner_preds_df["seed"] == s]
        yt = sub["true_label"].tolist()
        yp = sub["predicted_label"].tolist()

        acc_ci = wilson_score_interval(int(np.sum(np.array(yt) == np.array(yp))), len(yt))
        bal_acc_ci = bootstrap_metric_ci(yt, yp, balanced_accuracy_score)
        macro_f1_ci = bootstrap_metric_ci(yt, yp, lambda a, b: f1_score(a, b, average="macro", zero_division=0))
        weighted_f1_ci = bootstrap_metric_ci(yt, yp, lambda a, b: f1_score(a, b, average="weighted", zero_division=0))

        ci_results[f"seed_{s}"] = {
            "test_accuracy": float(accuracy_score(yt, yp)),
            "accuracy_wilson_95_ci": list(acc_ci),
            "test_balanced_accuracy": float(balanced_accuracy_score(yt, yp)),
            "balanced_accuracy_bootstrap_95_ci": list(bal_acc_ci),
            "test_macro_f1": float(f1_score(yt, yp, average="macro", zero_division=0)),
            "macro_f1_bootstrap_95_ci": list(macro_f1_ci),
            "test_weighted_f1": float(f1_score(yt, yp, average="weighted", zero_division=0)),
            "weighted_f1_bootstrap_95_ci": list(weighted_f1_ci),
            "confusion_matrix": confusion_matrix(yt, yp, labels=[0, 1, 2, 3]).tolist()
        }

    with open(EVAL_DIR / "confidence_intervals.json", "w") as f:
        json.dump(ci_results, f, indent=2)
    print(f"Saved confidence intervals: {EVAL_DIR / 'confidence_intervals.json'}")

    # -------------------------------------------------------------
    # 6. PAIRED STATISTICAL COMPARISONS (McNEMAR'S TEST)
    # -------------------------------------------------------------
    print("\nComputing Paired McNemar's Tests on Fixed Test Rows...")
    paired_results = {}
    for s in SEEDS:
        w_sub = preds_all_df[(preds_all_df["model_code"] == "raw_rot5") & (preds_all_df["seed"] == s)].sort_values("image_id")
        base_sub = preds_all_df[(preds_all_df["model_code"] == "raw_no_aug") & (preds_all_df["seed"] == s)].sort_values("image_id")
        leg_sub = preds_all_df[(preds_all_df["model_code"] == "raw_rot10") & (preds_all_df["seed"] == s)].sort_values("image_id")
        clahe_sub = preds_all_df[(preds_all_df["model_code"] == "clahe15_rot5") & (preds_all_df["seed"] == s)].sort_values("image_id")

        yt = w_sub["true_label"].tolist()
        yp_winner = w_sub["predicted_label"].tolist()
        yp_base = base_sub["predicted_label"].tolist()
        yp_leg = leg_sub["predicted_label"].tolist()
        yp_clahe = clahe_sub["predicted_label"].tolist()

        paired_results[f"seed_{s}"] = {
            "winner_vs_raw_no_aug": mcnemar_test_paired(yt, yp_winner, yp_base),
            "winner_vs_raw_rot10_legacy": mcnemar_test_paired(yt, yp_winner, yp_leg),
            "winner_vs_clahe15_rot5": mcnemar_test_paired(yt, yp_winner, yp_clahe)
        }

    with open(EVAL_DIR / "mcnemar_paired_comparisons.json", "w") as f:
        json.dump(paired_results, f, indent=2)
    print(f"Saved McNemar's paired comparisons: {EVAL_DIR / 'mcnemar_paired_comparisons.json'}")

    # -------------------------------------------------------------
    # 7. GENERATE VISUALIZATION PANEL
    # -------------------------------------------------------------
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))

    # Panel 1: Test Accuracy and Macro-F1 by Model (Mean ± Std)
    ax1 = axes[0, 0]
    models_display = ["raw_rot5\n(Winner)", "clahe15_rot5", "raw_rot10\n(Legacy)", "raw_no_aug\n(Baseline)"]
    x = np.arange(len(models_display))
    w = 0.28
    ax1.bar(x - w/2, agg_df["test_macro_f1_mean"] * 100, yerr=agg_df["test_macro_f1_std"] * 100, width=w, label="Test Macro-F1 (%)", capsize=5, color="#1f77b4")
    ax1.bar(x + w/2, agg_df["test_accuracy_mean"] * 100, yerr=agg_df["test_accuracy_std"] * 100, width=w, label="Test Accuracy (%)", capsize=5, color="#ff7f0e")
    ax1.set_xticks(x)
    ax1.set_xticklabels(models_display, fontsize=9)
    ax1.set_ylim(92, 100)
    ax1.set_ylabel("Metric Score (%)")
    ax1.set_title("Test Cohort Performance Across Models (Mean ± Std, N=960)")
    ax1.legend(loc="lower right")
    ax1.grid(axis="y", linestyle="--", alpha=0.5)

    # Panel 2: Test Macro-F1 by Seed
    ax2 = axes[0, 1]
    seeds_list = [17, 42, 2026]
    w = 0.2
    x_s = np.arange(len(seeds_list))
    for idx, (m_code, _, _) in enumerate(MODELS_TO_EVAL):
        sub = summaries_df[summaries_df["model_code"] == m_code].set_index("seed").reindex(seeds_list)
        ax2.bar(x_s + (idx - 1.5)*w, sub["test_macro_f1"] * 100, width=w, label=m_code)
    ax2.set_xticks(x_s)
    ax2.set_xticklabels([f"Seed {s}" for s in seeds_list])
    ax2.set_ylim(92, 100)
    ax2.set_ylabel("Test Macro-F1 (%)")
    ax2.set_title("Seed-to-Seed Test Macro-F1 Consistency")
    ax2.legend(fontsize=8, loc="lower right")
    ax2.grid(axis="y", linestyle="--", alpha=0.5)

    # Panel 3: Class-wise Recall for Winner on Test Cohort
    ax3 = axes[1, 0]
    winner_sub = summaries_df[summaries_df["model_code"] == "raw_rot5"].set_index("seed").reindex(seeds_list)
    classes_abbr = ["Mild (135)", "Mod (9)", "Non (480)", "VeryMild (336)"]
    w = 0.22
    for s_idx, s in enumerate(seeds_list):
        row = winner_sub.loc[s]
        vals = [row["mild_recall"]*100, row["moderate_recall"]*100, row["non_recall"]*100, row["verymild_recall"]*100]
        ax3.bar(np.arange(4) + (s_idx - 1)*w, vals, width=w, label=f"Seed {s}")
    ax3.set_xticks(range(4))
    ax3.set_xticklabels(classes_abbr)
    ax3.set_ylim(85, 102)
    ax3.set_ylabel("Recall (%)")
    ax3.set_title("Test Class-Wise Recall: Winning raw_rot5 Pipeline")
    ax3.legend(loc="lower left")
    ax3.grid(axis="y", linestyle="--", alpha=0.5)

    # Panel 4: Confusion Matrix for Top Test Run (raw_rot5_seed42)
    ax4 = axes[1, 1]
    top_run = summaries_df[summaries_df["model_code"] == "raw_rot5"].sort_values("test_macro_f1", ascending=False).iloc[0]
    top_cm = np.array(json.loads(top_run["confusion_matrix"]))
    cax = ax4.imshow(top_cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(cax, ax=ax4)
    ax4.set_xticks(range(4))
    ax4.set_yticks(range(4))
    classes_short = ["Mild", "Mod", "Non", "VeryMild"]
    ax4.set_xticklabels(classes_short)
    ax4.set_yticklabels(classes_short)
    for i in range(4):
        for j in range(4):
            ax4.text(j, i, str(top_cm[i, j]), ha="center", va="center", color="white" if top_cm[i, j] > top_cm.max()/2 else "black")
    ax4.set_xlabel("Predicted Class")
    ax4.set_ylabel("Actual Class")
    ax4.set_title(f"Test Confusion Matrix: raw_rot5_seed{top_run['seed']} (Acc: {top_run['test_accuracy']*100:.2f}%)")

    plt.tight_layout()
    panel_out = EVAL_DIR / "final_test_evaluation_panel.png"
    plt.savefig(panel_out, dpi=300)
    print(f"Saved test evaluation panel to {panel_out}")

if __name__ == "__main__":
    run_stage7_evaluation()
