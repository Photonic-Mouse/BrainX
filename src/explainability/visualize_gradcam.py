import os
import cv2
import torch
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

from src.models.resnet import create_model
from src.explainability.gradcam import GradCAM


# ==============================
# CONFIGURATION
# ==============================

CSV_PATH = r"C:\HEHAW\BrainX\dataset\splits\splits.csv"
OUTPUT_DIR = r"C:\HEHAW\BrainX\results\gradcam"

CHECKPOINTS = {
    "baseline": {
        "path": r"C:\HEHAW\BrainX\checkpoints\resnet18_baseline.pth",
        "preprocessing": "original",
        "display_name": "Original MRI (Baseline)"
    },
    "he": {
        "path": r"C:\HEHAW\BrainX\checkpoints\resnet18_he.pth",
        "preprocessing": "HE",
        "display_name": "Histogram Equalization (HE)"
    },
    "clahe": {
        "path": r"C:\HEHAW\BrainX\checkpoints\resnet18_clahe.pth",
        "preprocessing": "CLAHE",
        "display_name": "CLAHE"
    }
}

CLASS_NAMES = [
    "Mild",
    "Moderate",
    "Non Demented",
    "Very Mild"
]

SAMPLES_PER_CLASS = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform_norm = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def preprocess_image(raw_gray_224, mode):
    """Apply the specified MRI preprocessing to a 224x224 grayscale uint8 image."""
    if mode == "HE":
        prep_gray = cv2.equalizeHist(raw_gray_224)
    elif mode == "CLAHE":
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        prep_gray = clahe.apply(raw_gray_224)
    else:  # original
        prep_gray = raw_gray_224.copy()

    rgb = cv2.cvtColor(prep_gray, cv2.COLOR_GRAY2RGB)
    pil_img = Image.fromarray(rgb)
    tensor = transform_norm(pil_img)
    return prep_gray, rgb, tensor


def main():
    print(f"Using device: {DEVICE}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Load and group test samples
    df = pd.read_csv(CSV_PATH)
    test_df = df[df["split"] == "test"].copy()

    selected_samples = []
    for label_idx, class_name in enumerate(CLASS_NAMES):
        class_subset = test_df[test_df["label"] == label_idx]
        chosen = class_subset.head(SAMPLES_PER_CLASS)
        for _, row in chosen.iterrows():
            selected_samples.append({
                "image_path": row["image_path"],
                "label": label_idx,
                "class_name": class_name
            })

    print(f"Selected {len(selected_samples)} test samples ({SAMPLES_PER_CLASS} per class).")

    # 2. Load models and initialize GradCAM instances
    models = {}
    gradcams = {}
    for key, cfg in CHECKPOINTS.items():
        m = create_model(num_classes=4)
        m.load_state_dict(
            torch.load(cfg["path"], map_location=DEVICE, weights_only=True)
        )
        m = m.to(DEVICE)
        m.eval()
        models[key] = m
        gradcams[key] = GradCAM(m, m.layer4[-1])
        print(f"Loaded model for: {key} ({cfg['path']})")

    # 3. Create per-model directories
    for key in CHECKPOINTS.keys():
        for cname in CLASS_NAMES:
            os.makedirs(os.path.join(OUTPUT_DIR, key, cname), exist_ok=True)
    for cname in CLASS_NAMES:
        os.makedirs(os.path.join(OUTPUT_DIR, "comparison", cname), exist_ok=True)

    # 4. Generate visualizations
    total_individual_saved = 0
    total_comparison_saved = 0

    class_sample_counters = {cname: 0 for cname in CLASS_NAMES}

    for item in selected_samples:
        img_path = item["image_path"]
        label = item["label"]
        actual_name = item["class_name"]
        class_sample_counters[actual_name] += 1
        sample_num = class_sample_counters[actual_name]

        # Read raw grayscale and resize to 224x224
        raw = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if raw is None:
            print(f"Warning: unable to read {img_path}")
            continue
        raw_224 = cv2.resize(raw, (224, 224), interpolation=cv2.INTER_LINEAR)

        sample_model_results = {}

        for key, cfg in CHECKPOINTS.items():
            prep_gray, prep_rgb, tensor = preprocess_image(raw_224, cfg["preprocessing"])
            input_tensor = tensor.unsqueeze(0).to(DEVICE)

            heatmap, output = gradcams[key].generate(input_tensor)

            probs = torch.softmax(output, dim=1)[0]
            pred_idx = output.argmax(dim=1).item()
            pred_name = CLASS_NAMES[pred_idx]
            conf = probs[pred_idx].item()

            # Heatmap colorization & overlay
            heatmap_uint8 = np.uint8(255 * heatmap)
            colored_heatmap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
            colored_heatmap = cv2.cvtColor(colored_heatmap, cv2.COLOR_BGR2RGB)

            overlay = cv2.addWeighted(prep_rgb, 0.6, colored_heatmap, 0.4, 0)

            sample_model_results[key] = {
                "prep_gray": prep_gray,
                "prep_rgb": prep_rgb,
                "heatmap": heatmap,
                "overlay": overlay,
                "pred_name": pred_name,
                "conf": conf,
                "is_correct": (pred_idx == label)
            }

            # Individual 4-panel visualization
            fig, axes = plt.subplots(1, 4, figsize=(16, 4))
            axes[0].imshow(raw_224, cmap="gray")
            axes[0].set_title("Original MRI")
            axes[0].axis("off")

            axes[1].imshow(prep_gray, cmap="gray")
            axes[1].set_title(f"Preprocessed ({cfg['preprocessing']})")
            axes[1].axis("off")

            axes[2].imshow(heatmap, cmap="jet")
            axes[2].set_title("Grad-CAM Heatmap")
            axes[2].axis("off")

            axes[3].imshow(overlay)
            axes[3].set_title("Heatmap Overlay")
            axes[3].axis("off")

            status = "Correct" if pred_idx == label else "Incorrect"
            fig.suptitle(
                f"Model: {cfg['display_name']} | Actual: {actual_name} | "
                f"Predicted: {pred_name} ({conf:.2%}) [{status}]",
                fontsize=12, y=1.02
            )
            plt.tight_layout()

            save_file = os.path.join(
                OUTPUT_DIR, key, actual_name, f"sample_{sample_num}.png"
            )
            plt.savefig(save_file, dpi=180, bbox_inches="tight")
            plt.close()
            total_individual_saved += 1

        # Direct 3-model comparative panel (3 rows x 4 columns)
        fig, axes = plt.subplots(3, 4, figsize=(16, 12))
        model_keys = ["baseline", "he", "clahe"]

        for row_idx, m_key in enumerate(model_keys):
            res = sample_model_results[m_key]
            m_disp = CHECKPOINTS[m_key]["display_name"]
            status = "Correct" if res["is_correct"] else "Incorrect"

            axes[row_idx, 0].imshow(raw_224, cmap="gray")
            axes[row_idx, 0].set_title(f"{m_disp}\nOriginal MRI")
            axes[row_idx, 0].axis("off")

            axes[row_idx, 1].imshow(res["prep_gray"], cmap="gray")
            axes[row_idx, 1].set_title(f"Preprocessed ({CHECKPOINTS[m_key]['preprocessing']})")
            axes[row_idx, 1].axis("off")

            axes[row_idx, 2].imshow(res["heatmap"], cmap="jet")
            axes[row_idx, 2].set_title("Grad-CAM Heatmap")
            axes[row_idx, 2].axis("off")

            axes[row_idx, 3].imshow(res["overlay"])
            axes[row_idx, 3].set_title(f"Overlay\nPred: {res['pred_name']} ({res['conf']:.1%}) [{status}]")
            axes[row_idx, 3].axis("off")

        fig.suptitle(
            f"Explainability Comparison across Models | Class: {actual_name} | Sample #{sample_num}",
            fontsize=15, y=0.99
        )
        plt.tight_layout()

        comp_save_file = os.path.join(
            OUTPUT_DIR, "comparison", actual_name, f"sample_{sample_num}_comparison.png"
        )
        plt.savefig(comp_save_file, dpi=180, bbox_inches="tight")
        plt.close()
        total_comparison_saved += 1

    print("\n========== GRAD-CAM GENERATION SUMMARY ==========")
    print(f"Total individual model visualizations saved: {total_individual_saved} (Target: 60)")
    print(f"Total multi-model comparison figures saved: {total_comparison_saved} (Target: 20)")
    print(f"All visualizations saved under: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
