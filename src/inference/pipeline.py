import os
import io
import time
import uuid
import base64
import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

from src.models.resnet import create_model
from src.explainability.gradcam import GradCAM

# Official class labels
CLASS_NAMES = [
    "Mild Demented",
    "Moderate Demented",
    "Non Demented",
    "Very Mild Demented"
]

CHECKPOINT_CONFIGS = {
    "baseline": {
        "id": "baseline",
        "name": "Baseline ResNet18",
        "description": "Standard ResNet18 trained on original unenhanced MRI scans",
        "path": r"checkpoints\resnet18_baseline.pth",
        "preprocessing": "original",
        "benchmark_acc": "94.38%",
        "benchmark_macro_f1": "95.30%"
    },
    "he": {
        "id": "he",
        "name": "Histogram Equalization (HE) ResNet18",
        "description": "ResNet18 trained on global histogram-equalized MRI scans",
        "path": r"checkpoints\resnet18_he.pth",
        "preprocessing": "HE",
        "benchmark_acc": "78.23%",
        "benchmark_macro_f1": "79.76%"
    },
    "clahe": {
        "id": "clahe",
        "name": "CLAHE ResNet18",
        "description": "ResNet18 trained on Contrast Limited Adaptive Histogram Equalization",
        "path": r"checkpoints\resnet18_clahe.pth",
        "preprocessing": "CLAHE",
        "benchmark_acc": "95.10%",
        "benchmark_macro_f1": "94.61%"
    }
}

RUNS_DIR = os.path.abspath(os.path.join("results", "app_runs"))
os.makedirs(RUNS_DIR, exist_ok=True)


class MRIAnalysisEngine:
    _instance = None

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[MRIAnalysisEngine] Initializing on device: {self.device}")

        # Standard ImageNet normalization matching dataset.py
        self.transform_norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        self.models = {}
        self.gradcams = {}
        self._load_models()

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _load_models(self):
        """Loads all three trained models once and registers Grad-CAM hooks."""
        for key, cfg in CHECKPOINT_CONFIGS.items():
            ckpt_path = cfg["path"]
            if not os.path.exists(ckpt_path):
                raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}")

            model = create_model(num_classes=4)
            state_dict = torch.load(
                ckpt_path,
                map_location=self.device,
                weights_only=True
            )
            model.load_state_dict(state_dict)
            model = model.to(self.device)
            model.eval()

            # Target the final convolutional block of ResNet18
            target_layer = model.layer4[-1]
            gradcam = GradCAM(model, target_layer)

            self.models[key] = model
            self.gradcams[key] = gradcam
            print(f"[MRIAnalysisEngine] Successfully loaded {cfg['name']} from {ckpt_path}")

    @staticmethod
    def apply_preprocessing(raw_gray_224, mode: str):
        """Applies exact preprocessing matching dataset.py."""
        if mode == "HE":
            prep_gray = cv2.equalizeHist(raw_gray_224)
        elif mode == "CLAHE":
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            prep_gray = clahe.apply(raw_gray_224)
        else:
            prep_gray = raw_gray_224.copy()

        # Convert grayscale to RGB for ResNet18
        prep_rgb = cv2.cvtColor(prep_gray, cv2.COLOR_GRAY2RGB)
        return prep_gray, prep_rgb

    @staticmethod
    def encode_image_base64(image_np, is_bgr=False):
        """Encodes an RGB or grayscale numpy image into a base64 JPEG data URL."""
        if is_bgr:
            to_encode = image_np
        elif len(image_np.shape) == 2:
            to_encode = image_np
        else:
            to_encode = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        success, buffer = cv2.imencode(".jpg", to_encode, [cv2.IMWRITE_JPEG_QUALITY, 92])
        if not success:
            return None
        b64_str = base64.b64encode(buffer).decode("utf-8")
        return f"data:image/jpeg;base64,{b64_str}"

    def analyze(self, image_bytes: bytes, filename: str = "uploaded_mri.jpg"):
        """Executes full three-model inference, Grad-CAM generation, and agreement analysis."""
        # 1. Decode raw image
        t_start = time.time()
        np_arr = np.frombuffer(image_bytes, np.uint8)
        raw_gray = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)

        if raw_gray is None:
            raise ValueError("The provided file could not be decoded as a valid MRI image.")

        orig_h, orig_w = raw_gray.shape
        run_id = f"run_{int(time.time())}_{uuid.uuid4().hex[:6]}"
        run_dir = os.path.join(RUNS_DIR, run_id)
        os.makedirs(run_dir, exist_ok=True)

        print("\n" + "=" * 76)
        print(f" [CLINICAL INFERENCE DISPATCH] Run ID: {run_id}")
        print(f" Timestamp : {time.strftime('%Y-%m-%d %H:%M:%S')} UTC | Source: {filename} ({orig_w}x{orig_h})")
        print(f" Hardware  : {str(self.device).upper()} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
        print("-" * 76)

        # Resize to 224x224 as required by the ResNet18 pipeline
        raw_224 = cv2.resize(raw_gray, (224, 224), interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(os.path.join(run_dir, "original_raw.png"), raw_224)

        # 2. Preprocess into 3 modalities
        prep_images = {}
        prep_tensors = {}
        for key, cfg in CHECKPOINT_CONFIGS.items():
            prep_gray, prep_rgb = self.apply_preprocessing(raw_224, cfg["preprocessing"])
            prep_images[key] = {
                "gray": prep_gray,
                "rgb": prep_rgb,
                "b64": self.encode_image_base64(prep_gray)
            }
            cv2.imwrite(os.path.join(run_dir, f"prep_{key}.png"), prep_gray)

            pil_img = Image.fromarray(prep_rgb)
            tensor = self.transform_norm(pil_img).unsqueeze(0).to(self.device)
            prep_tensors[key] = tensor

        # 3. Model Inference & Grad-CAM for each model
        model_results = {}
        predicted_classes = {}

        for key, cfg in CHECKPOINT_CONFIGS.items():
            tensor = prep_tensors[key]
            prep_rgb = prep_images[key]["rgb"]
            gradcam = self.gradcams[key]

            # Generate Grad-CAM (and forward pass)
            heatmap, output = gradcam.generate(tensor)

            probs = torch.softmax(output, dim=1)[0].cpu().numpy().tolist()
            pred_idx = int(np.argmax(probs))
            pred_class = CLASS_NAMES[pred_idx]
            conf = float(probs[pred_idx])
            predicted_classes[key] = pred_class

            # Create colored heatmap & overlay
            heatmap_uint8 = np.uint8(255 * heatmap)
            colored_heatmap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
            colored_heatmap_rgb = cv2.cvtColor(colored_heatmap, cv2.COLOR_BGR2RGB)

            # Overlay: 60% MRI + 40% Grad-CAM heatmap
            overlay_rgb = cv2.addWeighted(prep_rgb, 0.6, colored_heatmap_rgb, 0.4, 0)

            # Save visualization assets
            heatmap_path = os.path.join(run_dir, f"heatmap_{key}.png")
            overlay_path = os.path.join(run_dir, f"overlay_{key}.png")
            cv2.imwrite(heatmap_path, colored_heatmap)
            cv2.imwrite(overlay_path, cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR))

            # Probability mapping
            prob_dict = {CLASS_NAMES[i]: round(float(probs[i]), 4) for i in range(4)}

            model_results[key] = {
                "id": key,
                "name": cfg["name"],
                "preprocessing": cfg["preprocessing"],
                "predicted_class": pred_class,
                "predicted_index": pred_idx,
                "confidence": round(conf, 4),
                "confidence_pct": round(conf * 100, 2),
                "probabilities": prob_dict,
                "benchmark_acc": cfg["benchmark_acc"],
                "benchmark_macro_f1": cfg["benchmark_macro_f1"],
                "visualizations": {
                    "original_b64": self.encode_image_base64(raw_224),
                    "preprocessed_b64": prep_images[key]["b64"],
                    "heatmap_b64": self.encode_image_base64(colored_heatmap_rgb),
                    "overlay_b64": self.encode_image_base64(overlay_rgb),
                    "heatmap_file": os.path.relpath(heatmap_path, os.getcwd()),
                    "overlay_file": os.path.relpath(overlay_path, os.getcwd())
                }
            }

        # 4. Model Agreement & Consensus Analysis
        preds = list(predicted_classes.values())
        unique_preds = set(preds)
        
        if len(unique_preds) == 1:
            agreement_status = "Full Consensus (3/3 Models Agree)"
            agreement_level = "high"
            consensus_class = preds[0]
            agreement_summary = (
                f"All three independent models unanimously classified this MRI as {consensus_class}. "
                "Both standard baseline and contrast-enhanced networks arrived at identical diagnostic categorizations."
            )
        elif len(unique_preds) == 2:
            agreement_status = "Majority Agreement (2/3 Models Agree)"
            agreement_level = "medium"
            # Find majority class
            consensus_class = max(set(preds), key=preds.count)
            dissenting_models = [k for k, v in predicted_classes.items() if v != consensus_class]
            agreeing_models = [k for k, v in predicted_classes.items() if v == consensus_class]
            
            dissenter_str = ", ".join([CHECKPOINT_CONFIGS[m]["name"] for m in dissenting_models])
            agreeing_str = " and ".join([CHECKPOINT_CONFIGS[m]["name"] for m in agreeing_models])
            dissenting_pred = predicted_classes[dissenting_models[0]]

            agreement_summary = (
                f"Majority consensus ({consensus_class}) reached by {agreeing_str}. "
                f"The dissenting model ({dissenter_str}) predicted {dissenting_pred}. "
                "This divergence frequently indicates subtle borderline morphology (e.g. between Very Mild and Mild impairment)."
            )
        else:
            agreement_status = "Divergent Predictions (3-Way Split)"
            agreement_level = "low"
            consensus_class = "None (Disagreement)"
            agreement_summary = (
                "Each of the three models produced a differing classification. "
                "This indicates high diagnostic ambiguity in tissue contrast or anatomical boundary presentation."
            )

        agreement_details = {
            "status": agreement_status,
            "level": agreement_level,
            "consensus_class": consensus_class,
            "summary": agreement_summary,
            "predictions_by_model": predicted_classes,
            "unanimous": (len(unique_preds) == 1)
        }

        # 5. Preprocessing Comparison Pack
        preprocessing_comparison = {
            "original": self.encode_image_base64(raw_224),
            "he": prep_images["he"]["b64"],
            "clahe": prep_images["clahe"]["b64"]
        }

        elapsed_ms = (time.time() - t_start) * 1000.0
        print(f" [PYTORCH FORWARD PASSES & GRAD-CAM BACKPROP COMPLETED IN {elapsed_ms:.1f}ms]")
        for k in ["baseline", "he", "clahe"]:
            m = model_results[k]
            print(f"  * {m['name']:<36} : {m['predicted_class']:<20} ({m['confidence_pct']:>5.1f}%)")
        print(f" [CONSENSUS] {agreement_status} -> {consensus_class}")
        print(f" [DISK AUDIT] 9 artifacts saved to: {run_dir}")
        print("=" * 76 + "\n")

        response_payload = {
            "run_id": run_id,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            "filename": filename,
            "latency_ms": round(elapsed_ms, 1),
            "image_dimensions": f"{orig_w}x{orig_h}",
            "device": str(self.device),
            "class_names": CLASS_NAMES,
            "models": model_results,
            "agreement": agreement_details,
            "preprocessing": preprocessing_comparison,
            "run_dir": run_dir
        }

        return response_payload
