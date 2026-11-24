import os
import base64
import urllib.request
from PIL import Image

diagrams = {
    'diagram_pipeline.png': """flowchart LR
    A["Raw 2D Axial MRI<br/>128x128 Grayscale"] --> B["Bilinear Rescale<br/>224x224"]
    B --> C["3-Channel Replication<br/>R=G=B (ImageNet Norm)"]
    C --> D{"Execution Mode"}
    D -->|"Training"| E["Random Rotation<br/>[-5 deg, +5 deg]"]
    D -->|"Inference"| F["Deterministic Identity"]
    E --> G["ResNet-18 Backbone<br/>Pretrained ImageNet"]
    F --> G
    G --> H["Global Average Pool<br/>512-dim Vector"]
    H --> I["Linear Head<br/>4 Disease Classes"]
    I --> J["Inverse-Weighted CE Loss<br/>w_mod = 24.89"]
    G -.-> K["Layer-4 Grad-CAM<br/>model.layer4[-1]"]
    I -.-> K
    K --> L["Visual Heatmap<br/>Ventricle and Temporal Saliency"]
""",
    'diagram_resnet.png': """flowchart LR
    In["Input Tensor<br/>3 x 224 x 224"] --> Stem["Stem Layer<br/>Conv1 7x7 (64-ch, s2)<br/>MaxPool 3x3 (s2)"]
    Stem --> L1["Stage 1 (layer1)<br/>2x BasicBlock (64-ch)<br/>Output: 56x56x64"]
    L1 --> L2["Stage 2 (layer2)<br/>2x BasicBlock (128-ch)<br/>Output: 28x28x128"]
    L2 --> L3["Stage 3 (layer3)<br/>2x BasicBlock (256-ch)<br/>Output: 14x14x256"]
    L3 --> L4["Stage 4 (layer4)<br/>2x BasicBlock (512-ch)<br/>Output: 7x7x512<br/>[XAI Hook Point]"]
    L4 --> GAP["Global Avg Pool<br/>Spatial 7x7 Collapse<br/>512-dim Vector"]
    GAP --> FC["Linear Head<br/>nn.Linear(512, 4)<br/>Logits: 4 Disease Classes"]
""",
    'diagram_gradcam.png': """flowchart LR
    Target["Target Class Logit Y_c"] -->|"Backprop Gradients"| Grads["Gradients dY_c / dA^k<br/>at model.layer4[-1]"]
    Grads -->|"Global Avg Pooling"| Weights["Neuron Weights alpha_k^c<br/>(1/UV) sum(dY_c / dA_ij^k)"]
    Act["Activations A^k (512 Maps)<br/>Forward Hook on layer4[-1]"] --> LinComb
    Weights --> LinComb["Weighted Linear Combination<br/>sum(alpha_k^c * A^k)"]
    LinComb --> Relu["ReLU Rectification<br/>(Positive Features)"]
    Relu --> Interp["Bilinear Upsampling<br/>7x7 -> 224x224"]
    Interp --> Norm["Min-Max Normalization<br/>[0.0, 1.0]"]
    Norm --> Heatmap["Heatmap Overlay<br/>Ventricles & Hippocampus"]
""",
    'diagram_7stages.png': """flowchart LR
    subgraph P1 ["Phase 1: Foundations"]
        S1["Stage 1: Data Audit<br/>6,400 scans, 70/15/15 split"] --> S2["Stage 2: Reproducibility<br/>Seeds 17, 42, 2026 (95.49%)"]
    end
    subgraph P2 ["Phase 2: Ablation Studies"]
        S3["Stage 3: Preprocessing<br/>Raw vs. HE vs. CLAHE"] --> S4["Stage 4: Augmentation<br/>Over-Augmentation (±5° Won)"]
    end
    subgraph P3 ["Phase 3: Optimization & Test"]
        S5["Stage 5: Imbalance Weights<br/>100% Moderate Recall"] --> S6["Stage 6: Validation Lock<br/>raw_rot5 Rank 1 (97.66%)"]
        S6 --> S7["Stage 7: Final Test<br/>97.50% Peak Acc (p=0.0015)"]
    end
    P1 --> P2 --> P3
"""
}

out_dir = os.path.join('results', 'architecture_diagrams')
os.makedirs(out_dir, exist_ok=True)

for name, code in diagrams.items():
    b64 = base64.b64encode(code.strip().encode('utf-8')).decode('ascii')
    url = f'https://mermaid.ink/img/{b64}?bgColor=FFFFFF'
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = resp.read()
            out_path = os.path.join(out_dir, name)
            with open(out_path, 'wb') as f:
                f.write(data)
            im = Image.open(out_path)
            print(f'Generated {name}: size={im.size}')
    except Exception as e:
        print(f'Error generating {name}: {e}')
