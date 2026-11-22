import os
import torch
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score
)

from src.data.dataset import BrainMRIDataset
from src.models.resnet import create_model


CSV_PATH = r"C:\HEHAW\BrainX\dataset\splits\splits.csv"
MODEL_PATH = r"C:\HEHAW\BrainX\checkpoints\resnet18_clahe.pth"
OUTPUT_PATH = r"C:\HEHAW\BrainX\results\clahe"

os.makedirs(OUTPUT_PATH, exist_ok=True)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

class_names = [
    "Mild",
    "Moderate",
    "Non Demented",
    "Very Mild"
]

# Load test dataset
test_dataset = BrainMRIDataset(
    CSV_PATH,
    split="test",
    preprocessing="CLAHE"
)

test_loader = DataLoader(
    test_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=0
)

# Load trained model
model = create_model(num_classes=4)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True
    )
)

model = model.to(DEVICE)
model.eval()

all_labels = []
all_predictions = []

# Test evaluation
with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(DEVICE)

        outputs = model(images)

        predictions = outputs.argmax(dim=1)

        all_labels.extend(labels.tolist())
        all_predictions.extend(
            predictions.cpu().tolist()
        )

# Metrics
accuracy = accuracy_score(
    all_labels,
    all_predictions
)

macro_f1 = f1_score(
    all_labels,
    all_predictions,
    average="macro",
    zero_division=0
)

print("\n========== CLAHE TEST RESULTS ==========")

print(f"Test Accuracy: {accuracy:.4f}")
print(f"Test Macro F1: {macro_f1:.4f}")

print("\nClassification Report:")

print(
    classification_report(
        all_labels,
        all_predictions,
        labels=[0, 1, 2, 3],
        target_names=class_names,
        zero_division=0
    )
)

# Confusion matrix
cm = confusion_matrix(
    all_labels,
    all_predictions,
    labels=[0, 1, 2, 3]
)

print("\nConfusion Matrix:")
print(cm)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

fig, ax = plt.subplots(figsize=(9, 7))

disp.plot(
    ax=ax,
    cmap="Blues",
    values_format="d",
    xticks_rotation=30
)

plt.title("ResNet18 CLAHE - Test Confusion Matrix")
plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_PATH, "confusion_matrix.png"),
    dpi=200
)

plt.close()

# Save predictions
results = pd.DataFrame({
    "actual": all_labels,
    "predicted": all_predictions
})

results.to_csv(
    os.path.join(OUTPUT_PATH, "test_predictions.csv"),
    index=False
)

# Save classification report as text
report_text = classification_report(
    all_labels,
    all_predictions,
    labels=[0, 1, 2, 3],
    target_names=class_names,
    zero_division=0
)

with open(os.path.join(OUTPUT_PATH, "classification_report.txt"), "w") as f:
    f.write(f"Test Accuracy: {accuracy:.4f}\n")
    f.write(f"Test Macro F1: {macro_f1:.4f}\n\n")
    f.write("Classification Report:\n")
    f.write(report_text)
    f.write("\n\nConfusion Matrix:\n")
    f.write(str(cm))
    f.write("\n")

print("\nResults saved at:", OUTPUT_PATH)
