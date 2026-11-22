
import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd

from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from src.data.dataset import BrainMRIDataset
from src.models.resnet import create_model


# Configuration
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = str(REPO_ROOT / "Dataset" / "splits" / "splits.csv")
SAVE_PATH = str(REPO_ROOT / "checkpoints")

BATCH_SIZE = 16
EPOCHS = 8
LEARNING_RATE = 0.0001

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

os.makedirs(SAVE_PATH, exist_ok=True)

print("Using device:", DEVICE)


# Load datasets
train_dataset = BrainMRIDataset(
    CSV_PATH,
    split="train",
    preprocessing="HE"
)

val_dataset = BrainMRIDataset(
    CSV_PATH,
    split="validation",
    preprocessing="HE"
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)


# Calculate class weights
df = pd.read_csv(CSV_PATH)
train_df = df[df["split"] == "train"]

class_counts = (
    train_df["label"]
    .value_counts()
    .sort_index()
)

class_weights = (
    len(train_df) / (4 * class_counts)
).values

class_weights = torch.tensor(
    class_weights,
    dtype=torch.float32,
    device=DEVICE
)

print("Class weights:", class_weights)


# Create model
model = create_model(num_classes=4)
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss(
    weight=class_weights
)

optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# Training
best_f1 = -1

for epoch in range(EPOCHS):

    model.train()

    running_loss = 0.0

    for images, labels in train_loader:

        images = images.to(
            DEVICE,
            non_blocking=True
        )

        labels = labels.to(
            DEVICE,
            non_blocking=True
        )

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    train_loss = running_loss / len(train_loader)


    # Validation
    model.eval()

    all_labels = []
    all_predictions = []

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            all_labels.extend(labels.cpu().tolist())
            all_predictions.extend(
                predictions.cpu().tolist()
            )

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

    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] "
        f"Loss: {train_loss:.4f} "
        f"Val Accuracy: {accuracy:.4f} "
        f"Val Macro F1: {macro_f1:.4f}"
    )

    # Save best model based on macro F1
    if macro_f1 > best_f1:

        best_f1 = macro_f1

        torch.save(
            model.state_dict(),
            os.path.join(
                SAVE_PATH,
                "resnet18_he.pth"
            )
        )

        print("Best model saved!")


print("\nHE Training completed.")

# Final validation report using best checkpoint
model.load_state_dict(
    torch.load(
        os.path.join(
            SAVE_PATH,
            "resnet18_he.pth"
        ),
        map_location=DEVICE,
        weights_only=True
    )
)

model.eval()

all_labels = []
all_predictions = []

with torch.no_grad():

    for images, labels in val_loader:

        images = images.to(DEVICE)

        outputs = model(images)

        predictions = outputs.argmax(dim=1)

        all_labels.extend(labels.tolist())
        all_predictions.extend(
            predictions.cpu().tolist()
        )

print("\nValidation Classification Report:")

print(
    classification_report(
        all_labels,
        all_predictions,
        target_names=[
            "Mild",
            "Moderate",
            "Non Demented",
            "Very Mild"
        ],
        zero_division=0
    )
)

print("\nConfusion Matrix:")

print(
    confusion_matrix(
        all_labels,
        all_predictions
    )
)

print("\nBest validation Macro F1:", best_f1)
