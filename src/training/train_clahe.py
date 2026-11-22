
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


# ==============================
# CONFIGURATION
# ==============================
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = str(REPO_ROOT / "Dataset" / "splits" / "splits.csv")
SAVE_PATH = str(REPO_ROOT / "checkpoints")
RESULT_PATH = str(REPO_ROOT / "results" / "clahe")

PREPROCESSING = "CLAHE"

MODEL_NAME = "resnet18_clahe.pth"

BATCH_SIZE = 16
EPOCHS = 8
LEARNING_RATE = 0.0001

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

os.makedirs(SAVE_PATH, exist_ok=True)
os.makedirs(RESULT_PATH, exist_ok=True)

print("Using device:", DEVICE)
print("Preprocessing:", PREPROCESSING)


# ==============================
# LOAD DATASETS
# ==============================

train_dataset = BrainMRIDataset(
    CSV_PATH,
    split="train",
    preprocessing=PREPROCESSING
)

val_dataset = BrainMRIDataset(
    CSV_PATH,
    split="validation",
    preprocessing=PREPROCESSING
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


# ==============================
# CLASS WEIGHTS
# ==============================

df = pd.read_csv(CSV_PATH)

train_df = df[df["split"] == "train"]

class_counts = (
    train_df["label"]
    .value_counts()
    .reindex(range(4), fill_value=0)
)

if (class_counts == 0).any():
    raise ValueError("A class is missing from training data.")

class_weights = (
    len(train_df) / (4 * class_counts)
).values

class_weights = torch.tensor(
    class_weights,
    dtype=torch.float32,
    device=DEVICE
)

print("Class weights:", class_weights)


# ==============================
# CREATE MODEL
# ==============================

model = create_model(num_classes=4)
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss(
    weight=class_weights
)

optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ==============================
# TRAINING
# ==============================

best_f1 = -1

history = []

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


    # ==============================
    # VALIDATION
    # ==============================

    model.eval()

    all_labels = []
    all_predictions = []

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            predictions = outputs.argmax(dim=1)

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

    history.append({
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "val_accuracy": accuracy,
        "val_macro_f1": macro_f1
    })


    # ==============================
    # SAVE BEST MODEL
    # ==============================

    if macro_f1 > best_f1:

        best_f1 = macro_f1

        torch.save(
            model.state_dict(),
            os.path.join(SAVE_PATH, MODEL_NAME)
        )

        print("Best model saved!")


# ==============================
# SAVE TRAINING HISTORY
# ==============================

history_df = pd.DataFrame(history)

history_df.to_csv(
    os.path.join(RESULT_PATH, "training_history.csv"),
    index=False
)


# ==============================
# LOAD BEST CHECKPOINT
# ==============================

model.load_state_dict(
    torch.load(
        os.path.join(SAVE_PATH, MODEL_NAME),
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


# ==============================
# VALIDATION REPORT
# ==============================

class_names = [
    "Mild",
    "Moderate",
    "Non Demented",
    "Very Mild"
]

print("\n========== CLAHE VALIDATION RESULTS ==========")

print(
    classification_report(
        all_labels,
        all_predictions,
        labels=[0, 1, 2, 3],
        target_names=class_names,
        zero_division=0
    )
)

cm = confusion_matrix(
    all_labels,
    all_predictions,
    labels=[0, 1, 2, 3]
)

print("\nConfusion Matrix:")
print(cm)

print("\nBest validation Macro F1:", best_f1)

print("\nTraining completed.")

print("Checkpoint:", os.path.join(SAVE_PATH, MODEL_NAME))

print("Training history:", RESULT_PATH)
