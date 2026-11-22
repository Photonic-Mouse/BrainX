
import os
import torch
from pathlib import Path

from torch.utils.data import DataLoader
from src.data.dataset import BrainMRIDataset

REPO_ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = REPO_ROOT / "Dataset" / "splits" / "splits.csv"

train_dataset = BrainMRIDataset(
    CSV_PATH,
    split="train",
    preprocessing="original"
)

train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,
    num_workers=0
)

images, labels = next(iter(train_loader))

print("Image batch shape:", images.shape)
print("Label batch shape:", labels.shape)
print("Labels:", labels)
print("Image tensor device:", images.device)

# Verify CUDA
if torch.cuda.is_available():
    images = images.to("cuda")
    print("GPU test successful!")
    print("GPU image batch:", images.shape)
