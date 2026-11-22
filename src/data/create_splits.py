
import os
import pandas as pd

from pathlib import Path
from sklearn.model_selection import train_test_split

# Dataset location
REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = REPO_ROOT / "Dataset"

# Output location
OUTPUT_PATH = DATASET_PATH / "splits"
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

# Class labels
classes = [
    "Mild_Demented",
    "Moderate_Demented",
    "Non_Demented",
    "Very_Mild_Demented"
]

# Supported image formats
extensions = {".jpg", ".jpeg", ".png", ".bmp"}

data = []

# Collect image paths and labels
for label, class_name in enumerate(classes):

    folder = DATASET_PATH / class_name

    if not folder.exists():
        raise FileNotFoundError(f"Missing folder: {folder}")

    images = [
        file for file in folder.iterdir()
        if file.is_file() and file.suffix.lower() in extensions
    ]

    for image in images:
        data.append({
            "image_path": str(image.resolve()),
            "class_name": class_name,
            "label": label
        })

df = pd.DataFrame(data)

if df.empty:
    raise ValueError("No images found!")

print("Total images:", len(df))

# First split: 70% training, 30% temporary
train_df, temp_df = train_test_split(
    df,
    test_size=0.30,
    stratify=df["label"],
    random_state=42
)

# Second split: 15% validation, 15% testing
val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    stratify=temp_df["label"],
    random_state=42
)

# Assign split names
train_df = train_df.copy()
val_df = val_df.copy()
test_df = test_df.copy()

train_df["split"] = "train"
val_df["split"] = "validation"
test_df["split"] = "test"

# Combine all splits
final_df = pd.concat(
    [train_df, val_df, test_df],
    ignore_index=True
)

# Save CSV
csv_path = OUTPUT_PATH / "splits.csv"
final_df.to_csv(csv_path, index=False)

# Print split summary
print("\n========== SPLIT SUMMARY ==========")

for split in ["train", "validation", "test"]:

    subset = final_df[final_df["split"] == split]

    print(f"\n{split.upper()}: {len(subset)} images")

    print(
        subset["class_name"]
        .value_counts()
        .reindex(classes, fill_value=0)
    )

print("\nSplits saved at:", csv_path)
