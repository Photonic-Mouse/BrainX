
import os
import hashlib
from collections import Counter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

DATASET_PATH = r"C:\HEHAW\BrainX\Dataset"

classes = [
    "Mild_Demented",
    "Moderate_Demented",
    "Non_Demented",
    "Very_Mild_Demented"
]

extensions = (".jpg", ".jpeg", ".png", ".bmp")

image_sizes = Counter()
hashes = {}
corrupted = []
total = 0

fig, axes = plt.subplots(4, 4, figsize=(12, 12))

for row, class_name in enumerate(classes):

    folder = os.path.join(DATASET_PATH, class_name)

    images = [
        f for f in os.listdir(folder)
        if f.lower().endswith(extensions)
    ]

    print(f"\n{class_name}: {len(images)} images")

    for file in images:
        path = os.path.join(folder, file)
        total += 1

        try:
            with Image.open(path) as img:
                img.verify()

            with Image.open(path) as img:
                image_sizes[img.size] += 1

            with open(path, "rb") as f:
                file_hash = hashlib.md5(f.read()).hexdigest()

            hashes.setdefault(file_hash, []).append(path)

        except Exception:
            corrupted.append(path)

    # Display four sample images per class
    for col, file in enumerate(images[:4]):
        path = os.path.join(folder, file)

        with Image.open(path) as img:
            axes[row, col].imshow(img.convert("RGB"))

        axes[row, col].axis("off")

        if col == 0:
            axes[row, col].set_title(class_name)

plt.tight_layout()

output_path = r"C:\HEHAW\BrainX\sample_images.png"

plt.savefig(output_path, dpi=150, bbox_inches="tight")
plt.close()

print(f"\nSample images saved at: {output_path}")

duplicate_groups = [
    paths for paths in hashes.values()
    if len(paths) > 1
]

print("\n========== DATASET REPORT ==========")
print("Total images:", total)
print("Image dimensions:", image_sizes)
print("Corrupted images:", len(corrupted))
print("Exact duplicate groups:", len(duplicate_groups))

print("\nSample image grid saved as sample_images.png")
