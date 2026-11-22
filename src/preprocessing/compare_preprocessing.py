
import cv2
import os
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATASET_PATH = r"C:\HEHAW\BrainX\Dataset"
OUTPUT_PATH = r"C:\HEHAW\BrainX\results\preprocessing"

classes = [
    "Mild_Demented",
    "Moderate_Demented",
    "Non_Demented",
    "Very_Mild_Demented"
]

os.makedirs(OUTPUT_PATH, exist_ok=True)

fig, axes = plt.subplots(4, 3, figsize=(10, 13))

for row, class_name in enumerate(classes):

    folder = os.path.join(DATASET_PATH, class_name)

    files = [
        f for f in os.listdir(folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    image_path = os.path.join(folder, files[0])

    # Read MRI image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Histogram Equalization
    he_image = cv2.equalizeHist(image)

    # CLAHE
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    clahe_image = clahe.apply(image)

    # Display results
    axes[row, 0].imshow(image, cmap="gray")
    axes[row, 1].imshow(he_image, cmap="gray")
    axes[row, 2].imshow(clahe_image, cmap="gray")

    axes[row, 0].set_ylabel(class_name)

    for col in range(3):
        axes[row, col].axis("off")

axes[0, 0].set_title("Original")
axes[0, 1].set_title("Histogram Equalization")
axes[0, 2].set_title("CLAHE")

plt.tight_layout()

output_file = os.path.join(
    OUTPUT_PATH, "preprocessing_comparison.png"
)

plt.savefig(output_file, dpi=150)
plt.close()

print("Preprocessing comparison completed!")
print("Saved at:", output_file)
