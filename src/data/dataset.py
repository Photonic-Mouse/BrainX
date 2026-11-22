
import cv2
import torch
import pandas as pd
import numpy as np

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class BrainMRIDataset(Dataset):

    def __init__(
        self,
        csv_path,
        split="train",
        preprocessing="original"
    ):

        self.data = pd.read_csv(csv_path)

        self.data = self.data[
            self.data["split"] == split
        ].reset_index(drop=True)

        self.preprocessing = preprocessing

        # Augmentation only for training
        if split == "train":

            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomRotation(10),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])

        else:

            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        row = self.data.iloc[index]

        image_path = row["image_path"]
        label = int(row["label"])

        # Load grayscale MRI
        image = cv2.imread(
            image_path,
            cv2.IMREAD_GRAYSCALE
        )

        if image is None:
            raise ValueError(f"Cannot read image: {image_path}")

        # Apply preprocessing
        if self.preprocessing == "HE":

            image = cv2.equalizeHist(image)

        elif self.preprocessing == "CLAHE":

            clahe = cv2.createCLAHE(
                clipLimit=2.0,
                tileGridSize=(8, 8)
            )

            image = clahe.apply(image)

        # Convert grayscale to RGB for pretrained ResNet18
        image = cv2.cvtColor(
            image,
            cv2.COLOR_GRAY2RGB
        )

        image = Image.fromarray(image)

        image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)
