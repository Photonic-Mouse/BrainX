
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights


def create_model(num_classes=4):

    # Load pretrained ResNet18
    model = models.resnet18(
        weights=ResNet18_Weights.DEFAULT
    )

    # Replace final classification layer
    input_features = model.fc.in_features

    model.fc = nn.Linear(
        input_features,
        num_classes
    )

    return model
