# model.py
import torch.nn as nn
from torchvision import models

def get_model(num_classes: int, freeze_features: bool = True):
    """
    Build an EfficientNet-B2 model for classification.
    
    Args:
        num_classes (int): Number of output classes.
        freeze_features (bool): Whether to freeze feature extractor layers.
        
    Returns:
        nn.Module: Ready-to-train model.
    """
    # Updated API for torchvision 0.15+
    weights = models.EfficientNet_B2_Weights.IMAGENET1K_V1
    model = models.efficientnet_b2(weights=weights)

    # Optionally freeze feature extractor
    if freeze_features:
        for param in model.features.parameters():
            param.requires_grad = False

    # Replace classifier head
    num_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(num_features, num_classes)
    )

    return model
