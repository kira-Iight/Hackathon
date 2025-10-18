# dataset.py
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from pathlib import Path
import cv2
import torch
import random

from config import IMG_SIZE


class PlantDataset(Dataset):
    """Custom dataset for plant species classification."""

    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = Path(self.image_paths[idx])
        label = self.labels[idx]

        # Try reading the image robustly
        try:
            image = cv2.imread(str(img_path))
            if image is None:
                raise FileNotFoundError(f"Image not found: {img_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
        except Exception:
            # fallback gray image if missing/corrupted
            image = Image.new("RGB", IMG_SIZE, color=(128, 128, 128))

        if self.transform:
            image = self.transform(image)

        return image, label


def get_transforms():
    """Return training and validation torchvision transforms."""
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.1),
        transforms.GaussianBlur(3, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    return train_transform, val_transform
