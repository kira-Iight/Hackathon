# pipeline.py
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim

from config import BATCH_SIZE, NUM_EPOCHS, DEVICE
from dataset import PlantDataset, get_transforms
from model import get_model
from train_utils import train_model, evaluate_model


def prepare_data(labels_path: str, plant_type: str):
    """
    Prepare data paths, labels, and class mapping.
    """
    df = pd.read_csv(labels_path)
    images_dir = Path(f"data/images_{plant_type}")

    if not images_dir.exists():
        raise FileNotFoundError(f"Directory {images_dir} not found")

    image_paths = []
    labels = []

    unique_types = list(pd.Series(df['type']).astype(str).unique())
    label_map = {name: idx for idx, name in enumerate(unique_types)}

    missing = 0
    for _, row in df.iterrows():
        fname = str(row['filename'])
        cls = str(row['type'])
        candidate = images_dir / fname

        # fallback search if image not found
        if not candidate.exists():
            found = list(images_dir.rglob(fname))
            if found:
                candidate = found[0]
            else:
                missing += 1
                continue

        image_paths.append(str(candidate))
        labels.append(label_map[cls])

    if missing:
        print(f"Missing {missing} images (replaced with placeholders)")

    # Split train/val
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths, labels, test_size=0.2, stratify=labels, random_state=42
    )

    print(f"Data ready: {len(train_paths)} train, {len(val_paths)} val samples, {len(unique_types)} classes")

    return train_paths, val_paths, train_labels, val_labels, unique_types


def train_pipeline(labels_path: str, plant_type: str):
    """
    Full training pipeline for one plant type.
    """
    # --- Data preparation ---
    train_paths, val_paths, train_labels, val_labels, class_names = prepare_data(labels_path, plant_type)
    train_transform, val_transform = get_transforms()

    train_dataset = PlantDataset(train_paths, train_labels, train_transform)
    val_dataset = PlantDataset(val_paths, val_labels, val_transform)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=0, pin_memory=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=0, pin_memory=True
    )

    # --- Model ---
    num_classes = len(class_names)
    model = get_model(num_classes)
    model = model.to(DEVICE)

    # --- Optimizer, loss, scheduler ---
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)

    # --- Training ---
    print(f"\nStarting training for '{plant_type}' with {num_classes} classes.")
    model, history = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler,
        epochs=NUM_EPOCHS, class_names=class_names
    )

    # --- Evaluation ---
    evaluate_model(model, val_loader, class_names)

    # --- Save checkpoint ---
    ckpt_dir = Path("checkpoints")
    ckpt_dir.mkdir(exist_ok=True)
    ckpt_path = ckpt_dir / f"{plant_type}_best.pt"
    torch.save(model.state_dict(), ckpt_path)
    print(f"Model saved to: {ckpt_path}")

    return model, history
