# train_utils.py
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from config import DEVICE


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler=None, epochs=10, class_names=None):
    """
    Full training loop with validation and learning rate tracking.
    """
    model = model.to(DEVICE)

    best_acc = 0.0
    best_state = None
    history = {'train_loss': [], 'val_loss': [], 'val_acc': [], 'lr': []}

    for epoch in range(1, epochs + 1):
        print(f"\n===== Epoch {epoch}/{epochs} =====")
        
        # ---- TRAIN PHASE ----
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in tqdm(train_loader, desc=f"Train {epoch}", leave=False):
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / len(train_loader.dataset)
        train_acc = correct / total

        # ---- VALIDATION PHASE ----
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Val {epoch}", leave=False):
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_loss /= len(val_loader.dataset)
        val_acc = val_correct / val_total

        # Scheduler step
        if scheduler:
            scheduler.step(val_loss)

        # Save history
        lr = optimizer.param_groups[0]['lr']
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['lr'].append(lr)

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.3f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.3f} | LR: {lr:.6f}")

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = model.state_dict()

    print(f"\nBest Validation Accuracy: {best_acc:.3f}")
    model.load_state_dict(best_state)
    return model, history


def evaluate_model(model, val_loader, class_names):
    """
    Evaluate model on validation set and print full classification report.
    """
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Evaluation"):
            images = images.to(DEVICE)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    print("\n" + "=" * 60)
    print(" CLASSIFICATION REPORT ")
    print("=" * 60)
    print(classification_report(all_labels, all_preds, target_names=class_names, digits=3))

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

    return all_preds, all_labels, cm
