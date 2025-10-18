# config.py
import torch
import random
import numpy as np
import os

# === GLOBAL SETTINGS ===
IMG_SIZE = (224, 224)
BATCH_SIZE = 256
NUM_EPOCHS = 100
SEED = 42

# === DEVICE CONFIGURATION ===
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# === REPRODUCIBILITY ===
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print(f"Device: {DEVICE}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Epochs: {NUM_EPOCHS}")
