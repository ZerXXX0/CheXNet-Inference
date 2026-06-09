#!/usr/bin/env python3
"""Execute CheXNet transfer learning fine-tuning.

Adapted for the 4 target classes discovered dynamically from the test set:
- 'atelektasis'
- 'efusi pleura'
- 'infiltrat'
- 'kavitas'
"""

from __future__ import annotations

import time
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.metrics import classification_report, f1_score

# 1. Configuration
DATASET_DIR = Path("Splitted dataset")
CHECKPOINT_PATH = Path("model.pth.tar")
IMAGE_SIZE = 224
BATCH_SIZE = 8
LR = 1e-4
EPOCHS = 5  # Train for 5 epochs
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {DEVICE}")

# Discover classes from test subdirectories (excluding 'normal')
CLASS_NAMES = sorted([
    d.name for d in (DATASET_DIR / "test").iterdir()
    if d.is_dir() and d.name != "normal"
])
NUM_CLASSES = len(CLASS_NAMES)
print(f"Target classes ({NUM_CLASSES}): {CLASS_NAMES}")

# 2. Custom Multi-Label Dataset
class ChestXRayDataset(Dataset):
    def __init__(self, split_dir: Path, class_names: list, transform=None):
        self.split_dir = split_dir
        self.class_names = class_names
        self.transform = transform
        self.image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
        
        self.class_to_idx = {name: idx for idx, name in enumerate(class_names)}
        self.records = self._build_records()
        self.image_names = list(self.records.keys())
        
    def _build_records(self):
        records = {}
        if not self.split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {self.split_dir}")
            
        for class_dir in self.split_dir.iterdir():
            if not class_dir.is_dir():
                continue
                
            class_name = class_dir.name
            class_idx = self.class_to_idx.get(class_name)
            
            for img_path in class_dir.iterdir():
                if img_path.is_file() and img_path.suffix.lower() in self.image_extensions:
                    img_name = img_path.name
                    if img_name not in records:
                        records[img_name] = {
                            "path": img_path,
                            "labels": np.zeros(len(self.class_names), dtype=np.float32)
                        }
                    if class_idx is not None:
                        records[img_name]["labels"][class_idx] = 1.0
                        
        return records

    def __len__(self):
        return len(self.image_names)
        
    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        record = self.records[img_name]
        
        img = Image.open(record["path"]).convert("RGB")
        label = torch.tensor(record["labels"], dtype=torch.float32)
        
        if self.transform:
            img = self.transform(img)
            
        return img, label

# 3. Dataloaders Setup
train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomRotation(15),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = ChestXRayDataset(DATASET_DIR / "train", CLASS_NAMES, transform=train_transform)
val_dataset = ChestXRayDataset(DATASET_DIR / "valid", CLASS_NAMES, transform=val_transform)
test_dataset = ChestXRayDataset(DATASET_DIR / "test", CLASS_NAMES, transform=val_transform)

print(f"Train samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")
print(f"Test samples: {len(test_dataset)}")

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# 4. Model Loading with Classifier Weight Transfer
DEFAULT_CHEXNET14_CLASS_NAMES = [
    "atelectasis",
    "cardiomegaly",
    "effusion",
    "infiltration",
    "mass",
    "nodule",
    "pneumonia",
    "pneumothorax",
    "consolidation",
    "edema",
    "emphysema",
    "fibrosis",
    "pleural_thickening",
    "hernia",
]

DATASET_NAME_TO_CANONICAL = {
    "kavitas": "cavity",
    "infiltrat": "infiltration",
    "limfadenopati": "lymphadenopathy",
    "tuberkuloma": "tuberculoma",
    "bronkiektasis": "bronchiectasis",
    "pneumothorax": "pneumothorax",
    "efusi pleura": "effusion",
    "atelektasis": "atelectasis",
}

def canonicalize_name(name):
    norm = name.strip().lower().replace("-", " ").replace("_", " ")
    mapped = DATASET_NAME_TO_CANONICAL.get(norm, norm)
    return mapped.replace(" ", "_")

def strip_module_prefix(state_dict):
    has_module_prefix = any(k.startswith("module.") for k in state_dict.keys())
    if not has_module_prefix:
        return state_dict
    return {k.replace("module.", "", 1): v for k, v in state_dict.items()}

def normalize_densenet_keys(state_dict):
    normalized = {}
    replacements = (
        (".norm.1.", ".norm1."),
        (".relu.1.", ".relu1."),
        (".conv.1.", ".conv1."),
        (".norm.2.", ".norm2."),
        (".relu.2.", ".relu2."),
        (".conv.2.", ".conv2."),
    )
    for key, value in state_dict.items():
        new_key = key
        if new_key.startswith("densenet121."):
            new_key = new_key.replace("densenet121.", "", 1)
        for old, new in replacements:
            new_key = new_key.replace(old, new)
        if new_key.startswith("classifier.0."):
            new_key = new_key.replace("classifier.0.", "classifier.", 1)
        normalized[new_key] = value
    return normalized

def load_pretrained_chexnet(checkpoint_path, target_class_names):
    model = models.densenet121(weights=None)
    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, 14)
    
    print(f"Loading pre-trained weights from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    state_dict = strip_module_prefix(state_dict)
    state_dict = normalize_densenet_keys(state_dict)
    
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"Loaded checkpoint successfully. Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
    
    orig_weight = model.classifier.weight.data.clone()
    orig_bias = model.classifier.bias.data.clone()
    
    num_classes = len(target_class_names)
    new_classifier = nn.Linear(in_features, num_classes)
    
    model_class_to_idx = {name: idx for idx, name in enumerate(DEFAULT_CHEXNET14_CLASS_NAMES)}
    transferred_count = 0
    
    for target_idx, target_name in enumerate(target_class_names):
        canonical_target = canonicalize_name(target_name)
        if canonical_target in model_class_to_idx:
            orig_idx = model_class_to_idx[canonical_target]
            new_classifier.weight.data[target_idx] = orig_weight[orig_idx]
            new_classifier.bias.data[target_idx] = orig_bias[orig_idx]
            transferred_count += 1
            print(f"  Transferred pre-trained weights for: '{target_name}' -> '{canonical_target}' (index {orig_idx})")
        else:
            print(f"  Randomly initialized weights for: '{target_name}' (not in pre-trained CheXNet classes)")
            
    print(f"Classifier weight transfer complete: {transferred_count}/{num_classes} classes initialized from pre-trained weights.")
    
    model.classifier = new_classifier
    return model

model = load_pretrained_chexnet(CHECKPOINT_PATH, CLASS_NAMES).to(DEVICE)

# 5. Training Setup
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=2)

history = {
    "train_loss": [],
    "val_loss": [],
    "val_micro_f1": [],
    "val_macro_f1": []
}

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for images, labels in tqdm(loader, desc="Training", leave=False):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
    return running_loss / len(loader.dataset)

@torch.inference_mode()
def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    for images, labels in tqdm(loader, desc="Validation", leave=False):
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)
        
        running_loss += loss.item() * images.size(0)
        
        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).float()
        
        all_preds.append(preds.cpu().numpy())
        all_labels.append(labels.cpu().numpy())
        
    avg_loss = running_loss / len(loader.dataset)
    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    
    micro_f1 = f1_score(all_labels, all_preds, average="micro", zero_division=0)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    
    return avg_loss, micro_f1, macro_f1

# 6. Training Execution
best_val_loss = float("inf")

for epoch in range(1, EPOCHS + 1):
    print(f"\n--- Epoch {epoch}/{EPOCHS} ---")
    
    train_loss = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
    val_loss, val_micro, val_macro = validate(model, val_loader, criterion, DEVICE)
    
    scheduler.step(val_loss)
    
    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["val_micro_f1"].append(val_micro)
    history["val_macro_f1"].append(val_macro)
    
    print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
    print(f"Val Micro-F1: {val_micro:.4f} | Val Macro-F1: {val_macro:.4f}")
    
    # Save the model if it improves
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': val_loss,
        }, 'best_model.pth')
        print("=> Saved best model checkpoint to best_model.pth")

# 7. Generate curves and save plots
print("\nSaving training curves to training_curves.png...")
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history["train_loss"], label="Train Loss", color="royalblue")
plt.plot(history["val_loss"], label="Val Loss", color="orange")
plt.title("Loss Curves")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history["val_micro_f1"], label="Val Micro-F1", color="forestgreen")
plt.plot(history["val_macro_f1"], label="Val Macro-F1", color="purple")
plt.title("Validation F1 Curves")
plt.xlabel("Epoch")
plt.ylabel("F1 Score")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("training_curves.png", dpi=150)
plt.close()

# 8. Evaluation
if Path("best_model.pth").exists():
    print("\nLoading best model weights for final evaluation...")
    checkpoint = torch.load("best_model.pth", map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])

model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in tqdm(test_loader, desc="Testing"):
        images = images.to(DEVICE)
        logits = model(images)
        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).float()
        
        all_preds.append(preds.cpu().numpy())
        all_labels.append(labels.numpy())

all_preds = np.vstack(all_preds)
all_labels = np.vstack(all_labels)

print("\n=== Final Test Classification Report ===")
print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES, zero_division=0))
print("Fine-tuning completed successfully!")
