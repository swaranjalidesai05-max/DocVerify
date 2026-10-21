#!/usr/bin/env python3
"""
DocVerify - Phase 6 Dedicated Patch Verification Model Trainer
Author: Senior ML, Computer Vision & Security Research Engineer

Constructs an identity-disjoint local patch dataset from native-resolution document scans:
- Genuine field patches extracted from Real documents (Class 0)
- Manipulated field patches extracted from Fake documents using annotated bounding boxes (Class 1)
Trains a specialized Patch Classifier (EfficientNet-B0) to detect micro-textures,
splicing boundaries, and character inpainting directly at native sensor scale.

Guarantees:
- Strict identity-disjoint splitting (uses dataset_manifest.csv cluster boundaries)
- Zero test contamination
- Output saved to models/phase6/dedicated_patch_model.pth

Outputs:
- models/phase6/dedicated_patch_model.pth
- outputs/phase6/patch_model_metrics.json
"""

import os
import sys
import json
import random
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    MANIFEST_PATH, PHASE6_MODELS_DIR, PHASE6_OUTPUTS,
    IMAGENET_MEAN, IMAGENET_STD, RANDOM_SEED, setup_logger
)

logger = setup_logger("PatchModelTrainer")

# Set random seeds
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)


def extract_contextual_crop(img: Image.Image, box: list, context_scale=2.5, min_dim=64):
    """
    Extracts a contextual patch around bounding box [x, y, w, h] at native resolution.
    Expands the box by context_scale while maintaining bounding within image limits.
    """
    img_w, img_h = img.size
    x, y, w, h = box
    
    # Calculate expanded box
    cx, cy = x + w / 2.0, y + h / 2.0
    nw = max(w * context_scale, min_dim)
    nh = max(h * context_scale, min_dim)
    
    x1 = max(0, int(cx - nw / 2.0))
    y1 = max(0, int(cy - nh / 2.0))
    x2 = min(img_w, int(cx + nw / 2.0))
    y2 = min(img_h, int(cy + nh / 2.0))
    
    crop = img.crop((x1, y1, x2, y2))
    return crop


class PatchDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples  # list of (image_path, box, label)
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, box, label = self.samples[idx]
        with Image.open(img_path) as img:
            img_rgb = img.convert("RGB")
            crop = extract_contextual_crop(img_rgb, box)
            
        if self.transform:
            crop = self.transform(crop)
            
        return crop, torch.tensor(label, dtype=torch.long)


def build_patch_samples():
    """Builds train, val, test patch samples from dataset_manifest.csv."""
    df = pd.read_csv(MANIFEST_PATH)
    logger.info(f"Loaded manifest with {len(df)} records. Building patch dataset...")
    
    samples = {"train": [], "val": [], "test": []}
    
    for idx, row in df.iterrows():
        p = Path(row['absolute_path'])
        if not p.exists():
            p = PROJECT_ROOT / row['image_path']
            
        split = row['split']
        label = int(row['label'])
        
        # Parse bounding box
        box = None
        if pd.notna(row.get('bounding_box_json')):
            try:
                b = json.loads(row['bounding_box_json'])
                if isinstance(b, list) and len(b) >= 4:
                    box = b[:4]
            except Exception:
                pass
                
        if box is None:
            # For real documents without single bounding box in manifest, select realistic field prior
            # Standard ID date/name field location
            with Image.open(p) as img:
                w, h = img.size
            # Use typical text region prior: bottom third or middle
            box = [int(w * 0.35), int(h * 0.45), int(w * 0.25), int(h * 0.08)]
            
        samples[split].append((p, box, label))
        
    for s, lst in samples.items():
        reals = sum(1 for x in lst if x[2] == 0)
        fakes = sum(1 for x in lst if x[2] == 1)
        logger.info(f"Split [{s}]: {len(lst)} patches ({reals} Real, {fakes} Fake)")
        
    return samples


def train_patch_model(epochs=6, batch_size=32, lr=1e-4):
    PHASE6_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    out_model_path = PHASE6_MODELS_DIR / "dedicated_patch_model.pth"
    
    samples = build_patch_samples()
    
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    
    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    
    train_ds = PatchDataset(samples["train"], transform=train_transform)
    val_ds = PatchDataset(samples["val"], transform=eval_transform)
    test_ds = PatchDataset(samples["test"], transform=eval_transform)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    
    logger.info("Initializing dedicated patch classifier backbone (EfficientNet-B0)...")
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_val_f1 = -1.0
    best_state = None
    
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for crops, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(crops)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(targets)
            
        scheduler.step()
        train_loss = running_loss / len(train_ds)
        
        # Validation
        model.eval()
        val_preds, val_targets, val_probs = [], [], []
        with torch.no_grad():
            for crops, targets in val_loader:
                outputs = model(crops)
                probs = torch.softmax(outputs, dim=1)[:, 1].numpy()
                preds = np.argmax(outputs.numpy(), axis=1)
                val_probs.extend(probs)
                val_preds.extend(preds)
                val_targets.extend(targets.numpy())
                
        val_acc = accuracy_score(val_targets, val_preds)
        val_f1 = f1_score(val_targets, val_preds, zero_division=0)
        val_auc = roc_auc_score(val_targets, val_probs)
        
        logger.info(f"Epoch {epoch}/{epochs} | Train Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f} | Val AUC: {val_auc:.4f}")
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = model.state_dict().copy()
            torch.save({
                "epoch": epoch,
                "model_state_dict": best_state,
                "val_f1": best_val_f1,
                "val_acc": val_acc
            }, out_model_path)
            logger.info(f"-> Saved new best patch model checkpoint (Val F1: {best_val_f1:.4f})")
            
    # Final evaluation on untouched frozen test set
    logger.info("Evaluating dedicated patch model on frozen Test Split (N=320)...")
    model.load_state_dict(best_state)
    model.eval()
    
    test_preds, test_targets, test_probs = [], [], []
    with torch.no_grad():
        for crops, targets in test_loader:
            outputs = model(crops)
            probs = torch.softmax(outputs, dim=1)[:, 1].numpy()
            preds = np.argmax(outputs.numpy(), axis=1)
            test_probs.extend(probs)
            test_preds.extend(preds)
            test_targets.extend(targets.numpy())
            
    test_acc = float(accuracy_score(test_targets, test_preds))
    test_prec = float(precision_score(test_targets, test_preds, zero_division=0))
    test_rec = float(recall_score(test_targets, test_preds, zero_division=0))
    test_f1 = float(f1_score(test_targets, test_preds, zero_division=0))
    test_auc = float(roc_auc_score(test_targets, test_probs))
    
    metrics = {
        "model_architecture": "EfficientNet-B0 (Dedicated Patch Classifier)",
        "checkpoint": str(out_model_path),
        "validation_f1": float(best_val_f1),
        "test_metrics": {
            "accuracy": test_acc,
            "precision": test_prec,
            "recall": test_rec,
            "f1_score": test_f1,
            "roc_auc": test_auc
        }
    }
    
    logger.info(f"DEDICATED PATCH MODEL TEST RESULTS: Acc: {test_acc:.4f}, Prec: {test_prec:.4f}, Rec: {test_rec:.4f}, F1: {test_f1:.4f}, AUC: {test_auc:.4f}")
    
    out_metrics_path = PHASE6_OUTPUTS / "patch_model_metrics.json"
    with open(out_metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
        
    return metrics


if __name__ == "__main__":
    train_patch_model(epochs=6, batch_size=32, lr=1e-4)
