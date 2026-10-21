#!/usr/bin/env python3
"""
DocVerify - Phase 8 Remediated Dedicated Patch Classifier
Author: Senior ML, Computer Vision & Security Research Engineer

Remediation Focus:
1. Resolves AUD-PATCH-01: Uses src/bbox_parser.py to correctly parse SIDTD dictionary bounding boxes.
2. Resolves AUD-PATCH-02: Supersedes invalidated 99.38% claim with true sensor-scale crop evaluation.
3. Resolves AUD-PATCH-03: Uses copy.deepcopy for state dict checkpointing, preventing in-place parameter mutation.
4. Strict train/val/test segregation: Train strictly on Train split (N=1,565), select best epoch on Val split (N=337), evaluate on Frozen Test (N=320).
"""

import sys
import copy
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.bbox_parser import parse_sidtd_bbox
from src.config import (
    MANIFEST_PATH, IMAGENET_MEAN, IMAGENET_STD, RANDOM_SEED, setup_logger
)

logger = setup_logger("Phase8PatchTrainer")

# Enforce deterministic random seeds
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

MODELS_PHASE8_DIR = PROJECT_ROOT / "models" / "phase8"
OUTPUTS_PHASE8_DIR = PROJECT_ROOT / "outputs" / "phase8" / "patch_model"
MODELS_PHASE8_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_PHASE8_DIR.mkdir(parents=True, exist_ok=True)


def extract_contextual_crop(img: Image.Image, box: list, context_scale: float = 2.0, min_dim: int = 64) -> Image.Image:
    """
    Extracts a contextual patch around bounding box [x, y, w, h] at native sensor resolution.
    Expands the box by context_scale while strictly clipping within image bounds.
    """
    img_w, img_h = img.size
    x, y, w, h = box

    cx, cy = x + w / 2.0, y + h / 2.0
    nw = max(w * context_scale, min_dim)
    nh = max(h * context_scale, min_dim)

    x1 = max(0, int(cx - nw / 2.0))
    y1 = max(0, int(cy - nh / 2.0))
    x2 = min(img_w, int(cx + nw / 2.0))
    y2 = min(img_h, int(cy + nh / 2.0))

    return img.crop((x1, y1, x2, y2))


class Phase8PatchDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples  # (img_path, box, label, template, forgery_type, img_filename)
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, box, label, tpl, ftype, fname = self.samples[idx]
        with Image.open(img_path) as img:
            img_rgb = img.convert("RGB")
            crop = extract_contextual_crop(img_rgb, box, context_scale=2.0)

        if self.transform:
            crop = self.transform(crop)

        return crop, torch.tensor(label, dtype=torch.long), idx


def build_remediated_samples():
    """Builds train, val, and test patch samples using parse_sidtd_bbox."""
    df = pd.read_csv(MANIFEST_PATH)
    logger.info(f"Loaded manifest with {len(df)} records. Parsing remediated bounding boxes...")

    samples = {"train": [], "val": [], "test": []}
    sample_metadata = []

    for idx, row in df.iterrows():
        p = Path(row["absolute_path"])
        if not p.exists():
            p = PROJECT_ROOT / row["image_path"]

        split = row["split"]
        label = int(row["label"])
        tpl = str(row["template"])
        fname = str(row["image_filename"])
        ftype = str(row.get("forgery_type", "None"))
        img_w = int(row["width"])
        img_h = int(row["height"])
        raw_box = row["bounding_box_json"]

        box, status = parse_sidtd_bbox(
            bbox_raw=raw_box,
            img_width=img_w,
            img_height=img_h,
            template_id=tpl,
            sample_id=fname
        )

        item = (p, box, label, tpl, ftype, fname)
        samples[split].append(item)
        sample_metadata.append({
            "image_filename": fname,
            "split": split,
            "label": label,
            "template": tpl,
            "forgery_type": ftype,
            "box": box,
            "bbox_status": status
        })

    for s, lst in samples.items():
        reals = sum(1 for x in lst if x[2] == 0)
        fakes = sum(1 for x in lst if x[2] == 1)
        logger.info(f"Split [{s}]: {len(lst)} patches ({reals} Real, {fakes} Fake)")

    return samples, sample_metadata


def train_and_evaluate_patch_model(epochs: int = 6, batch_size: int = 32, lr: float = 1e-4):
    out_model_path = MODELS_PHASE8_DIR / "dedicated_patch_model.pth"

    samples, sample_metadata = build_remediated_samples()

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

    train_ds = Phase8PatchDataset(samples["train"], transform=train_transform)
    val_ds = Phase8PatchDataset(samples["val"], transform=eval_transform)
    test_ds = Phase8PatchDataset(samples["test"], transform=eval_transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    logger.info("Initializing EfficientNet-B0 backbone with ImageNet-1K pretrained weights...")
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    model.to(device)

    # Class weighting to account for slight imbalance (703 Real, 862 Fake in train)
    n_real = sum(1 for x in samples["train"] if x[2] == 0)
    n_fake = sum(1 for x in samples["train"] if x[2] == 1)
    weights = torch.tensor([n_fake / n_real, 1.0], device=device, dtype=torch.float32)
    criterion = nn.CrossEntropyLoss(weight=weights)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_f1 = -1.0
    best_epoch = -1
    best_state = None
    epoch_history = []

    logger.info("Commencing Phase 8 Dedicated Patch Classifier Training...")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        train_preds, train_targets = [], []

        for crops, targets, _ in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Train]"):
            crops, targets = crops.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(crops)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(targets)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            train_preds.extend(preds)
            train_targets.extend(targets.cpu().numpy())

        scheduler.step()
        train_loss = total_loss / len(train_ds)
        train_acc = accuracy_score(train_targets, train_preds)

        # Validation evaluation
        model.eval()
        val_preds, val_probs, val_targets = [], [], []
        with torch.no_grad():
            for crops, targets, _ in val_loader:
                crops, targets = crops.to(device), targets.to(device)
                outputs = model(crops)
                probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
                preds = torch.argmax(outputs, dim=1).cpu().numpy()

                val_probs.extend(probs)
                val_preds.extend(preds)
                val_targets.extend(targets.cpu().numpy())

        val_acc = accuracy_score(val_targets, val_preds)
        val_f1 = f1_score(val_targets, val_preds, zero_division=0)
        val_auc = roc_auc_score(val_targets, val_probs)

        logger.info(
            f"Epoch {epoch}/{epochs} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f} | Val AUC: {val_auc:.4f}"
        )

        epoch_history.append({
            "epoch": epoch,
            "train_loss": float(train_loss),
            "train_acc": float(train_acc),
            "val_acc": float(val_acc),
            "val_f1": float(val_f1),
            "val_auc": float(val_auc)
        })

        # AUD-PATCH-03 Remediation: Use true deepcopy of state dict
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            logger.info(f"[*] New best validation F1: {val_f1:.4f} at Epoch {epoch}. State dict deepcopied.")

    # Save best checkpoint
    logger.info(f"Restoring best checkpoint from Epoch {best_epoch} (Val F1: {best_val_f1:.4f})...")
    model.load_state_dict(best_state)

    checkpoint = {
        "model_architecture": "EfficientNet-B0 (Dedicated Patch Classifier)",
        "phase": "Phase 8 Remediated",
        "best_epoch": best_epoch,
        "best_val_f1": float(best_val_f1),
        "random_seed": RANDOM_SEED,
        "model_state_dict": best_state
    }
    torch.save(checkpoint, out_model_path)
    logger.info(f"Saved remediated patch model to: {out_model_path}")

    # =========================================================================
    # Step 5: Independent Evaluation on Frozen Test Set
    # =========================================================================
    logger.info("Evaluating Remediated Patch Classifier on FROZEN Test Set (N=320)...")
    model.eval()
    test_probs, test_preds, test_targets, test_indices = [], [], [], []
    with torch.no_grad():
        for crops, targets, idxs in test_loader:
            crops, targets = crops.to(device), targets.to(device)
            outputs = model(crops)
            probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            preds = torch.argmax(outputs, dim=1).cpu().numpy()

            test_probs.extend(probs)
            test_preds.extend(preds)
            test_targets.extend(targets.cpu().numpy())
            test_indices.extend(idxs.numpy())

    test_acc = accuracy_score(test_targets, test_preds)
    test_prec = precision_score(test_targets, test_preds, zero_division=0)
    test_rec = recall_score(test_targets, test_preds, zero_division=0)
    test_f1 = f1_score(test_targets, test_preds, zero_division=0)
    test_auc = roc_auc_score(test_targets, test_probs)
    cm = confusion_matrix(test_targets, test_preds).tolist()

    logger.info("=" * 60)
    logger.info(f"REMEDIATED PATCH MODEL TEST RESULTS:")
    logger.info(f"Accuracy:  {test_acc * 100:.2f}%")
    logger.info(f"Precision: {test_prec * 100:.2f}%")
    logger.info(f"Recall:    {test_rec * 100:.2f}%")
    logger.info(f"F1 Score:  {test_f1 * 100:.2f}%")
    logger.info(f"ROC-AUC:   {test_auc:.4f}")
    logger.info(f"Confusion Matrix: TN={cm[0][0]}, FP={cm[0][1]}, FN={cm[1][0]}, TP={cm[1][1]}")
    logger.info("=" * 60)

    # Per-template and per-forgery-type evaluation
    test_samples_list = samples["test"]
    test_records = []
    for i, idx_val in enumerate(test_indices):
        p, box, lbl, tpl, ftype, fname = test_samples_list[idx_val]
        test_records.append({
            "image_filename": fname,
            "template": tpl,
            "forgery_type": ftype,
            "true_label": int(lbl),
            "pred_label": int(test_preds[i]),
            "patch_prob_fake": float(test_probs[i])
        })
    df_test_res = pd.DataFrame(test_records)
    df_test_res.to_csv(OUTPUTS_PHASE8_DIR / "patch_test_predictions.csv", index=False)

    # Per-template metrics
    per_template = {}
    for tpl_name, grp in df_test_res.groupby("template"):
        acc_tpl = accuracy_score(grp["true_label"], grp["pred_label"])
        per_template[tpl_name] = {
            "samples": len(grp),
            "accuracy": float(acc_tpl)
        }

    # Per-forgery-type metrics
    per_forgery = {}
    for f_name, grp in df_test_res[df_test_res["true_label"] == 1].groupby("forgery_type"):
        rec_f = recall_score(grp["true_label"], grp["pred_label"], zero_division=0)
        per_forgery[f_name] = {
            "samples": len(grp),
            "detection_rate": float(rec_f)
        }

    # Generate complete metrics JSON
    metrics_payload = {
        "model": "Phase 8 Remediated Dedicated Patch Classifier",
        "checkpoint": str(out_model_path),
        "backbone": "EfficientNet-B0",
        "training_provenance": {
            "train_samples": len(train_ds),
            "val_samples": len(val_ds),
            "test_samples": len(test_ds),
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": lr,
            "best_epoch": best_epoch,
            "random_seed": RANDOM_SEED
        },
        "remediation_status": {
            "previous_claim_status": "99.38% test accuracy from EXP-06 is INVALIDATED / SUPERSEDED DUE TO PATCH-BOX PARSING ERROR",
            "parsing_remediation": "All 1,222 SIDTD forged bounding boxes parsed directly via src/bbox_parser.py",
            "checkpoint_remediation": "copy.deepcopy applied to prevent parameter mutation"
        },
        "epoch_history": epoch_history,
        "test_metrics": {
            "accuracy": float(test_acc),
            "precision": float(test_prec),
            "recall": float(test_rec),
            "f1_score": float(test_f1),
            "roc_auc": float(test_auc),
            "confusion_matrix": {
                "true_negative": cm[0][0],
                "false_positive": cm[0][1],
                "false_negative": cm[1][0],
                "true_positive": cm[1][1]
            }
        },
        "per_template_accuracy": per_template,
        "per_forgery_detection_rate": per_forgery
    }

    with open(OUTPUTS_PHASE8_DIR / "patch_model_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)
    logger.info(f"Saved patch model metrics to: {OUTPUTS_PHASE8_DIR / 'patch_model_metrics.json'}")

    return metrics_payload


if __name__ == "__main__":
    train_and_evaluate_patch_model(epochs=6, batch_size=32, lr=1e-4)
