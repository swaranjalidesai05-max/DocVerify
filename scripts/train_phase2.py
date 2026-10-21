#!/usr/bin/env python3
"""
DocVerify - Phase 2 AI Forgery Classification Baseline (EfficientNet-B0)
Author: Senior ML, Computer Vision & Research Engineer

Pipeline:
1. Loads dataset_manifest.csv (with strictly verified identity-disjoint splits).
2. Custom PyTorch Dataset ensuring:
   - Safe Pillow in-memory RGB conversion: Image.open(path).convert('RGB')
   - Preprocessing: Resize to 256, CenterCrop to 224, ImageNet normalization.
   - Augmentation (Train only): Mild random horizontal flip, slight rotation (+/-5 deg),
     mild ColorJitter (brightness=0.1, contrast=0.1). Kept conservative to preserve forgery cues.
3. Model Architecture:
   - EfficientNet-B0 backbone (ImageNet-pretrained).
   - Replaced classifier head: Linear(1280, 2) binary output.
   - Fine-tunes the network using cross-entropy with class-weighting derived from TRAIN split only.
4. Checkpointing & Optimization:
   - AdamW optimizer, CosineAnnealingLR scheduler.
   - Checkpoints best validation model (val_loss / val_f1) to models/phase2/best_model.pth.
5. Final Evaluation (ONCE on untouched TEST split):
   - Computes: Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix, FP, FN.
   - Per-class analysis (Real vs Fake).
   - Forgery-type performance breakdown (Inpaint_and_Rewrite vs Crop_and_Replace).
   - Document-type performance breakdown.
   - Error analysis on representative FP and FN samples with image artifacts.
6. Generates full visualization figures and machine-readable phase2_results.json.
"""

import json
import time
import random
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torchvision.transforms as transforms
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve
)
from tqdm import tqdm


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class SIDTDDataset(Dataset):
    """Custom Dataset with safe RGB conversion and clean metadata tracking."""
    def __init__(self, df: pd.DataFrame, base_dir: Path, transform=None):
        self.df = df.reset_index(drop=True)
        self.base_dir = base_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # Resolve path safely
        img_p = Path(row["absolute_path"])
        if not img_p.exists():
            img_p = self.base_dir / row["image_path"]

        # SAFE PIL RGB LOADING (handles 500 CMYK JPEGs in memory)
        with Image.open(img_p) as img:
            img_rgb = img.convert("RGB")

        if self.transform:
            tensor = self.transform(img_rgb)
        else:
            tensor = transforms.ToTensor()(img_rgb)

        label = int(row["label"])
        return tensor, label, idx


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_preds, all_labels = [], []
    
    for inputs, labels, _ in tqdm(loader, desc="Training Batch", leave=False):
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        preds = torch.argmax(outputs, dim=1).detach().cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.detach().cpu().numpy())
        
    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    return epoch_loss, epoch_acc


def evaluate_model(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds, all_labels, all_probs, all_indices = [], [], [], []
    
    with torch.no_grad():
        for inputs, labels, indices in tqdm(loader, desc="Evaluating", leave=False):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            
            probs = torch.softmax(outputs, dim=1)[:, 1].detach().cpu().numpy()
            preds = torch.argmax(outputs, dim=1).detach().cpu().numpy()
            
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels.detach().cpu().numpy())
            all_indices.extend(indices.numpy())
            
    eval_loss = running_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, zero_division=0)
    rec = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except Exception:
        auc = 0.5
        
    return {
        "loss": eval_loss,
        "acc": acc,
        "prec": prec,
        "rec": rec,
        "f1": f1,
        "auc": auc,
        "preds": np.array(all_preds),
        "probs": np.array(all_probs),
        "labels": np.array(all_labels),
        "indices": np.array(all_indices)
    }


def run_phase2():
    SEED = 42
    set_seed(SEED)
    
    BASE_DIR = Path("E:/DocverifyAi")
    MANIFEST_PATH = BASE_DIR / "outputs" / "dataset_manifest" / "dataset_manifest.csv"
    OUTPUT_PHASE2 = BASE_DIR / "outputs" / "phase2"
    MODEL_DIR = BASE_DIR / "models" / "phase2"
    VIZ_DIR = OUTPUT_PHASE2 / "visualizations"
    ERR_DIR = OUTPUT_PHASE2 / "error_analysis"
    
    for d in [OUTPUT_PHASE2, MODEL_DIR, VIZ_DIR, ERR_DIR]:
        d.mkdir(parents=True, exist_ok=True)
        
    print(f"Loading verified manifest: {MANIFEST_PATH}")
    df = pd.read_csv(MANIFEST_PATH)
    
    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "val"].copy()
    test_df = df[df["split"] == "test"].copy()
    
    print(f"Dataset Loaded: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    
    # Preprocessing & conservative document augmentations
    # Mean and std from ImageNet
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]
    
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=4),
        transforms.ColorJitter(brightness=0.08, contrast=0.08),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    
    eval_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    
    train_dataset = SIDTDDataset(train_df, BASE_DIR, transform=train_transform)
    val_dataset = SIDTDDataset(val_df, BASE_DIR, transform=eval_transform)
    test_dataset = SIDTDDataset(test_df, BASE_DIR, transform=eval_transform)
    
    BATCH_SIZE = 32
    NUM_WORKERS = 0  # 0 on Windows avoids spawn/IPC issues
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    
    # Calculate class weights strictly from the TRAIN split
    train_reals = (train_df["label"] == 0).sum()
    train_fakes = (train_df["label"] == 1).sum()
    total_train = len(train_df)
    weight_real = total_train / (2.0 * train_reals)
    weight_fake = total_train / (2.0 * train_fakes)
    class_weights = torch.tensor([weight_real, weight_fake], dtype=torch.float)
    print(f"Train class weights: Real (0)={weight_real:.3f}, Fake (1)={weight_fake:.3f}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing on device: {device}")
    
    # Model definition: EfficientNet-B0 baseline
    weights = models.EfficientNet_B0_Weights.DEFAULT
    model = models.efficientnet_b0(weights=weights)
    
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, 2)
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    EPOCHS = 10
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    # Record experiment config
    config = {
        "model": "EfficientNet-B0",
        "input_size": 224,
        "resize_size": 256,
        "pretrained_weights": "EfficientNet_B0_Weights.DEFAULT",
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "learning_rate": 1e-4,
        "weight_decay": 1e-4,
        "optimizer": "AdamW",
        "scheduler": "CosineAnnealingLR",
        "loss": "CrossEntropyLoss with Train-derived Class Weights",
        "class_weights": {"real": round(weight_real, 4), "fake": round(weight_fake, 4)},
        "seed": SEED,
        "device": str(device)
    }
    with open(OUTPUT_PHASE2 / "experiment_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
        
    print(f"Model parameters: {config['trainable_parameters']:,} trainable of {config['total_parameters']:,} total.")
    print("Beginning Training Loop...")
    
    history = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [], "val_f1": []
    }
    
    best_val_f1 = -1.0
    best_val_loss = float("inf")
    best_epoch = -1
    
    start_time = time.time()
    
    for epoch in range(1, EPOCHS + 1):
        ep_start = time.time()
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_res = evaluate_model(model, val_loader, criterion, device)
        scheduler.step()
        ep_duration = time.time() - ep_start
        
        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(val_res["loss"])
        history["val_acc"].append(val_res["acc"])
        history["val_f1"].append(val_res["f1"])
        
        print(f"Epoch [{epoch:02d}/{EPOCHS:02d}] ({ep_duration:.1f}s) - "
              f"Train Loss: {tr_loss:.4f}, Acc: {tr_acc:.4f} | "
              f"Val Loss: {val_res['loss']:.4f}, Acc: {val_res['acc']:.4f}, F1: {val_res['f1']:.4f}, AUC: {val_res['auc']:.4f}")
        
        # Checkpoint best validation model based on Validation F1
        if val_res["f1"] > best_val_f1:
            best_val_f1 = val_res["f1"]
            best_val_loss = val_res["loss"]
            best_epoch = epoch
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_metrics": {k: float(v) for k, v in val_res.items() if isinstance(v, (int, float, np.floating))},
                "config": config
            }, MODEL_DIR / "best_model.pth")
            print(f"  -> Saved new best model checkpoint (Val F1: {val_res['f1']:.4f})")
            
    total_train_time = time.time() - start_time
    print(f"Training completed in {total_train_time/60:.2f} minutes. Best Epoch: {best_epoch} with Val F1: {best_val_f1:.4f}")
    
    # Save final model state
    torch.save(model.state_dict(), MODEL_DIR / "final_model.pth")
    
    # ----------------------------------------------------
    # FINAL EVALUATION ON UNTOUCHED TEST SPLIT
    # ----------------------------------------------------
    print("\n" + "="*50)
    print("FINAL EVALUATION ON UNTOUCHED TEST SPLIT")
    print("="*50)
    
    # Load best checkpoint
    checkpoint = torch.load(MODEL_DIR / "best_model.pth", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    test_res = evaluate_model(model, test_loader, criterion, device)
    
    cm = confusion_matrix(test_res["labels"], test_res["preds"])
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    
    print(f"Test Accuracy:  {test_res['acc']:.4f}")
    print(f"Test Precision: {test_res['prec']:.4f}")
    print(f"Test Recall:    {test_res['rec']:.4f}")
    print(f"Test F1-Score:  {test_res['f1']:.4f}")
    print(f"Test ROC-AUC:   {test_res['auc']:.4f}")
    print(f"Confusion Matrix: [TN={tn}, FP={fp}, FN={fn}, TP={tp}]")
    print(f"False Positive Rate: {fpr:.4f} | False Negative Rate: {fnr:.4f}")
    
    # Per-Class Analysis
    test_df_eval = test_df.iloc[test_res["indices"]].copy()
    test_df_eval["pred"] = test_res["preds"]
    test_df_eval["prob_fake"] = test_res["probs"]
    test_df_eval["prob_real"] = 1.0 - test_res["probs"]
    test_df_eval["correct"] = (test_df_eval["pred"] == test_df_eval["label"])
    
    real_mask = (test_df_eval["label"] == 0)
    fake_mask = (test_df_eval["label"] == 1)
    
    real_acc = test_df_eval[real_mask]["correct"].mean()
    fake_acc = test_df_eval[fake_mask]["correct"].mean()
    
    # Forgery-type performance breakdown
    forgery_metrics = {}
    for ftype, group in test_df_eval[fake_mask].groupby("forgery_type"):
        f_acc = accuracy_score(group["label"], group["pred"])
        f_rec = recall_score(group["label"], group["pred"], zero_division=0)
        forgery_metrics[ftype] = {
            "samples": len(group),
            "accuracy": round(float(f_acc), 4),
            "recall": round(float(f_rec), 4)
        }
        print(f"Forgery Type [{ftype}]: Count={len(group)}, Recall={f_rec:.4f}, Accuracy={f_acc:.4f}")
        
    # Document-type performance breakdown
    doc_metrics = {}
    for template_name, group in test_df_eval.groupby("template"):
        t_acc = accuracy_score(group["label"], group["pred"])
        doc_metrics[template_name] = {
            "samples": len(group),
            "accuracy": round(float(t_acc), 4)
        }
        
    # Error Analysis & Visual Artifacts
    fps = test_df_eval[(test_df_eval["label"] == 0) & (test_df_eval["pred"] == 1)]
    fns = test_df_eval[(test_df_eval["label"] == 1) & (test_df_eval["pred"] == 0)]
    tps = test_df_eval[(test_df_eval["label"] == 1) & (test_df_eval["pred"] == 1)]
    tns = test_df_eval[(test_df_eval["label"] == 0) & (test_df_eval["pred"] == 0)]
    
    print(f"\nError Analysis Summary: Total FP={len(fps)}, Total FN={len(fns)}")
    
    # Save representative error images
    for idx, row in fps.head(4).iterrows():
        try:
            with Image.open(row["absolute_path"]) as im:
                im_rgb = im.convert("RGB")
                im_rgb.save(ERR_DIR / f"FP_{row['image_filename']}")
        except Exception as e:
            print(f"Could not save FP visual: {e}")
            
    for idx, row in fns.head(4).iterrows():
        try:
            with Image.open(row["absolute_path"]) as im:
                im_rgb = im.convert("RGB")
                im_rgb.save(ERR_DIR / f"FN_{row['image_filename']}")
        except Exception as e:
            print(f"Could not save FN visual: {e}")
            
    # ----------------------------------------------------
    # GENERATE VISUALIZATIONS
    # ----------------------------------------------------
    sns.set_theme(style="whitegrid")
    
    # 1. Training & Validation Curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    epochs_range = range(1, EPOCHS + 1)
    
    ax1.plot(epochs_range, history["train_loss"], "o-", label="Train Loss", color="#2980b9", lw=2)
    ax1.plot(epochs_range, history["val_loss"], "s-", label="Val Loss", color="#e74c3c", lw=2)
    ax1.set_title("Cross-Entropy Loss (Train vs Val)", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Loss", fontsize=11)
    ax1.legend(fontsize=11)
    
    ax2.plot(epochs_range, history["train_acc"], "o-", label="Train Accuracy", color="#27ae60", lw=2)
    ax2.plot(epochs_range, history["val_acc"], "s-", label="Val Accuracy", color="#8e44ad", lw=2)
    ax2.plot(epochs_range, history["val_f1"], "^--", label="Val F1-Score", color="#f39c12", lw=2)
    ax2.set_title("Accuracy & F1 (Train vs Val)", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("Metric Score", fontsize=11)
    ax2.legend(fontsize=11)
    
    plt.tight_layout()
    plt.savefig(VIZ_DIR / "training_curves.png", dpi=200)
    plt.close()
    
    # 2. Confusion Matrix
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Pred Real (0)", "Pred Fake (1)"],
                yticklabels=["Actual Real (0)", "Actual Fake (1)"],
                annot_kws={"size": 14, "weight": "bold"})
    plt.title(f"Test Confusion Matrix (Acc: {test_res['acc']:.4f})", fontsize=13, fontweight="bold")
    plt.ylabel("Ground Truth", fontsize=12)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.tight_layout()
    plt.savefig(VIZ_DIR / "confusion_matrix.png", dpi=200)
    plt.close()
    
    # 3. ROC Curve
    fpr_curve, tpr_curve, _ = roc_curve(test_res["labels"], test_res["probs"])
    plt.figure(figsize=(7, 6))
    plt.plot(fpr_curve, tpr_curve, color="#e67e22", lw=2.5, label=f"EfficientNet-B0 (AUC = {test_res['auc']:.4f})")
    plt.plot([0, 1], [0, 1], color="gray", lw=1.5, linestyle="--")
    plt.title("Receiver Operating Characteristic (ROC) Curve", fontsize=13, fontweight="bold")
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    plt.ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=11)
    plt.legend(loc="lower right", fontsize=11)
    plt.tight_layout()
    plt.savefig(VIZ_DIR / "roc_curve.png", dpi=200)
    plt.close()
    
    # 4. Save Machine-Readable Results JSON
    results_json = {
        "experiment_name": "Phase2_EfficientNet_B0_Baseline",
        "dataset_summary": {
            "total_samples": 2222,
            "train_samples": len(train_df),
            "val_samples": len(val_df),
            "test_samples": len(test_df),
            "train_class_distribution": {"real": int(train_reals), "fake": int(train_fakes)},
            "val_class_distribution": {
                "real": int((val_df["label"] == 0).sum()),
                "fake": int((val_df["label"] == 1).sum())
            },
            "test_class_distribution": {
                "real": int((test_df["label"] == 0).sum()),
                "fake": int((test_df["label"] == 1).sum())
            }
        },
        "model_architecture": "EfficientNet-B0 (ImageNet-pretrained)",
        "training_time_seconds": round(total_train_time, 2),
        "best_epoch": best_epoch,
        "best_val_f1": round(float(best_val_f1), 4),
        "best_val_loss": round(float(best_val_loss), 4),
        "test_metrics": {
            "accuracy": round(float(test_res["acc"]), 4),
            "precision": round(float(test_res["prec"]), 4),
            "recall": round(float(test_res["rec"]), 4),
            "f1_score": round(float(test_res["f1"]), 4),
            "roc_auc": round(float(test_res["auc"]), 4),
            "false_positive_rate": round(float(fpr), 4),
            "false_negative_rate": round(float(fnr), 4),
            "confusion_matrix": {
                "true_negative": int(tn),
                "false_positive": int(fp),
                "false_negative": int(fn),
                "true_positive": int(tp)
            }
        },
        "per_class_accuracy": {
            "real_class_accuracy": round(float(real_acc), 4),
            "fake_class_accuracy": round(float(fake_acc), 4)
        },
        "forgery_type_metrics": forgery_metrics,
        "document_type_metrics": doc_metrics,
        "training_history": history
    }
    
    with open(OUTPUT_PHASE2 / "phase2_results.json", "w", encoding="utf-8") as f:
        json.dump(results_json, f, indent=2)
        
    print(f"Phase 2 complete! Results written to: {OUTPUT_PHASE2 / 'phase2_results.json'}")
    return results_json


if __name__ == "__main__":
    run_phase2()
