#!/usr/bin/env python3
"""
DocVerify - Phase 6 Shortcut & Spurious Bias Audit
Author: Senior ML, Computer Vision & Security Research Engineer

Investigates whether non-semantic metadata, color statistics, resolution,
and compression artifacts have predictive power separating REAL vs FAKE documents.
Trains diagnostic classifiers (Logistic Regression / Random Forest) on:
1. Metadata features (Width, Height, Aspect Ratio, File Size)
2. Color statistics (Mean R, G, B, Std R, G, B, Brightness, Contrast)
3. Noise & Sharpness features (Laplacian variance, Sobel gradients)
4. Combined Non-Semantic Shortcut Classifier

Outputs:
- outputs/phase6/shortcut_results.json
"""

import os
import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import cv2
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score

# Add project root to sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import MANIFEST_PATH, PHASE6_OUTPUTS, setup_logger, RANDOM_SEED

logger = setup_logger("ShortcutAudit")


def extract_non_semantic_features(img_path: Path):
    """Extracts purely non-semantic metadata, color, and noise statistics."""
    file_size = img_path.stat().st_size
    
    with Image.open(img_path) as img:
        color_mode_is_cmyk = 1 if img.mode == "CMYK" else 0
        img_rgb = img.convert("RGB")
        w, h = img.size
        aspect_ratio = float(w) / float(h)
        
        arr = np.array(img_rgb)
        
    # Color channel statistics
    r_mean = float(np.mean(arr[:, :, 0]))
    g_mean = float(np.mean(arr[:, :, 1]))
    b_mean = float(np.mean(arr[:, :, 2]))
    r_std = float(np.std(arr[:, :, 0]))
    g_std = float(np.std(arr[:, :, 1]))
    b_std = float(np.std(arr[:, :, 2]))
    
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))
    
    # Sharpness / noise proxy
    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    
    return {
        # Metadata group
        "width": w,
        "height": h,
        "aspect_ratio": aspect_ratio,
        "file_size": file_size,
        "color_mode_is_cmyk": color_mode_is_cmyk,
        # Color group
        "r_mean": r_mean,
        "g_mean": g_mean,
        "b_mean": b_mean,
        "r_std": r_std,
        "g_std": g_std,
        "b_std": b_std,
        "brightness": brightness,
        "contrast": contrast,
        # Noise group
        "laplacian_var": laplacian_var
    }


def run_shortcut_audit():
    logger.info("Starting Phase 6 Shortcut & Bias Diagnostic Audit...")
    df = pd.read_csv(MANIFEST_PATH)
    
    features = []
    logger.info("Extracting non-semantic features across all 2,222 documents...")
    for idx, row in df.iterrows():
        p = Path(row['absolute_path'])
        if not p.exists():
            p = PROJECT_ROOT / row['image_path']
        feats = extract_non_semantic_features(p)
        feats['split'] = row['split']
        feats['label'] = row['label']
        feats['template'] = row['template']
        features.append(feats)
        
    feat_df = pd.DataFrame(features)
    
    train_data = feat_df[feat_df['split'] == 'train']
    val_data = feat_df[feat_df['split'] == 'val']
    test_data = feat_df[feat_df['split'] == 'test']
    
    feature_groups = {
        "metadata_only": ["width", "height", "aspect_ratio", "file_size"],
        "color_mode_only": ["color_mode_is_cmyk"],
        "color_stats_only": ["r_mean", "g_mean", "b_mean", "r_std", "g_std", "b_std", "brightness", "contrast"],
        "noise_sharpness_only": ["laplacian_var"],
        "combined_shortcuts": [
            "width", "height", "aspect_ratio", "file_size", "color_mode_is_cmyk",
            "r_mean", "g_mean", "b_mean", "r_std", "g_std", "b_std", "brightness", "contrast", "laplacian_var"
        ]
    }
    
    results = {}
    
    for group_name, cols in feature_groups.items():
        logger.info(f"Training diagnostic classifier on feature group: {group_name}")
        X_train = train_data[cols].values
        y_train = train_data['label'].values
        X_test = test_data[cols].values
        y_test = test_data['label'].values
        
        # Normalize
        mean = np.mean(X_train, axis=0, keepdims=True)
        std = np.std(X_train, axis=0, keepdims=True) + 1e-7
        X_train_norm = (X_train - mean) / std
        X_test_norm = (X_test - mean) / std
        
        clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=RANDOM_SEED)
        clf.fit(X_train_norm, y_train)
        
        preds = clf.predict(X_test_norm)
        probs = clf.predict_proba(X_test_norm)[:, 1]
        
        acc = float(accuracy_score(y_test, preds))
        roc_auc = float(roc_auc_score(y_test, probs))
        f1 = float(f1_score(y_test, preds, zero_division=0))
        prec = float(precision_score(y_test, preds, zero_division=0))
        rec = float(recall_score(y_test, preds, zero_division=0))
        
        results[group_name] = {
            "features": cols,
            "accuracy": acc,
            "roc_auc": roc_auc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1
        }
        logger.info(f"Group [{group_name}] Test Acc: {acc:.4f}, AUC: {roc_auc:.4f}")

    # Critical CMYK analysis
    cmyk_reals = df[df['color_mode'] == 'CMYK']
    logger.info(f"Total CMYK images: {len(cmyk_reals)} (All are REAL documents).")
    
    # Save results
    out_json = PHASE6_OUTPUTS / "shortcut_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
        
    logger.info(f"Saved shortcut audit results to {out_json}")
    return results


if __name__ == "__main__":
    run_shortcut_audit()
