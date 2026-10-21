#!/usr/bin/env python3
"""
DocVerify - Phase 6 Preprocessing Geometry & Image Standardization Evaluation
Author: Senior ML, Computer Vision & Security Research Engineer

Evaluates the impact of:
1. Preprocessing Geometry: CenterCrop(224) vs Letterbox Padding(224) vs Letterbox Padding(384)
2. Color Standardization: sRGB conversion
3. Compression Standardization: Uniform JPEG recompression (Q=85) to eliminate file-size/compression shortcuts
Evaluated on the frozen test set (N=320) using the frozen baseline EfficientNet-B0.

Outputs:
- outputs/phase6/preprocessing_ablation.json
"""

import sys
import io
import json
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import efficientnet_b0
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, brier_score_loss, confusion_matrix

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    FROZEN_TEST_MANIFEST_PATH, PHASE2_MODEL_PATH, PHASE6_OUTPUTS,
    IMAGENET_MEAN, IMAGENET_STD, setup_logger
)

logger = setup_logger("PreprocessingEval")


def letterbox_image(image: Image.Image, target_size=(224, 224), fill_color=(128, 128, 128)) -> Image.Image:
    """Resizes an image preserving aspect ratio and pads with fill_color to reach target_size."""
    orig_w, orig_h = image.size
    target_w, target_h = target_size
    
    scale = min(target_w / orig_w, target_h / orig_h)
    new_w = int(round(orig_w * scale))
    new_h = int(round(orig_h * scale))
    
    resized = image.resize((new_w, new_h), Image.Resampling.BILINEAR)
    new_img = Image.new("RGB", (target_w, target_h), fill_color)
    
    pad_left = (target_w - new_w) // 2
    pad_top = (target_h - new_h) // 2
    new_img.paste(resized, (pad_left, pad_top))
    return new_img


def load_model():
    model = efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    state = torch.load(PHASE2_MODEL_PATH, map_location="cpu", weights_only=False)
    if "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    else:
        model.load_state_dict(state)
    model.eval()
    return model


def run_preprocessing_evaluation():
    logger.info("Starting Phase 6 Preprocessing Geometry & Standardization Evaluation...")
    test_df = pd.read_csv(FROZEN_TEST_MANIFEST_PATH)
    model = load_model()
    
    norm = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    to_tensor = transforms.ToTensor()
    
    # 1. Historical CenterCrop Preprocessing
    centercrop_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        norm
    ])
    
    configs = {
        "A_centercrop_224_baseline": {
            "desc": "Historical Phase 2: Resize(256) -> CenterCrop(224)",
            "process": lambda img: centercrop_transform(img.convert("RGB"))
        },
        "B_letterbox_224": {
            "desc": "Geometry-Preserving: Aspect-Ratio Letterbox Padding to 224x224",
            "process": lambda img: norm(to_tensor(letterbox_image(img.convert("RGB"), (224, 224))))
        },
        "C_letterbox_384": {
            "desc": "High-Resolution Letterbox: Aspect-Ratio Letterbox Padding to 384x384",
            "process": lambda img: norm(to_tensor(letterbox_image(img.convert("RGB"), (384, 384))))
        },
        "D_standardized_jpeg_recompressed_224": {
            "desc": "Standardized Compression: Uniform JPEG Recompression (Q=85) + CenterCrop(224)",
            "process": None  # Handled in loop
        }
    }
    
    results = {}
    
    for cfg_key, cfg in configs.items():
        logger.info(f"Evaluating configuration: {cfg_key} ({cfg['desc']})")
        y_true = []
        y_pred = []
        y_prob = []
        
        with torch.no_grad():
            for idx, row in test_df.iterrows():
                p = Path(row['absolute_path'])
                if not p.exists():
                    p = PROJECT_ROOT / row['image_path']
                    
                with Image.open(p) as img:
                    if cfg_key == "D_standardized_jpeg_recompressed_224":
                        # Standardize JPEG in memory to eliminate compression shortcut
                        buf = io.BytesIO()
                        img.convert("RGB").save(buf, format="JPEG", quality=85)
                        buf.seek(0)
                        std_img = Image.open(buf)
                        tensor = centercrop_transform(std_img)
                    else:
                        tensor = cfg["process"](img)
                        
                tensor = tensor.unsqueeze(0)
                logits = model(tensor)
                probs = torch.softmax(logits, dim=1).squeeze().numpy()
                pred = int(np.argmax(probs))
                
                y_true.append(int(row['label']))
                y_pred.append(pred)
                y_prob.append(float(probs[1]))
                
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        y_prob = np.array(y_prob)
        
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        acc = float(accuracy_score(y_true, y_pred))
        prec = float(precision_score(y_true, y_pred, zero_division=0))
        rec = float(recall_score(y_true, y_pred, zero_division=0))
        f1 = float(f1_score(y_true, y_pred, zero_division=0))
        roc_auc = float(roc_auc_score(y_true, y_prob))
        brier = float(brier_score_loss(y_true, y_prob))
        
        results[cfg_key] = {
            "description": cfg["desc"],
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "roc_auc": roc_auc,
            "brier_score": brier,
            "confusion_matrix": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)}
        }
        logger.info(f"[{cfg_key}] Acc: {acc:.4f}, Prec: {prec:.4f}, Rec: {rec:.4f}, F1: {f1:.4f}, AUC: {roc_auc:.4f}, FN: {fn}, FP: {fp}")
        
    out_path = PHASE6_OUTPUTS / "preprocessing_ablation.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved preprocessing ablation results to {out_path}")
    return results


if __name__ == "__main__":
    run_preprocessing_evaluation()
