#!/usr/bin/env python3
"""
DocVerify - Phase 6 Production-Equivalent Tiled & Canonical Prior Field Inspection
Author: Senior ML, Computer Vision & Security Research Engineer

Evaluates local patch verification under strict PRODUCTION-EQUIVALENT conditions:
1. No ground truth bounding boxes are used.
2. Two realistic extraction strategies:
   a) Canonical Template Prior Fields: uses CANONICAL_TEMPLATE_LAYOUTS to extract predefined field locations
   b) Uniform Sliding-Window Tiling: slices the document into overlapping native-scale tiles (e.g. 224x224, stride 112)
3. Evaluates aggregation methods:
   - MAX: maximum patch forgery probability across all inspected crops
   - MEAN: average patch forgery probability
   - TOP-3 MEAN: average of the top 3 highest forgery probabilities
4. Tests on the frozen test set (N=320) and compares against baseline & oracle performance.

Outputs:
- outputs/phase6/tiled_inspection_results.json
"""

import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import efficientnet_b0
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    PHASE6_OUTPUTS, PHASE6_MODELS_DIR, FROZEN_TEST_MANIFEST_PATH,
    IMAGENET_MEAN, IMAGENET_STD, setup_logger
)
from src.template_layouts import CANONICAL_TEMPLATE_LAYOUTS

logger = setup_logger("TiledInspection")


def load_patch_model(checkpoint_path: Path):
    """Loads the trained dedicated patch model."""
    model = efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def get_canonical_crops(img: Image.Image, template_type: str, context_scale=1.5):
    """Extracts candidate crops using canonical template priors."""
    w, h = img.size
    layout = CANONICAL_TEMPLATE_LAYOUTS.get(template_type)
    crops = []
    
    if layout and "fields" in layout:
        for field_name, f_data in layout["fields"].items():
            nx, ny, nw, nh = f_data["norm_bbox"]
            bx, by, bw, bh = int(nx * w), int(ny * h), int(nw * w), int(nh * h)
            
            # Add context margin
            cx, cy = bx + bw / 2.0, by + bh / 2.0
            cw = max(bw * context_scale, 64)
            ch = max(bh * context_scale, 64)
            
            x1 = max(0, int(cx - cw / 2.0))
            y1 = max(0, int(cy - ch / 2.0))
            x2 = min(w, int(cx + cw / 2.0))
            y2 = min(h, int(cy + ch / 2.0))
            
            crop = img.crop((x1, y1, x2, y2))
            crops.append((crop, f"prior_{field_name}"))
            
    if not crops:
        # Fallback to central horizontal bands if template unknown
        bands = [
            (int(0.25 * w), int(0.20 * h), int(0.50 * w), int(0.25 * h)),
            (int(0.25 * w), int(0.45 * h), int(0.50 * w), int(0.25 * h)),
            (int(0.25 * w), int(0.70 * h), int(0.50 * w), int(0.25 * h)),
        ]
        for idx, (bx, by, bw, bh) in enumerate(bands):
            crops.append((img.crop((bx, by, bx + bw, by + bh)), f"generic_band_{idx}"))
            
    return crops


def get_sliding_window_crops(img: Image.Image, tile_size=256, stride=160):
    """Extracts overlapping uniform sliding tiles across the document canvas."""
    w, h = img.size
    crops = []
    
    # If image smaller than tile size, return image itself
    if w <= tile_size or h <= tile_size:
        return [(img, "full_tile")]
        
    y = 0
    while y < h:
        x = 0
        while x < w:
            x2 = min(w, x + tile_size)
            y2 = min(h, y + tile_size)
            x1 = max(0, x2 - tile_size)
            y1 = max(0, y2 - tile_size)
            
            crop = img.crop((x1, y1, x2, y2))
            crops.append((crop, f"tile_{x1}_{y1}"))
            
            if x2 >= w:
                break
            x += stride
        if y2 >= h:
            break
        y += stride
        
    return crops


def score_crops(model, crops, transform):
    """Runs batch inference on crops and returns list of forgery probabilities."""
    if not crops:
        return [0.0]
        
    tensors = [transform(c[0]) for c in crops]
    batch = torch.stack(tensors)
    with torch.no_grad():
        outputs = model(batch)
        probs = torch.softmax(outputs, dim=1)[:, 1].numpy()
    return probs.tolist()


def evaluate_tiled_inspection():
    logger.info("Initializing Production-Equivalent Tiled & Canonical Prior Inspection...")
    
    checkpoint_path = PHASE6_MODELS_DIR / "dedicated_patch_model.pth"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing patch model checkpoint at {checkpoint_path}")
        
    model = load_patch_model(checkpoint_path)
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    
    df_test = pd.read_csv(FROZEN_TEST_MANIFEST_PATH)
    logger.info(f"Loaded frozen test manifest with {len(df_test)} samples.")
    
    # Store aggregated scores per sample
    canonical_max, canonical_mean, canonical_top3 = [], [], []
    tiled_max, tiled_mean, tiled_top3 = [], [], []
    targets = []
    
    records = []
    
    for idx, row in df_test.iterrows():
        p = Path(row['absolute_path'])
        if not p.exists():
            p = PROJECT_ROOT / row['image_path']
            
        label = int(row['label'])
        template = str(row['template'])
        targets.append(label)
        
        with Image.open(p) as img:
            img_rgb = img.convert("RGB")
            
            # Strategy 1: Canonical Template Priors
            c_crops = get_canonical_crops(img_rgb, template)
            c_probs = score_crops(model, c_crops, transform)
            
            c_max_p = float(np.max(c_probs))
            c_mean_p = float(np.mean(c_probs))
            c_top3_p = float(np.mean(sorted(c_probs, reverse=True)[:min(3, len(c_probs))]))
            
            canonical_max.append(c_max_p)
            canonical_mean.append(c_mean_p)
            canonical_top3.append(c_top3_p)
            
            # Strategy 2: Uniform Sliding-Window Tiling (256x256, stride 160)
            t_crops = get_sliding_window_crops(img_rgb, tile_size=256, stride=160)
            t_probs = score_crops(model, t_crops, transform)
            
            t_max_p = float(np.max(t_probs))
            t_mean_p = float(np.mean(t_probs))
            t_top3_p = float(np.mean(sorted(t_probs, reverse=True)[:min(3, len(t_probs))]))
            
            tiled_max.append(t_max_p)
            tiled_mean.append(t_mean_p)
            tiled_top3.append(t_top3_p)
            
            records.append({
                "image_name": row['image_filename'],
                "template_type": template,
                "label": label,
                "canonical_max": c_max_p,
                "canonical_mean": c_mean_p,
                "canonical_top3": c_top3_p,
                "tiled_max": t_max_p,
                "tiled_mean": t_mean_p,
                "tiled_top3": t_top3_p
            })
            
        if (idx + 1) % 40 == 0 or (idx + 1) == len(df_test):
            logger.info(f"Processed {idx + 1}/{len(df_test)} test documents...")
            
    targets = np.array(targets)
    
    def compute_metrics(scores, threshold=0.5):
        preds = (np.array(scores) >= threshold).astype(int)
        acc = float(accuracy_score(targets, preds))
        prec = float(precision_score(targets, preds, zero_division=0))
        rec = float(recall_score(targets, preds, zero_division=0))
        f1 = float(f1_score(targets, preds, zero_division=0))
        auc = float(roc_auc_score(targets, scores))
        return {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "roc_auc": auc
        }
        
    results = {
        "dataset_size": len(targets),
        "reals": int(np.sum(targets == 0)),
        "fakes": int(np.sum(targets == 1)),
        "canonical_priors": {
            "max_aggregation": compute_metrics(canonical_max),
            "mean_aggregation": compute_metrics(canonical_mean),
            "top3_mean_aggregation": compute_metrics(canonical_top3)
        },
        "uniform_tiling": {
            "max_aggregation": compute_metrics(tiled_max),
            "mean_aggregation": compute_metrics(tiled_mean),
            "top3_mean_aggregation": compute_metrics(tiled_top3)
        }
    }
    
    logger.info("============================================================")
    logger.info("PRODUCTION-EQUIVALENT LOCAL INSPECTION BENCHMARK RESULTS")
    logger.info("============================================================")
    logger.info(f"Canonical Priors (MAX):      Acc: {results['canonical_priors']['max_aggregation']['accuracy']:.4f}, AUC: {results['canonical_priors']['max_aggregation']['roc_auc']:.4f}")
    logger.info(f"Canonical Priors (TOP3):     Acc: {results['canonical_priors']['top3_mean_aggregation']['accuracy']:.4f}, AUC: {results['canonical_priors']['top3_mean_aggregation']['roc_auc']:.4f}")
    logger.info(f"Uniform Tiling (MAX):        Acc: {results['uniform_tiling']['max_aggregation']['accuracy']:.4f}, AUC: {results['uniform_tiling']['max_aggregation']['roc_auc']:.4f}")
    logger.info(f"Uniform Tiling (TOP3):       Acc: {results['uniform_tiling']['top3_mean_aggregation']['accuracy']:.4f}, AUC: {results['uniform_tiling']['top3_mean_aggregation']['roc_auc']:.4f}")
    logger.info(f"Uniform Tiling (MEAN):       Acc: {results['uniform_tiling']['mean_aggregation']['accuracy']:.4f}, AUC: {results['uniform_tiling']['mean_aggregation']['roc_auc']:.4f}")
    logger.info("============================================================")
    
    out_json = PHASE6_OUTPUTS / "tiled_inspection_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
        
    out_csv = PHASE6_OUTPUTS / "tiled_inspection_predictions.csv"
    pd.DataFrame(records).to_csv(out_csv, index=False)
    
    logger.info(f"Saved results to {out_json} and {out_csv}")
    return results


if __name__ == "__main__":
    evaluate_tiled_inspection()
