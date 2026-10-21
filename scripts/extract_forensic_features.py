#!/usr/bin/env python3
"""
DocVerify - Phase 6 Forensic Feature Extractor (ELA, Noise, Edge Consistency)
Author: Senior ML, Computer Vision & Security Research Engineer

Extracts independent scientific forensic features without ground truth:
1. Error Level Analysis (ELA): Computes digital difference after JPEG recompression (Q=90).
   Measures mean, std, max, and high-frequency energy of compression residuals.
2. Noise Residual Analysis: Uses median filtering / Wavelet/Laplacian denoising to isolate
   sensor noise pattern, computing local noise variance consistency across quadrants.
3. Edge / Frequency Sharpness: High-pass Laplacian energy to detect sharpness mismatch
   between inpainted/spliced characters and the document background.

Generates:
- outputs/phase6/forensic_features_test.csv
- outputs/phase6/forensic_features_val.csv
- outputs/phase6/forensic_feature_distributions.json
"""

import sys
import json
import io
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image, ImageChops, ImageEnhance, ImageFilter

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    MANIFEST_PATH, FROZEN_TEST_MANIFEST_PATH, PHASE6_OUTPUTS, setup_logger
)

logger = setup_logger("ForensicFeatureExtractor")


def compute_ela_features(img: Image.Image, quality=90, scale=10):
    """
    Computes Error Level Analysis (ELA) statistics.
    Recompresses image to JPEG at specified quality and measures residual difference.
    """
    # Convert to RGB
    img_rgb = img.convert("RGB")
    
    # Save to memory buffer as JPEG
    buf = io.BytesIO()
    img_rgb.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    recompressed = Image.open(buf)
    
    # Compute absolute difference
    diff = ImageChops.difference(img_rgb, recompressed)
    diff_arr = np.asarray(diff, dtype=np.float32)
    
    mean_diff = float(np.mean(diff_arr))
    std_diff = float(np.std(diff_arr))
    max_diff = float(np.max(diff_arr))
    
    # Measure regional variance (quadrants)
    h, w, _ = diff_arr.shape
    q1 = diff_arr[:h//2, :w//2]
    q2 = diff_arr[:h//2, w//2:]
    q3 = diff_arr[h//2:, :w//2]
    q4 = diff_arr[h//2:, w//2:]
    
    q_means = [np.mean(q1), np.mean(q2), np.mean(q3), np.mean(q4)]
    ela_inconsistency = float(np.std(q_means))
    
    return {
        "ela_mean": mean_diff,
        "ela_std": std_diff,
        "ela_max": max_diff,
        "ela_inconsistency": ela_inconsistency
    }


def compute_noise_residual_features(img: Image.Image):
    """
    Extracts high-frequency noise residual using 3x3 median filtering subtraction.
    Measures local noise variance across the document.
    """
    gray = img.convert("L")
    arr = np.asarray(gray, dtype=np.float32)
    
    # Apply median filter
    med = gray.filter(ImageFilter.MedianFilter(size=3))
    med_arr = np.asarray(med, dtype=np.float32)
    
    # Residual
    residual = np.abs(arr - med_arr)
    noise_mean = float(np.mean(residual))
    noise_std = float(np.std(residual))
    
    # Quadrant variance inconsistency
    h, w = residual.shape
    q1 = residual[:h//2, :w//2]
    q2 = residual[:h//2, w//2:]
    q3 = residual[h//2:, :w//2]
    q4 = residual[h//2:, w//2:]
    
    q_stds = [np.std(q1), np.std(q2), np.std(q3), np.std(q4)]
    noise_inconsistency = float(np.std(q_stds))
    
    return {
        "noise_mean": noise_mean,
        "noise_std": noise_std,
        "noise_inconsistency": noise_inconsistency
    }


def compute_laplacian_edge_features(img: Image.Image):
    """
    Computes Laplacian high-pass filter energy to detect character edge sharpness anomalies.
    """
    gray = img.convert("L")
    lap = gray.filter(ImageFilter.Kernel((3, 3), [0, 1, 0, 1, -4, 1, 0, 1, 0], scale=1, offset=0))
    lap_arr = np.asarray(lap, dtype=np.float32)
    
    edge_var = float(np.var(lap_arr))
    edge_max = float(np.max(lap_arr))
    
    return {
        "edge_variance": edge_var,
        "edge_max": edge_max
    }


def extract_features_for_dataset(df: pd.DataFrame, split_name="test"):
    logger.info(f"Extracting forensic features for split: {split_name} ({len(df)} samples)...")
    records = []
    
    for idx, row in df.iterrows():
        p = Path(row['absolute_path'])
        if not p.exists():
            p = PROJECT_ROOT / row['image_path']
            
        label = int(row['label'])
        template = str(row.get('template', row.get('template_type', 'unknown')))
        img_name = str(row.get('image_filename', row.get('image_name', p.name)))
        
        try:
            with Image.open(p) as img:
                ela_feats = compute_ela_features(img)
                noise_feats = compute_noise_residual_features(img)
                edge_feats = compute_laplacian_edge_features(img)
                
            rec = {
                "image_filename": img_name,
                "split": split_name,
                "template": template,
                "label": label,
                **ela_feats,
                **noise_feats,
                **edge_feats
            }
            records.append(rec)
        except Exception as e:
            logger.warning(f"Error extracting forensics for {p}: {e}")
            
        if (idx + 1) % 50 == 0 or (idx + 1) == len(df):
            logger.info(f"[{split_name}] Processed {idx + 1}/{len(df)} images...")
            
    df_out = pd.DataFrame(records)
    return df_out


def run_forensic_extraction():
    PHASE6_OUTPUTS.mkdir(parents=True, exist_ok=True)
    
    # 1. Test split
    df_test = pd.read_csv(FROZEN_TEST_MANIFEST_PATH)
    df_forensic_test = extract_features_for_dataset(df_test, split_name="test")
    test_csv = PHASE6_OUTPUTS / "forensic_features_test.csv"
    df_forensic_test.to_csv(test_csv, index=False)
    logger.info(f"Saved test forensic features to {test_csv}")
    
    # 2. Val split
    df_all = pd.read_csv(MANIFEST_PATH)
    df_val = df_all[df_all['split'] == 'val'].copy()
    df_forensic_val = extract_features_for_dataset(df_val, split_name="val")
    val_csv = PHASE6_OUTPUTS / "forensic_features_val.csv"
    df_forensic_val.to_csv(val_csv, index=False)
    logger.info(f"Saved val forensic features to {val_csv}")
    
    # Compute summary statistics
    real_test = df_forensic_test[df_forensic_test['label'] == 0]
    fake_test = df_forensic_test[df_forensic_test['label'] == 1]
    
    summary = {
        "test_counts": {"real": len(real_test), "fake": len(fake_test)},
        "real_distributions": {
            "ela_mean": float(real_test['ela_mean'].mean()),
            "ela_inconsistency": float(real_test['ela_inconsistency'].mean()),
            "noise_inconsistency": float(real_test['noise_inconsistency'].mean()),
            "edge_variance": float(real_test['edge_variance'].mean())
        },
        "fake_distributions": {
            "ela_mean": float(fake_test['ela_mean'].mean()),
            "ela_inconsistency": float(fake_test['ela_inconsistency'].mean()),
            "noise_inconsistency": float(fake_test['noise_inconsistency'].mean()),
            "edge_variance": float(fake_test['edge_variance'].mean())
        }
    }
    
    summary_json = PHASE6_OUTPUTS / "forensic_feature_distributions.json"
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        
    logger.info(f"Forensic summary: {json.dumps(summary, indent=2)}")
    return summary


if __name__ == "__main__":
    run_forensic_extraction()
