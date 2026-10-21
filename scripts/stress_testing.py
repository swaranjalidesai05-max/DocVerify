#!/usr/bin/env python3
"""
DocVerify - Phase 6 Environmental Stress Testing (Robustness Evaluation)
Author: Senior ML, Computer Vision & Security Research Engineer

Evaluates the robustness of the DocVerify system under real-world distortions:
1. JPEG Compression Degradation: Q in [30, 50, 70, 85, 95]
2. Gaussian Blur Perturbations: Sigma in [0.5, 1.0, 1.5, 2.0, 3.0]
3. Additive Gaussian Sensor Noise: Sigma in [5, 10, 15, 25]
4. Downscaling / Low Resolution: Scale factors [0.75, 0.50, 0.35]

Tests on frozen test set (N=320) and records graceful degradation curves.

Outputs:
- outputs/phase6/robustness_stress_testing.json
- outputs/phase6/robustness_curves.csv
"""

import sys
import json
import io
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image, ImageFilter
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import torch
import torchvision.transforms as transforms
from torchvision.models import efficientnet_b0

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    PHASE6_OUTPUTS, FROZEN_TEST_MANIFEST_PATH, PHASE2_MODEL_PATH,
    IMAGENET_MEAN, IMAGENET_STD, setup_logger
)

logger = setup_logger("StressTesting")


def load_baseline_model():
    model = efficientnet_b0(weights=None)
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 2)
    checkpoint = torch.load(PHASE2_MODEL_PATH, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def apply_jpeg_compression(img: Image.Image, quality: int):
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def apply_gaussian_blur(img: Image.Image, radius: float):
    return img.filter(ImageFilter.GaussianBlur(radius=radius))


def apply_gaussian_noise(img: Image.Image, sigma: float):
    arr = np.asarray(img, dtype=np.float32)
    noise = np.random.normal(0, sigma, arr.shape)
    noisy = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy)


def apply_downscaling(img: Image.Image, scale: float):
    w, h = img.size
    nw, nh = max(32, int(w * scale)), max(32, int(h * scale))
    down = img.resize((nw, nh), resample=Image.Resampling.BILINEAR)
    return down.resize((w, h), resample=Image.Resampling.BICUBIC)


def evaluate_perturbation(model, df_test, perturb_fn, transform):
    probs, targets = [], []
    for idx, row in df_test.iterrows():
        p = Path(row['absolute_path'])
        if not p.exists():
            p = PROJECT_ROOT / row['image_path']
        label = int(row['label'])
        targets.append(label)
        
        with Image.open(p) as img:
            img_rgb = img.convert("RGB")
            perturbed = perturb_fn(img_rgb)
            tensor = transform(perturbed).unsqueeze(0)
            with torch.no_grad():
                out = model(tensor)
                p_fake = float(torch.softmax(out, dim=1)[0, 1].numpy())
            probs.append(p_fake)
            
    targets = np.array(targets)
    preds = (np.array(probs) >= 0.5).astype(int)
    
    return {
        "accuracy": float(accuracy_score(targets, preds)),
        "f1_score": float(f1_score(targets, preds, zero_division=0)),
        "roc_auc": float(roc_auc_score(targets, probs))
    }


def run_stress_testing():
    logger.info("Initializing Environmental Stress Testing...")
    model = load_baseline_model()
    
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    
    df_test = pd.read_csv(FROZEN_TEST_MANIFEST_PATH)
    logger.info(f"Loaded test set ({len(df_test)} samples).")
    
    perturbations = {}
    curve_records = []
    
    # 1. JPEG Compression Tests
    jpeg_qualities = [95, 85, 70, 50, 30]
    logger.info(f"Running JPEG compression tests: {jpeg_qualities}")
    jpeg_results = {}
    for q in jpeg_qualities:
        m = evaluate_perturbation(model, df_test, lambda im: apply_jpeg_compression(im, q), transform)
        jpeg_results[f"Q_{q}"] = m
        curve_records.append({"perturbation": "JPEG Compression", "severity": q, **m})
        logger.info(f"JPEG Q={q:2d} -> Acc: {m['accuracy']:.4f}, AUC: {m['roc_auc']:.4f}")
    perturbations["jpeg_compression"] = jpeg_results
    
    # 2. Gaussian Blur Tests
    blur_radii = [0.5, 1.0, 1.5, 2.0, 3.0]
    logger.info(f"Running Gaussian blur tests: {blur_radii}")
    blur_results = {}
    for r in blur_radii:
        m = evaluate_perturbation(model, df_test, lambda im: apply_gaussian_blur(im, r), transform)
        blur_results[f"sigma_{r}"] = m
        curve_records.append({"perturbation": "Gaussian Blur", "severity": r, **m})
        logger.info(f"Blur sigma={r:.1f} -> Acc: {m['accuracy']:.4f}, AUC: {m['roc_auc']:.4f}")
    perturbations["gaussian_blur"] = blur_results
    
    # 3. Additive Gaussian Noise Tests
    noise_sigmas = [5, 10, 15, 25]
    logger.info(f"Running Gaussian noise tests: {noise_sigmas}")
    noise_results = {}
    for s in noise_sigmas:
        m = evaluate_perturbation(model, df_test, lambda im: apply_gaussian_noise(im, s), transform)
        noise_results[f"noise_sigma_{s}"] = m
        curve_records.append({"perturbation": "Gaussian Noise", "severity": s, **m})
        logger.info(f"Noise sigma={s:2d} -> Acc: {m['accuracy']:.4f}, AUC: {m['roc_auc']:.4f}")
    perturbations["gaussian_noise"] = noise_results
    
    # 4. Downscaling Tests
    scale_factors = [0.75, 0.50, 0.35]
    logger.info(f"Running Downscaling tests: {scale_factors}")
    downscale_results = {}
    for sc in scale_factors:
        m = evaluate_perturbation(model, df_test, lambda im: apply_downscaling(im, sc), transform)
        downscale_results[f"scale_{sc}"] = m
        curve_records.append({"perturbation": "Downscaling", "severity": sc, **m})
        logger.info(f"Scale={sc:.2f} -> Acc: {m['accuracy']:.4f}, AUC: {m['roc_auc']:.4f}")
    perturbations["downscaling"] = downscale_results
    
    # Save results
    with open(PHASE6_OUTPUTS / "robustness_stress_testing.json", "w", encoding="utf-8") as f:
        json.dump(perturbations, f, indent=2)
        
    pd.DataFrame(curve_records).to_csv(PHASE6_OUTPUTS / "robustness_curves.csv", index=False)
    logger.info("Stress testing completed successfully.")
    return perturbations


if __name__ == "__main__":
    run_stress_testing()
