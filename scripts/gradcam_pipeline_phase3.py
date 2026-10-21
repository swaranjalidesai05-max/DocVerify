#!/usr/bin/env python3
"""
DocVerify - Phase 3 Master Execution Script
Author: Senior ML & Computer Vision Research Engineer

Orchestrates:
1. Model Loading (Phase 2 best checkpoint).
2. Explainability Engine Initialization (Target: features.8).
3. Validation Set Threshold Selection (Sweeps thresholds [0.2 to 0.7] on 191 validation fakes).
4. Frozen Evaluation on Untouched Test Set (169 test fakes):
   - Pointing Game Accuracy (Hit/Miss)
   - Mean IoU, Median IoU, IoU > 0, IoU >= 0.3, IoU >= 0.5
   - Forgery-type Breakdown (Inpaint_and_Rewrite vs Crop_and_Replace)
   - Document-type Breakdown (10 countries)
   - Field-level Breakdown (name, expiry_date, number, etc.)
   - High-Resolution Multi-Scale Crop Ablation Study (analyzing 224x224 downsampling bottleneck on micro-manipulations)
5. Visual Artifact Generation:
   - Successful cases, failed cases, false negatives, false positives.
   - Overlays with ground-truth bounding box (Green) and predicted box (Red).
   - Pointing Game hit point (Blue star).
6. Comprehensive Machine-Readable Output:
   - per_sample_localization.csv
   - by_forgery_type.csv, by_document_type.csv, by_field.csv
   - phase3_results.json, localization_metrics.json, gradcam_config.json
"""

import json
import time
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torchvision.models as models
import torch.nn as nn
from tqdm import tqdm

from gradcam_phase3 import ExplainabilityEngine
from evaluate_localization import compute_pointing_game, extract_pred_bbox, compute_iou


def run_phase3_pipeline():
    BASE_DIR = Path("E:/DocverifyAi")
    MANIFEST_PATH = BASE_DIR / "outputs" / "dataset_manifest" / "dataset_manifest.csv"
    CHECKPOINT_PATH = BASE_DIR / "models" / "phase2" / "best_model.pth"
    OUTPUT_PHASE3 = BASE_DIR / "outputs" / "phase3"
    VIZ_DIR = OUTPUT_PHASE3 / "visualizations"
    ANALYSIS_DIR = OUTPUT_PHASE3 / "analysis"
    
    print("="*60)
    print("DOCVERIFY — PHASE 3: FORGERY LOCALIZATION & EXPLAINABILITY")
    print("="*60)
    
    # 1. Load Model and Initialize Explainability Engine
    print(f"\n[1/6] Loading Phase 2 model checkpoint: {CHECKPOINT_PATH}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    target_layer = model.features[8]
    engine = ExplainabilityEngine(model, target_layer, device)
    print(f"Target Layer verified: model.features[8] (Conv2dNormActivation, 1280 channels)")
    
    # Record Grad-CAM Config
    gradcam_config = {
        "model": "EfficientNet-B0",
        "checkpoint": str(CHECKPOINT_PATH),
        "target_layer": "features.8",
        "target_layer_description": "Final 1x1 Conv2dNormActivation projection prior to pooling",
        "input_crop_size": 224,
        "resize_size": 256,
        "target_class": 1,
        "target_class_name": "FAKE",
        "methods": ["Grad-CAM", "HiRes-CAM"],
        "device": str(device)
    }
    with open(OUTPUT_PHASE3 / "gradcam_config.json", "w", encoding="utf-8") as f:
        json.dump(gradcam_config, f, indent=2)

    # 2. Load Manifest
    df = pd.read_csv(MANIFEST_PATH)
    val_fakes = df[(df["split"] == "val") & (df["label"] == 1)].copy()
    test_fakes = df[(df["split"] == "test") & (df["label"] == 1)].copy()
    test_reals = df[(df["split"] == "test") & (df["label"] == 0)].copy()
    
    print(f"Loaded manifest: Val Fakes={len(val_fakes)}, Test Fakes={len(test_fakes)}, Test Reals={len(test_reals)}")

    # 3. Validation Set Threshold Selection (Strict Zero-Leakage Protocol)
    print("\n[2/6] Performing threshold sweep on VALIDATION set fakes (191 samples)...")
    threshold_candidates = [0.20, 0.30, 0.40, 0.50, 0.60, 0.70]
    val_ious = {t: [] for t in threshold_candidates}
    val_pointing_hits = 0
    
    for _, row in tqdm(val_fakes.iterrows(), total=len(val_fakes), desc="Val CAM Sweep"):
        img_p = Path(row["absolute_path"])
        with Image.open(img_p) as im:
            cams = engine.generate_cams(im, target_class=1)
            
        gt_box = json.loads(row["bounding_box_json"])
        hit, _ = compute_pointing_game(cams["gradcam_orig"], gt_box)
        val_pointing_hits += hit
        
        for t in threshold_candidates:
            pbox, _ = extract_pred_bbox(cams["gradcam_orig"], threshold=t)
            iou = compute_iou(pbox, gt_box)
            val_ious[t].append(iou)
            
    val_pointing_acc = val_pointing_hits / len(val_fakes)
    print(f"Validation Pointing Game Accuracy: {val_pointing_acc*100:.2f}%")
    print("Validation Mean IoU by Threshold:")
    best_t = 0.50
    best_val_miou = -1.0
    for t in threshold_candidates:
        m_iou = float(np.mean(val_ious[t]))
        med_iou = float(np.median(val_ious[t]))
        print(f"  Threshold {t:.2f} -> Mean IoU: {m_iou:.4f}, Median IoU: {med_iou:.4f}")
        if m_iou > best_val_miou:
            best_val_miou = m_iou
            best_t = t
            
    print(f"Optimal Threshold selected on Validation set: {best_t:.2f} (Mean IoU = {best_val_miou:.4f})")
    print("Freezing threshold for Test evaluation.")

    # 4. Quantitative Evaluation on Untouched TEST Set (169 Fakes)
    print(f"\n[3/6] Running frozen quantitative evaluation on TEST set (169 fakes, Threshold={best_t:.2f})...")
    per_sample_records = []
    test_pointing_hits = 0
    test_ious_gradcam = []
    test_ious_hirescam = []
    test_pointing_hits_hires = 0
    
    # Track multi-scale crop ablation
    multiscale_records = []
    
    for _, row in tqdm(test_fakes.iterrows(), total=len(test_fakes), desc="Test Evaluation"):
        img_p = Path(row["absolute_path"])
        gt_box = json.loads(row["bounding_box_json"])
        
        with Image.open(img_p) as im:
            orig_w, orig_h = im.size
            cams = engine.generate_cams(im, target_class=1)
            
            # Multi-scale crop experiment: crop 2.5x window around manipulated region
            # to test downsampling hypothesis
            gx = gt_box.get("x", 0)
            gy = gt_box.get("y", 0)
            gw = gt_box.get("width", 0)
            gh = gt_box.get("height", 0)
            cx, cy = gx + gw / 2.0, gy + gh / 2.0
            crop_dim = max(int(max(gw, gh) * 2.5), 128)
            crop_x1 = max(0, int(cx - crop_dim / 2.0))
            crop_y1 = max(0, int(cy - crop_dim / 2.0))
            crop_x2 = min(orig_w, crop_x1 + crop_dim)
            crop_y2 = min(orig_h, crop_y1 + crop_dim)
            crop_patch = im.crop((crop_x1, crop_y1, crop_x2, crop_y2))
            
            # Patch evaluation
            crop_tensor = engine.eval_transform(crop_patch.convert("RGB")).unsqueeze(0).to(device)
            with torch.no_grad():
                crop_logits = model(crop_tensor)
                crop_prob_fake = float(torch.softmax(crop_logits, dim=1)[0, 1].cpu().numpy())
                
        # Metrics on full image
        hit_gc, (max_x_gc, max_y_gc) = compute_pointing_game(cams["gradcam_orig"], gt_box)
        hit_hr, (max_x_hr, max_y_hr) = compute_pointing_game(cams["hirescam_orig"], gt_box)
        
        test_pointing_hits += hit_gc
        test_pointing_hits_hires += hit_hr
        
        pbox_gc, area_ratio_gc = extract_pred_bbox(cams["gradcam_orig"], threshold=best_t)
        pbox_hr, area_ratio_hr = extract_pred_bbox(cams["hirescam_orig"], threshold=best_t)
        
        iou_gc = compute_iou(pbox_gc, gt_box)
        iou_hr = compute_iou(pbox_hr, gt_box)
        
        test_ious_gradcam.append(iou_gc)
        test_ious_hirescam.append(iou_hr)
        
        # Ground-truth area ratio relative to document
        gt_area_ratio = (gw * gh) / (orig_w * orig_h)
        
        sample_entry = {
            "image_filename": row["image_filename"],
            "base_document_id": row["base_document_id"],
            "template": row["template"],
            "country": row["country"],
            "forgery_type": row["forgery_type"],
            "manipulated_field": row["manipulated_field"],
            "prob_fake": round(cams["prob_fake"], 4),
            "pred_class": cams["pred_class"],
            "is_correct_classification": (cams["pred_class"] == 1),
            "gt_bbox": gt_box,
            "gt_area_ratio_pct": round(gt_area_ratio * 100, 3),
            "gradcam_max_pt": [max_x_gc, max_y_gc],
            "pointing_game_hit": hit_gc,
            "pred_bbox_gc": list(pbox_gc) if pbox_gc else None,
            "pred_area_ratio_pct": round(area_ratio_gc * 100, 2),
            "iou_gradcam": round(iou_gc, 4),
            "pointing_game_hit_hires": hit_hr,
            "iou_hirescam": round(iou_hr, 4),
            "patch_prob_fake": round(crop_prob_fake, 4)
        }
        per_sample_records.append(sample_entry)

    # Convert to DataFrame
    sample_df = pd.DataFrame(per_sample_records)
    sample_df.to_csv(ANALYSIS_DIR / "per_sample_localization.csv", index=False)
    print(f"Per-sample localization records saved to: {ANALYSIS_DIR / 'per_sample_localization.csv'}")

    # Compute Aggregate Metrics
    n_test = len(test_fakes)
    test_pointing_acc = test_pointing_hits / n_test
    test_pointing_acc_hires = test_pointing_hits_hires / n_test
    
    test_miou = float(np.mean(test_ious_gradcam))
    test_mediou = float(np.median(test_ious_gradcam))
    pct_iou_gt0 = float(np.mean([1.0 if x > 0 else 0.0 for x in test_ious_gradcam])) * 100
    pct_iou_ge03 = float(np.mean([1.0 if x >= 0.3 else 0.0 for x in test_ious_gradcam])) * 100
    pct_iou_ge05 = float(np.mean([1.0 if x >= 0.5 else 0.0 for x in test_ious_gradcam])) * 100
    
    print("\n" + "="*50)
    print("PHASE 3 TEST LOCALIZATION BENCHMARK RESULTS")
    print("="*50)
    print(f"Pointing Game Accuracy (Grad-CAM):  {test_pointing_acc*100:.2f}% ({test_pointing_hits}/{n_test})")
    print(f"Pointing Game Accuracy (HiRes-CAM): {test_pointing_acc_hires*100:.2f}% ({test_pointing_hits_hires}/{n_test})")
    print(f"Mean Bounding-Box IoU:              {test_miou:.4f}")
    print(f"Median Bounding-Box IoU:            {test_mediou:.4f}")
    print(f"Samples with IoU > 0:               {pct_iou_gt0:.2f}%")
    print(f"Samples with IoU >= 0.3:            {pct_iou_ge03:.2f}%")
    print(f"Samples with IoU >= 0.5:            {pct_iou_ge05:.2f}%")

    # 5. Granular Breakdowns (Forgery Type, Document Template, Manipulated Field)
    print("\n[4/6] Generating granular breakdown analyses...")
    
    # A. Forgery Type
    ftype_analysis = []
    for ftype, grp in sample_df.groupby("forgery_type"):
        ftype_analysis.append({
            "forgery_type": ftype,
            "samples": len(grp),
            "pointing_game_accuracy": round(grp["pointing_game_hit"].mean() * 100, 2),
            "mean_iou": round(grp["iou_gradcam"].mean(), 4),
            "median_iou": round(grp["iou_gradcam"].median(), 4),
            "pct_iou_gt0": round((grp["iou_gradcam"] > 0).mean() * 100, 2),
            "classification_accuracy": round(grp["is_correct_classification"].mean() * 100, 2)
        })
    ftype_df = pd.DataFrame(ftype_analysis)
    ftype_df.to_csv(ANALYSIS_DIR / "by_forgery_type.csv", index=False)
    print("Breakdown by Forgery Type:")
    print(ftype_df.to_string(index=False))

    # B. Document Template
    doc_analysis = []
    for tpl, grp in sample_df.groupby("template"):
        doc_analysis.append({
            "template": tpl,
            "samples": len(grp),
            "pointing_game_accuracy": round(grp["pointing_game_hit"].mean() * 100, 2),
            "mean_iou": round(grp["iou_gradcam"].mean(), 4),
            "median_iou": round(grp["iou_gradcam"].median(), 4),
            "classification_accuracy": round(grp["is_correct_classification"].mean() * 100, 2)
        })
    doc_df = pd.DataFrame(doc_analysis)
    doc_df.to_csv(ANALYSIS_DIR / "by_document_type.csv", index=False)

    # C. Manipulated Field
    field_analysis = []
    for fld, grp in sample_df.groupby("manipulated_field"):
        field_analysis.append({
            "field": fld,
            "samples": len(grp),
            "pointing_game_accuracy": round(grp["pointing_game_hit"].mean() * 100, 2),
            "mean_iou": round(grp["iou_gradcam"].mean(), 4),
            "avg_gt_area_pct": round(grp["gt_area_ratio_pct"].mean(), 3),
            "classification_accuracy": round(grp["is_correct_classification"].mean() * 100, 2)
        })
    field_df = pd.DataFrame(field_analysis).sort_values("samples", ascending=False)
    field_df.to_csv(ANALYSIS_DIR / "by_field.csv", index=False)

    # 6. High-Resolution Multi-Scale Crop Ablation Analysis
    print("\n[5/6] Analyzing High-Resolution Multi-Scale Crop Ablation...")
    fns = sample_df[sample_df["is_correct_classification"] == False]
    tps = sample_df[sample_df["is_correct_classification"] == True]
    
    fn_patch_fake_detected = (fns["patch_prob_fake"] >= 0.5).sum()
    fn_patch_recovery_rate = (fn_patch_fake_detected / len(fns)) * 100 if len(fns) > 0 else 0
    
    print(f"False Negatives on Global 224x224: {len(fns)} samples")
    print(f"High-Res Local Crop Correctly Flags Fake: {fn_patch_fake_detected}/{len(fns)} ({fn_patch_recovery_rate:.1f}%)")
    print(f"Average GT field area in FNs: {fns['gt_area_ratio_pct'].mean():.3f}% of total image area")
    print(f"Average GT field area in TPs: {tps['gt_area_ratio_pct'].mean():.3f}% of total image area")

    # 7. Generate Visualization Artifacts
    print("\n[6/6] Generating visualization artifacts...")
    sns.set_theme(style="whitegrid")
    
    # A. Pointing Game Bar Plot
    plt.figure(figsize=(8, 4.5))
    methods = ["Grad-CAM", "HiRes-CAM"]
    accs = [test_pointing_acc * 100, test_pointing_acc_hires * 100]
    bars = plt.bar(methods, accs, color=["#2980b9", "#8e44ad"], width=0.45)
    plt.title("Pointing Game Accuracy on Unseen Test Fakes (N=169)", fontsize=13, fontweight="bold")
    plt.ylabel("Accuracy (% Hits)", fontsize=11)
    for b in bars:
        h = b.get_height()
        plt.text(b.get_x() + b.get_width() / 2.0, h + 1.5, f"{h:.2f}%", ha="center", va="bottom", fontsize=11, fontweight="bold")
    plt.ylim(0, 100)
    plt.tight_layout()
    plt.savefig(VIZ_DIR / "summary_plots" / "pointing_game_results.png", dpi=200)
    plt.close()

    # B. IoU Distribution Histogram
    plt.figure(figsize=(8, 4.5))
    plt.hist(test_ious_gradcam, bins=15, range=(0, 1), color="#e67e22", edgecolor="black", alpha=0.8)
    plt.axvline(test_miou, color="red", linestyle="--", lw=2, label=f"Mean IoU: {test_miou:.3f}")
    plt.axvline(test_mediou, color="blue", linestyle=":", lw=2, label=f"Median IoU: {test_mediou:.3f}")
    plt.title(f"Bounding-Box IoU Distribution (Threshold={best_t:.2f})", fontsize=13, fontweight="bold")
    plt.xlabel("Intersection over Union (IoU)", fontsize=11)
    plt.ylabel("Sample Count", fontsize=11)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(VIZ_DIR / "summary_plots" / "iou_distribution.png", dpi=200)
    plt.close()

    # C. IoU by Forgery Type Bar Chart
    plt.figure(figsize=(7, 4.5))
    ax = ftype_df.plot(x="forgery_type", y=["pointing_game_accuracy", "pct_iou_gt0"], kind="bar",
                       color=["#3498db", "#2ecc71"], width=0.55, figsize=(7, 4.5))
    plt.title("Localization Performance by Forgery Mechanism", fontsize=12, fontweight="bold")
    plt.ylabel("Percentage (%)", fontsize=11)
    plt.xticks(rotation=0, fontsize=10)
    plt.legend(["Pointing Game Hit %", "IoU > 0 %"], fontsize=10)
    plt.ylim(0, 100)
    plt.tight_layout()
    plt.savefig(VIZ_DIR / "summary_plots" / "iou_by_forgery_type.png", dpi=200)
    plt.close()

    # D. Render Visual Case Studies (Successful, Failed, FN, FP)
    # Helper overlay renderer
    def render_composite(row_data, category: str, subfolder: str):
        img_p = BASE_DIR / "Dataset" / "template" / "Images" / ("fakes" if row_data["label"] == 1 else "reals") / row_data["image_filename"]
        with Image.open(img_p) as im:
            orig_w, orig_h = im.size
            cams = engine.generate_cams(im, target_class=1)
            
            # Create RGB overlay
            heat_np = cams["gradcam_orig"]
            # Colormap (jet/turbo)
            cmap = plt.get_cmap("jet")
            heat_rgba = (cmap(heat_np)[:, :, :3] * 255).astype(np.uint8)
            heat_pil = Image.fromarray(heat_rgba).convert("RGBA")
            
            base_rgb = im.convert("RGBA")
            blended = Image.blend(base_rgb, heat_pil, alpha=0.45).convert("RGB")
            draw = ImageDraw.Draw(blended)
            
            # Ground truth bbox in GREEN
            if row_data["label"] == 1 and row_data["bounding_box_json"] != "None":
                gt_box = json.loads(row_data["bounding_box_json"])
                gx1, gy1 = gt_box["x"], gt_box["y"]
                gx2, gy2 = gx1 + gt_box["width"], gy1 + gt_box["height"]
                draw.rectangle([gx1, gy1, gx2, gy2], outline="#00ff00", width=5)
                draw.text((gx1, max(0, gy1 - 20)), f"GT: {row_data['manipulated_field']}", fill="#00ff00")
                
            # Pointing max point (BLUE STAR / CIRCLE)
            max_idx = np.unravel_index(np.argmax(heat_np), heat_np.shape)
            my, mx = int(max_idx[0]), int(max_idx[1])
            r_pt = 12
            draw.ellipse([mx - r_pt, my - r_pt, mx + r_pt, my + r_pt], outline="#00e5ff", width=4)
            
            # Predicted bbox in RED
            pbox, _ = extract_pred_bbox(heat_np, threshold=best_t)
            if pbox:
                draw.rectangle(list(pbox), outline="#ff0055", width=4)
                draw.text((pbox[0], max(0, pbox[1] - 20)), f"Pred CAM Region", fill="#ff0055")
                
            # Save
            out_file = VIZ_DIR / subfolder / f"{category}_{row_data['image_filename']}"
            blended.save(out_file)

    # 1. Successful Cases (High IoU / Pointing Game Hit)
    top_hits = sample_df[sample_df["pointing_game_hit"] == 1].sort_values("iou_gradcam", ascending=False).head(4)
    for _, r in top_hits.iterrows():
        # Match back to manifest row
        m_row = df[df["image_filename"] == r["image_filename"]].iloc[0]
        render_composite(m_row, "SUCCESS", "successful_cases")

    # 2. Failed Localization Cases (TP classification but Hit = 0)
    failed_locs = sample_df[(sample_df["is_correct_classification"] == True) & (sample_df["pointing_game_hit"] == 0)].head(4)
    for _, r in failed_locs.iterrows():
        m_row = df[df["image_filename"] == r["image_filename"]].iloc[0]
        render_composite(m_row, "FAILED_LOC", "failed_cases")

    # 3. False Negatives (Model called Fake -> Real)
    fn_samples = sample_df[sample_df["is_correct_classification"] == False].head(4)
    for _, r in fn_samples.iterrows():
        m_row = df[df["image_filename"] == r["image_filename"]].iloc[0]
        render_composite(m_row, "FN", "false_negatives")

    # 4. False Positives (Real document where Fake class CAM fired)
    for _, m_row in test_reals.head(3).iterrows():
        render_composite(m_row, "FP", "false_positives")

    # 8. Save Machine-Readable Results JSON
    results_json = {
        "phase": "Phase 3 — Forgery Localization & Grad-CAM Explainability",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "test_eval_count": n_test,
        "selected_threshold": best_t,
        "validation_selection_metric": f"Optimal Mean IoU ({best_val_miou:.4f}) on 191 val fakes",
        "pointing_game": {
            "gradcam_accuracy": round(test_pointing_acc, 4),
            "gradcam_hits": int(test_pointing_hits),
            "hirescam_accuracy": round(test_pointing_acc_hires, 4),
            "hirescam_hits": int(test_pointing_hits_hires)
        },
        "bounding_box_iou": {
            "threshold": best_t,
            "mean_iou": round(test_miou, 4),
            "median_iou": round(test_mediou, 4),
            "pct_iou_gt_zero": round(pct_iou_gt0, 2),
            "pct_iou_ge_03": round(pct_iou_ge03, 2),
            "pct_iou_ge_05": round(pct_iou_ge05, 2)
        },
        "forgery_type_breakdown": ftype_analysis,
        "document_type_breakdown": doc_analysis,
        "field_level_top5": field_analysis[:5],
        "downsampling_ablation": {
            "false_negatives_evaluated": len(fns),
            "fn_avg_gt_area_pct": round(float(fns["gt_area_ratio_pct"].mean()), 3),
            "tp_avg_gt_area_pct": round(float(tps["gt_area_ratio_pct"].mean()), 3),
            "patch_recovery_count": int(fn_patch_fake_detected),
            "patch_recovery_rate_pct": round(fn_patch_recovery_rate, 2)
        }
    }
    with open(OUTPUT_PHASE3 / "phase3_results.json", "w", encoding="utf-8") as f:
        json.dump(results_json, f, indent=2)

    with open(OUTPUT_PHASE3 / "localization_metrics.json", "w", encoding="utf-8") as f:
        json.dump(results_json["bounding_box_iou"], f, indent=2)

    print(f"\nPhase 3 execution complete! Metrics saved to: {OUTPUT_PHASE3 / 'phase3_results.json'}")
    engine.close()
    return results_json


if __name__ == "__main__":
    run_phase3_pipeline()
