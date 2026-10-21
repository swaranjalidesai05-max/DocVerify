#!/usr/bin/env python3
"""
DocVerify - Phase 4 Master Pipeline
Author: Senior ML & Computer Vision Research Engineer

Orchestrates:
1. Loads manifest and models:
   - Manifest: outputs/dataset_manifest/dataset_manifest.csv
   - Frozen Classifier: models/phase2/best_model.pth (EfficientNet-B0)
   - Real VIA templates (Annotations/reals/) for reference text & field layouts
2. Extracts candidate field crops and runs DocumentOCREngine:
   - Measures CER, WER, and Exact Match on real test documents.
   - Measures consistency, format validity, and character length differences on fake test documents.
3. Runs PatchVerificationEngine:
   - High-resolution localized crop around candidate fields.
   - Evaluates P_patch(FAKE) vs P_global(FAKE).
   - Re-evaluates False Negative recovery rate.
4. Generates structured Multi-Modal Evidence Features on Validation & Test sets.
5. Fits EvidenceFusionEngine on Validation features and benchmarks all ablation configurations on untouched Test set:
   - Config A: Global Baseline
   - Config B: Global + Patch
   - Config C: Global + OCR
   - Config D: Global + Patch + OCR (Full Multi-Modal Fusion)
6. Outputs:
   - outputs/phase4/evidence/per_sample_evidence.csv
   - outputs/phase4/phase4_results.json, phase4_config.json
   - Granular breakdown CSVs (by_forgery_type, by_document_type, by_field)
   - Publication-quality visualization plots
"""

import json
import time
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torchvision.models as models
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, roc_curve

from ocr_phase4 import DocumentOCREngine
from patch_verification_phase4 import PatchVerificationEngine
from evidence_fusion_phase4 import EvidenceFusionEngine


def run_phase4():
    BASE_DIR = Path("E:/DocverifyAi")
    MANIFEST_PATH = BASE_DIR / "outputs" / "dataset_manifest" / "dataset_manifest.csv"
    CHECKPOINT_PATH = BASE_DIR / "models" / "phase2" / "best_model.pth"
    OUTPUT_PHASE4 = BASE_DIR / "outputs" / "phase4"
    OCR_DIR = OUTPUT_PHASE4 / "ocr"
    PATCH_DIR = OUTPUT_PHASE4 / "patch_analysis"
    EVID_DIR = OUTPUT_PHASE4 / "evidence"
    FUSION_DIR = OUTPUT_PHASE4 / "fusion"
    ANALYSIS_DIR = OUTPUT_PHASE4 / "analysis"
    VIZ_DIR = OUTPUT_PHASE4 / "visualizations"
    
    print("="*65)
    print("DOCVERIFY — PHASE 4: OCR, HIGH-RES PATCH & EVIDENCE FUSION")
    print("="*65)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[1/6] Loading frozen Phase 2 model: {CHECKPOINT_PATH} on {device}")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    ocr_engine = DocumentOCREngine(languages=['en'], gpu=False)
    patch_engine = PatchVerificationEngine(model, device)
    
    df = pd.read_csv(MANIFEST_PATH)
    val_df = df[df["split"] == "val"].copy()
    test_df = df[df["split"] == "test"].copy()
    
    print(f"Loaded dataset: Val={len(val_df)} (Reals: {(val_df['label']==0).sum()}, Fakes: {(val_df['label']==1).sum()}) | "
          f"Test={len(test_df)} (Reals: {(test_df['label']==0).sum()}, Fakes: {(test_df['label']==1).sum()})")
    
    # Load all VIA template annotations to enable reference-text comparison and field layout
    via_templates = {}
    via_dir = BASE_DIR / "Dataset" / "template" / "Annotations" / "reals"
    for p in via_dir.glob("*.json"):
        tname = p.stem
        with open(p, "r", encoding="utf-8", errors="replace") as f:
            data = json.load(f)
        via_templates[tname] = {}
        for entry in data.get("_via_img_metadata", {}).values():
            fn = entry.get("filename") # e.g. 00.jpg
            via_templates[tname][fn] = entry.get("regions", [])

    # Record Phase 4 Config
    phase4_config = {
        "phase": "Phase 4 — OCR, Patch Verification & Evidence Fusion",
        "frozen_model_checkpoint": str(CHECKPOINT_PATH),
        "ocr_engine": "EasyOCR v1.7.2 (English)",
        "patch_context_multiplier": 2.5,
        "patch_min_dim": 128,
        "fusion_classifier": "LogisticRegression (L2, C=1.0)",
        "ablation_configurations": [
            "Global_Baseline",
            "Global_Plus_Patch",
            "Global_Plus_OCR",
            "Global_Plus_Patch_Plus_OCR"
        ],
        "device": str(device)
    }
    with open(OUTPUT_PHASE4 / "phase4_config.json", "w", encoding="utf-8") as f:
        json.dump(phase4_config, f, indent=2)

    # ----------------------------------------------------
    # Helper to extract evidence row for a single sample
    # ----------------------------------------------------
    def process_sample(row):
        img_p = Path(row["absolute_path"])
        label = int(row["label"])
        template = row["template"] # e.g. alb_id
        
        # 1. Global Prediction
        with Image.open(img_p) as im:
            orig_w, orig_h = im.size
            im_rgb = im.convert("RGB")
            
            # Global tensor
            tensor = patch_engine.eval_transform(im_rgb).unsqueeze(0).to(device)
            with torch.no_grad():
                g_logits = model(tensor)
                g_probs = torch.softmax(g_logits, dim=1).cpu().numpy()[0]
            p_global_real = float(g_probs[0])
            p_global_fake = float(g_probs[1])
            g_pred = int(np.argmax(g_probs))
            
            # 2. Field Bounding Box & Reference Text Retrieval
            gt_bbox = None
            ref_text = ""
            field_name = "document_body"
            
            if label == 1 and row["bounding_box_json"] != "None":
                # Fake sample has explicit manipulated field bbox
                gt_bbox = json.loads(row["bounding_box_json"])
                field_name = row["manipulated_field"]
                
                # Fetch reference text from source real template
                # src e.g. alb_id_00.jpg -> tname=alb_id, idx=00.jpg
                src_fn = row["base_document_id"] + ".jpg"
                idx_fn = src_fn.split("_")[-1]
                tname = "_".join(src_fn.split("_")[:-1])
                if tname in via_templates and idx_fn in via_templates[tname]:
                    for reg in via_templates[tname][idx_fn]:
                        if reg.get("region_attributes", {}).get("field_name") == field_name:
                            ref_text = reg.get("region_attributes", {}).get("value", "")
                            break
            else:
                # Real sample: select a prominent annotated field from its VIA template
                fn = row["image_filename"] # e.g. alb_id_11.jpg
                idx_fn = fn.split("_")[-1]
                tname = "_".join(fn.split("_")[:-1])
                if tname in via_templates and idx_fn in via_templates[tname]:
                    regions = via_templates[tname][idx_fn]
                    # Prefer name or number or date if available
                    chosen_reg = None
                    for r in regions:
                        f_curr = r.get("region_attributes", {}).get("field_name", "")
                        if f_curr in ["name", "surname", "number", "birth_date", "expiry_date"]:
                            chosen_reg = r
                            break
                    if chosen_reg is None and regions:
                        chosen_reg = regions[0]
                    if chosen_reg:
                        gt_bbox = chosen_reg.get("shape_attributes")
                        field_name = chosen_reg.get("region_attributes", {}).get("field_name", "field")
                        ref_text = chosen_reg.get("region_attributes", {}).get("value", "")
                        
            # If no bbox found, default to center crop
            if gt_bbox is None:
                gt_bbox = {"x": int(orig_w * 0.25), "y": int(orig_h * 0.25), "width": int(orig_w * 0.5), "height": int(orig_h * 0.2)}
                
            # 3. Patch Verification
            patch_crop, crop_coords = patch_engine.extract_patch(im_rgb, gt_bbox)
            patch_res = patch_engine.evaluate_patch(patch_crop)
            p_patch_fake = patch_res["patch_prob_fake"]
            patch_pred = patch_res["patch_pred_class"]
            
            # 4. OCR Extraction
            # Tight crop for OCR
            gx, gy = max(0, gt_bbox["x"]), max(0, gt_bbox["y"])
            gw, gh = max(10, gt_bbox["width"]), max(10, gt_bbox["height"])
            ocr_crop = im_rgb.crop((gx, gy, min(orig_w, gx + gw), min(orig_h, gy + gh)))
            ocr_res = ocr_engine.extract_text(ocr_crop)
            
            # 5. Consistency Features
            consistency = ocr_engine.check_field_consistency(field_name, ocr_res["normalized_text"])
            
            # Compute CER/WER if ref_text exists
            cer, wer, exact = 1.0, 1.0, 0
            if ref_text:
                cer, wer, exact = ocr_engine.compute_cer_wer(ocr_res["normalized_text"], ref_text)
                
            # Text mismatch flag (1 if non-empty ref_text differs substantially from OCR)
            text_mismatch_flag = 1 if (ref_text and exact == 0) else 0
            format_anomaly_flag = 1 if (consistency["is_date_field"] and not consistency["valid_date_format"]) else 0
            
            return {
                "image_filename": row["image_filename"],
                "split": row["split"],
                "true_label": label,
                "label_name": row["label_name"],
                "template": row["template"],
                "country": row["country"],
                "forgery_type": row["forgery_type"],
                "field_name": field_name,
                "p_global_fake": round(p_global_fake, 4),
                "global_pred": g_pred,
                "p_patch_fake": round(p_patch_fake, 4),
                "patch_pred": patch_pred,
                "raw_ocr_text": ocr_res["raw_text"],
                "normalized_ocr_text": ocr_res["normalized_text"],
                "ocr_confidence": round(ocr_res["confidence"], 4),
                "reference_text": ref_text,
                "cer": round(cer, 4),
                "wer": round(wer, 4),
                "exact_match": exact,
                "text_mismatch_flag": text_mismatch_flag,
                "format_anomaly_flag": format_anomaly_flag,
                "crop_coords": list(crop_coords)
            }

    # ----------------------------------------------------
    # Step 2: Extract Evidence on Validation and Test Sets
    # ----------------------------------------------------
    evidence_csv_path = EVID_DIR / "per_sample_evidence.csv"
    if evidence_csv_path.exists():
        print(f"\n[2/6] Loading existing pre-computed evidence table: {evidence_csv_path}")
        all_evidence = pd.read_csv(evidence_csv_path)
        val_feat_df = all_evidence[all_evidence["split"] == "val"].copy()
        test_feat_df = all_evidence[all_evidence["split"] == "test"].copy()
    else:
        print("\n[2/6] Extracting multi-modal evidence features on VALIDATION set (337 samples)...")
        val_records = []
        for _, row in tqdm(val_df.iterrows(), total=len(val_df), desc="Val Features"):
            val_records.append(process_sample(row))
        val_feat_df = pd.DataFrame(val_records)
        
        print("\n[3/6] Extracting multi-modal evidence features on TEST set (320 samples)...")
        test_records = []
        for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Test Features"):
            test_records.append(process_sample(row))
        test_feat_df = pd.DataFrame(test_records)
        
        all_evidence = pd.concat([val_feat_df, test_feat_df], ignore_index=True)
        all_evidence.to_csv(evidence_csv_path, index=False)
        print(f"Per-sample evidence saved to: {evidence_csv_path}")

    # ----------------------------------------------------
    # Step 3: Real Document OCR Quality Evaluation
    # ----------------------------------------------------
    print("\n[4/6] Evaluating Real-Document OCR Quality metrics...")
    real_test_ocr = test_feat_df[(test_feat_df["true_label"] == 0) & (test_feat_df["reference_text"] != "")]
    avg_cer = float(real_test_ocr["cer"].mean())
    avg_wer = float(real_test_ocr["wer"].mean())
    exact_acc = float(real_test_ocr["exact_match"].mean()) * 100
    avg_conf_real = float(real_test_ocr["ocr_confidence"].mean())
    
    print(f"Real Test Documents Evaluated: {len(real_test_ocr)}")
    print(f"Character Error Rate (CER): {avg_cer:.4f}")
    print(f"Word Error Rate (WER):      {avg_wer:.4f}")
    print(f"Exact Text Match Accuracy:  {exact_acc:.2f}%")
    print(f"Average OCR Confidence:     {avg_conf_real:.4f}")

    ocr_summary = {
        "engine": "EasyOCR v1.7.2",
        "real_test_samples_evaluated": len(real_test_ocr),
        "mean_cer": round(avg_cer, 4),
        "mean_wer": round(avg_wer, 4),
        "exact_match_accuracy_pct": round(exact_acc, 2),
        "mean_confidence_real": round(avg_conf_real, 4)
    }
    with open(OCR_DIR / "ocr_results.json", "w", encoding="utf-8") as f:
        json.dump(ocr_summary, f, indent=2)

    # ----------------------------------------------------
    # Step 4: Patch Verification & False Negative Recovery Analysis
    # ----------------------------------------------------
    print("\n[5/6] Benchmarking High-Resolution Patch Verification & False Negative Recovery...")
    test_fakes = test_feat_df[test_feat_df["true_label"] == 1]
    
    # Global False Negatives in Phase 2
    global_fns = test_fakes[test_fakes["global_pred"] == 0]
    recovered_by_patch = global_fns[global_fns["patch_pred"] == 1]
    fn_recovery_rate = (len(recovered_by_patch) / len(global_fns)) * 100 if len(global_fns) > 0 else 0.0
    
    print(f"Total Test Fakes: {len(test_fakes)}")
    print(f"Global False Negatives (Missed by EfficientNet-B0): {len(global_fns)} / {len(test_fakes)}")
    print(f"Recovered by High-Resolution Patch: {len(recovered_by_patch)} / {len(global_fns)} ({fn_recovery_rate:.2f}%)")
    
    # ----------------------------------------------------
    # Step 5: Multi-Modal Evidence Fusion & Ablation Benchmark
    # ----------------------------------------------------
    print("\n[6/6] Fitting and Benchmarking Evidence Fusion Ablations on TEST split...")
    fusion_engine = EvidenceFusionEngine()
    fusion_engine.fit_fusion_models(val_feat_df)
    
    ablation_results = []
    configs = [
        "Global_Baseline",
        "Global_Plus_Patch",
        "Global_Plus_OCR",
        "Global_Plus_Patch_Plus_OCR"
    ]
    
    for cfg in configs:
        res = fusion_engine.evaluate_configuration(cfg, test_feat_df)
        ablation_results.append(res)
        cm = res["confusion_matrix"]
        print(f"Config [{res['configuration']:26s}] -> Acc: {res['accuracy']:.4f} | Prec: {res['precision']:.4f} | "
              f"Rec: {res['recall']:.4f} | F1: {res['f1_score']:.4f} | AUC: {res['roc_auc']:.4f} | "
              f"CM: [TN={cm['TN']}, FP={cm['FP']}, FN={cm['FN']}, TP={cm['TP']}]")

    ablation_df = pd.DataFrame([
        {
            "System": r["configuration"],
            "Accuracy": r["accuracy"],
            "Precision": r["precision"],
            "Recall": r["recall"],
            "F1-Score": r["f1_score"],
            "ROC-AUC": r["roc_auc"],
            "Brier Score": r["brier_score"],
            "TN": r["confusion_matrix"]["TN"],
            "FP": r["confusion_matrix"]["FP"],
            "FN": r["confusion_matrix"]["FN"],
            "TP": r["confusion_matrix"]["TP"]
        } for r in ablation_results
    ])
    ablation_df.to_csv(FUSION_DIR / "ablation_benchmark.csv", index=False)
    
    # False Negative Tracking Table
    fn_tracking = {
        "total_global_fns": len(global_fns),
        "recovered_by_patch": len(recovered_by_patch),
        "patch_recovery_rate_pct": round(fn_recovery_rate, 2),
        "recovered_by_full_fusion": int(len(global_fns) - ablation_results[-1]["confusion_matrix"]["FN"]),
        "full_fusion_fn_count": ablation_results[-1]["confusion_matrix"]["FN"],
        "fn_reduction_count": int(ablation_results[0]["confusion_matrix"]["FN"] - ablation_results[-1]["confusion_matrix"]["FN"])
    }

    # Granular Breakdowns on Best Fusion Model
    best_preds = ablation_results[-1]["predicted_labels"]
    best_probs = ablation_results[-1]["predicted_probs"]
    test_feat_df["fusion_pred"] = best_preds
    test_feat_df["fusion_prob"] = best_probs
    test_feat_df["fusion_correct"] = (test_feat_df["fusion_pred"] == test_feat_df["true_label"])
    
    # By Forgery Type
    ftype_breakdown = []
    for ftype, grp in test_feat_df[test_feat_df["true_label"] == 1].groupby("forgery_type"):
        ftype_breakdown.append({
            "forgery_type": ftype,
            "samples": len(grp),
            "global_recall": round(float((grp["global_pred"] == 1).mean()), 4),
            "patch_recall": round(float((grp["patch_pred"] == 1).mean()), 4),
            "fusion_recall": round(float((grp["fusion_pred"] == 1).mean()), 4)
        })
    pd.DataFrame(ftype_breakdown).to_csv(ANALYSIS_DIR / "by_forgery_type.csv", index=False)
    
    # By Document Template
    doc_breakdown = []
    for tpl, grp in test_feat_df.groupby("template"):
        doc_breakdown.append({
            "template": tpl,
            "samples": len(grp),
            "global_accuracy": round(float((grp["global_pred"] == grp["true_label"]).mean()), 4),
            "fusion_accuracy": round(float((grp["fusion_pred"] == grp["true_label"]).mean()), 4)
        })
    pd.DataFrame(doc_breakdown).to_csv(ANALYSIS_DIR / "by_document_type.csv", index=False)

    # ----------------------------------------------------
    # Generate Visualizations
    # ----------------------------------------------------
    print("\nGenerating Phase 4 publication plots...")
    sns.set_theme(style="whitegrid")
    
    # 1. Ablation ROC Curves
    plt.figure(figsize=(8, 6))
    from sklearn.metrics import roc_curve
    for r in ablation_results:
        fpr, tpr, _ = roc_curve(test_feat_df["true_label"], r["predicted_probs"])
        plt.plot(fpr, tpr, lw=2, label=f"{r['configuration']} (AUC = {r['roc_auc']:.4f})")
    plt.plot([0, 1], [0, 1], color="gray", lw=1.5, linestyle="--")
    plt.title("Phase 4 Evidence Fusion ROC Curves (Test Split N=320)", fontsize=13, fontweight="bold")
    plt.xlabel("False Positive Rate", fontsize=11)
    plt.ylabel("True Positive Rate (Recall)", fontsize=11)
    plt.legend(loc="lower right", fontsize=9.5)
    plt.tight_layout()
    plt.savefig(VIZ_DIR / "roc_curve_ablations.png", dpi=200)
    plt.close()

    # 2. Confusion Matrix of Best Fusion System
    plt.figure(figsize=(7, 6))
    best_cm = confusion_matrix(test_feat_df["true_label"], best_preds)
    sns.heatmap(best_cm, annot=True, fmt="d", cmap="Greens", cbar=False,
                xticklabels=["Pred Real (0)", "Pred Fake (1)"],
                yticklabels=["Actual Real (0)", "Actual Fake (1)"],
                annot_kws={"size": 14, "weight": "bold"})
    plt.title(f"Full Fusion Confusion Matrix (Acc: {ablation_results[-1]['accuracy']:.4f})", fontsize=13, fontweight="bold")
    plt.ylabel("Ground Truth", fontsize=12)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.tight_layout()
    plt.savefig(VIZ_DIR / "confusion_matrix_fusion.png", dpi=200)
    plt.close()

    # 3. False Negative Reduction Comparison Bar Chart
    plt.figure(figsize=(7, 4.5))
    systems = [r["configuration"].replace("_", " ") for r in ablation_results]
    fn_counts = [r["confusion_matrix"]["FN"] for r in ablation_results]
    bars = plt.bar(systems, fn_counts, color=["#e74c3c", "#e67e22", "#3498db", "#2ecc71"], width=0.5)
    plt.title("False Negative Reduction across Ablation Configurations", fontsize=12, fontweight="bold")
    plt.ylabel("Missed Forgeries (FN Count)", fontsize=11)
    plt.xticks(rotation=15, ha="right", fontsize=9.5)
    for b in bars:
        h = b.get_height()
        plt.text(b.get_x() + b.get_width() / 2.0, h + 0.8, f"{int(h)}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    plt.ylim(0, 45)
    plt.tight_layout()
    plt.savefig(VIZ_DIR / "false_negative_reduction.png", dpi=200)
    plt.close()

    # Save Machine-Readable Results JSON
    phase4_results = {
        "phase": "Phase 4 — OCR, High-Resolution Patch & Evidence Fusion",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "test_eval_count": len(test_df),
        "real_count": int((test_df["label"] == 0).sum()),
        "fake_count": int((test_df["label"] == 1).sum()),
        "ocr_evaluation": ocr_summary,
        "patch_verification": {
            "total_test_fakes": len(test_fakes),
            "global_fns_count": len(global_fns),
            "patch_recovered_fns_count": len(recovered_by_patch),
            "fn_recovery_rate_pct": round(fn_recovery_rate, 2)
        },
        "ablation_benchmark": [
            {
                "configuration": r["configuration"],
                "accuracy": r["accuracy"],
                "precision": r["precision"],
                "recall": r["recall"],
                "f1_score": r["f1_score"],
                "roc_auc": r["roc_auc"],
                "brier_score": r["brier_score"],
                "confusion_matrix": r["confusion_matrix"]
            } for r in ablation_results
        ],
        "false_negative_tracking": fn_tracking,
        "forgery_type_breakdown": ftype_breakdown,
        "document_type_breakdown": doc_breakdown
    }
    with open(OUTPUT_PHASE4 / "phase4_results.json", "w", encoding="utf-8") as f:
        json.dump(phase4_results, f, indent=2)

    print(f"\nPhase 4 Master Pipeline finished successfully! Deliverables saved to: {OUTPUT_PHASE4}")
    return phase4_results


if __name__ == "__main__":
    run_phase4()
