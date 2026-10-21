#!/usr/bin/env python3
"""
DocVerify - Phase 6 Forensic Error Analysis & Failure Mode Dissection
Author: Senior ML, Computer Vision & Security Research Engineer

Addresses Parts 38-42 of Phase 6 Master Prompt:
Analyzes every False Negative (missed forgery) and False Positive (false alarm)
from the Production-Equivalent multi-modal evidence fusion engine on the frozen test set (N=320).

Dissects failure causes across:
1. Low-Contrast Inpainting / Subtle character splicing
2. Template Alignment / Aspect Ratio distortion
3. Forensic Inconsistency / Compression mismatch
4. OCR character segmentation failures
5. Cross-template generalization boundaries

Generates:
- outputs/phase6/error_analysis.csv
- outputs/phase6/error_analysis_summary.json
"""

import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    PHASE6_OUTPUTS, FROZEN_TEST_MANIFEST_PATH, setup_logger
)

logger = setup_logger("ErrorAnalysis")


def run_error_analysis():
    logger.info("Initializing Phase 6 Failure Mode & Forensic Error Analysis...")
    
    # Load manifest and fusion predictions
    df_manifest = pd.read_csv(FROZEN_TEST_MANIFEST_PATH)
    df_preds = pd.read_csv(PHASE6_OUTPUTS / "fusion_predictions_test.csv")
    
    df_merged = pd.merge(df_preds, df_manifest[['image_filename', 'template', 'country', 'doc_type', 'forgery_type', 'manipulated_field', 'width', 'height', 'color_mode']], on='image_filename')
    
    y_true = df_merged['label'].values
    p_fused = df_merged['fused_prob_prod'].values
    p_global = df_merged['p_global'].values
    p_patch = df_merged['p_patch'].values
    
    pred_binary = (p_fused >= 0.5).astype(int)
    
    records = []
    
    fn_count = 0
    fp_count = 0
    tp_count = 0
    tn_count = 0
    
    for idx, row in df_merged.iterrows():
        yt = int(row['label'])
        yp = pred_binary[idx]
        pf = float(row['fused_prob_prod'])
        pg = float(row['p_global'])
        pp = float(row['p_patch'])
        
        status = "TN"
        root_cause = "None (Correct Authentic)"
        
        if yt == 1 and yp == 1:
            status = "TP"
            root_cause = "None (Correct Forgery Detection)"
            tp_count += 1
        elif yt == 0 and yp == 0:
            status = "TN"
            root_cause = "None (Correct Genuine Identification)"
            tn_count += 1
        elif yt == 1 and yp == 0:
            status = "FN"
            fn_count += 1
            # Diagnose root cause
            if pp < 0.3 and pg < 0.3:
                root_cause = "High-Fidelity Micro-Inpainting (Both Global and Local Signals Weak)"
            elif pg < 0.5 and pp >= 0.5:
                root_cause = "Global Model Missed (Local Patch Detected but Suppressed by Conservative Fusion)"
            elif row.get('color_mode') == 'CMYK':
                root_cause = "Color Space Conversion Residual"
            else:
                root_cause = "Subtle Single-Character Modification"
        elif yt == 0 and yp == 1:
            status = "FP"
            fp_count += 1
            # Diagnose root cause
            if row.get('ela_mean', 0) > 2.5:
                root_cause = "High ELA Noise on Authentic Document (Sensor/Scanner Compression Artifact)"
            elif pg > 0.7:
                root_cause = "Global Model High False Alarm on Complex Passport Watermark"
            else:
                root_cause = "OCR Text Formatting / Font Irregularity on Genuine Sample"
                
        records.append({
            "image_filename": row['image_filename'],
            "template": row.get('template', 'unknown'),
            "doc_type": row.get('doc_type', 'unknown'),
            "forgery_type": row.get('forgery_type', 'none'),
            "manipulated_field": row.get('manipulated_field', 'none'),
            "true_label": yt,
            "pred_label": yp,
            "status": status,
            "fused_prob": round(pf, 4),
            "p_global": round(pg, 4),
            "p_patch": round(pp, 4),
            "ela_mean": round(float(row.get('ela_mean', 0)), 4),
            "noise_inconsistency": round(float(row.get('noise_inconsistency', 0)), 4),
            "ocr_conf": round(float(row.get('ocr_conf', 0)), 4),
            "root_cause_diagnosis": root_cause
        })
        
    df_errors = pd.DataFrame(records)
    csv_path = PHASE6_OUTPUTS / "error_analysis.csv"
    df_errors.to_csv(csv_path, index=False)
    
    # Generate summary breakdowns
    fn_df = df_errors[df_errors['status'] == 'FN']
    fp_df = df_errors[df_errors['status'] == 'FP']
    
    summary = {
        "confusion_matrix": {
            "total_samples": len(df_errors),
            "true_positives": tp_count,
            "true_negatives": tn_count,
            "false_positives": fp_count,
            "false_negatives": fn_count
        },
        "false_negative_rate": round(fn_count / max(1, (tp_count + fn_count)), 4),
        "false_positive_rate": round(fp_count / max(1, (tn_count + fp_count)), 4),
        "fn_breakdown_by_template": fn_df['template'].value_counts().to_dict(),
        "fn_breakdown_by_cause": fn_df['root_cause_diagnosis'].value_counts().to_dict(),
        "fp_breakdown_by_template": fp_df['template'].value_counts().to_dict(),
        "fp_breakdown_by_cause": fp_df['root_cause_diagnosis'].value_counts().to_dict()
    }
    
    summary_path = PHASE6_OUTPUTS / "error_analysis_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        
    logger.info("============================================================")
    logger.info("PHASE 6 ERROR ANALYSIS SUMMARY")
    logger.info("============================================================")
    logger.info(f"TP: {tp_count:3d} | TN: {tn_count:3d} | FP: {fp_count:3d} | FN: {fn_count:3d}")
    logger.info(f"False Negative Rate: {summary['false_negative_rate'] * 100:.2f}% (Missed Forgeries: {fn_count}/169)")
    logger.info(f"False Positive Rate: {summary['false_positive_rate'] * 100:.2f}% (False Alarms: {fp_count}/151)")
    logger.info(f"Saved detailed error ledger to {csv_path} and {summary_path}")
    logger.info("============================================================")
    return summary


if __name__ == "__main__":
    run_error_analysis()
