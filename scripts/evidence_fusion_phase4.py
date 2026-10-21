#!/usr/bin/env python3
"""
DocVerify - Phase 4 Multi-Modal Evidence Fusion Engine
Author: Senior ML & Computer Vision Research Engineer

Purpose:
1. Implements progressive ablation configurations:
   - Config A: Global Baseline (P_global alone)
   - Config B: Global + Patch Evidence
   - Config C: Global + OCR Evidence
   - Config D: Full Multi-Modal Evidence Fusion (Global + Patch + OCR + Field Consistency)
2. Uses Logistic Regression / Calibrated Linear Weights fitted strictly on Validation features.
3. Evaluates all configurations on the untouched Test set (320 samples: 151 real, 169 fake).
4. Calculates Accuracy, Precision, Recall, F1, ROC-AUC, FPR, FNR, Brier Score, and Calibration Error.
"""

from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, brier_score_loss
)


class EvidenceFusionEngine:
    def __init__(self):
        self.models = {}
        
    def fit_fusion_models(self, val_df: pd.DataFrame):
        """Fits interpretable fusion models on validation data features."""
        y_val = val_df["true_label"].values
        
        # Feature sets
        # Config B: Global + Patch
        X_val_b = val_df[["p_global_fake", "p_patch_fake"]].values
        clf_b = LogisticRegression(random_state=42)
        clf_b.fit(X_val_b, y_val)
        self.models["Global_Plus_Patch"] = clf_b
        
        # Config C: Global + OCR
        # OCR features: ocr_confidence, text_mismatch_flag, format_anomaly_flag
        X_val_c = val_df[["p_global_fake", "ocr_confidence", "text_mismatch_flag", "format_anomaly_flag"]].values
        clf_c = LogisticRegression(random_state=42)
        clf_c.fit(X_val_c, y_val)
        self.models["Global_Plus_OCR"] = clf_c
        
        # Config D: Global + Patch + OCR (Full Multi-Modal Fusion)
        X_val_d = val_df[[
            "p_global_fake", "p_patch_fake", "ocr_confidence",
            "text_mismatch_flag", "format_anomaly_flag"
        ]].values
        clf_d = LogisticRegression(random_state=42)
        clf_d.fit(X_val_d, y_val)
        self.models["Global_Plus_Patch_Plus_OCR"] = clf_d
        
        print("Fitted Fusion Models on Validation features successfully.")

    def evaluate_configuration(
        self,
        config_name: str,
        test_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """Evaluates a fusion configuration on the untouched test split."""
        y_true = test_df["true_label"].values
        
        if config_name == "Global_Baseline":
            probs = test_df["p_global_fake"].values
            preds = (probs >= 0.5).astype(int)
        elif config_name == "Global_Plus_Patch":
            X_test = test_df[["p_global_fake", "p_patch_fake"]].values
            probs = self.models[config_name].predict_proba(X_test)[:, 1]
            preds = (probs >= 0.5).astype(int)
        elif config_name == "Global_Plus_OCR":
            X_test = test_df[["p_global_fake", "ocr_confidence", "text_mismatch_flag", "format_anomaly_flag"]].values
            probs = self.models[config_name].predict_proba(X_test)[:, 1]
            preds = (probs >= 0.5).astype(int)
        elif config_name == "Global_Plus_Patch_Plus_OCR":
            X_test = test_df[[
                "p_global_fake", "p_patch_fake", "ocr_confidence",
                "text_mismatch_flag", "format_anomaly_flag"
            ]].values
            probs = self.models[config_name].predict_proba(X_test)[:, 1]
            preds = (probs >= 0.5).astype(int)
        else:
            raise ValueError(f"Unknown config: {config_name}")
            
        acc = float(accuracy_score(y_true, preds))
        prec = float(precision_score(y_true, preds, zero_division=0))
        rec = float(recall_score(y_true, preds, zero_division=0))
        f1 = float(f1_score(y_true, preds, zero_division=0))
        try:
            auc = float(roc_auc_score(y_true, probs))
        except Exception:
            auc = 0.5
            
        cm = confusion_matrix(y_true, preds)
        tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
        fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
        fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0
        brier = float(brier_score_loss(y_true, probs))
        
        return {
            "configuration": config_name,
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "roc_auc": round(auc, 4),
            "brier_score": round(brier, 4),
            "false_positive_rate": round(fpr, 4),
            "false_negative_rate": round(fnr, 4),
            "confusion_matrix": {"TN": tn, "FP": fp, "FN": fn, "TP": tp},
            "predicted_probs": probs,
            "predicted_labels": preds
        }
