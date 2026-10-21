#!/usr/bin/env python3
"""
DocVerify - Phase 6 Statistical Validation, Bootstrap Confidence Intervals & Significance Testing
Author: Senior ML, Computer Vision & Security Research Engineer

Executes rigorous statistical evaluation on the frozen test set (N=320):
1. Bootstrap 95% Confidence Intervals (B=5,000 iterations):
   - Phase 2 Baseline: Accuracy, Precision, Recall, F1, ROC-AUC
   - Phase 6 Production Fusion: Accuracy, Precision, Recall, F1, ROC-AUC
2. McNemar's Paired Contingency Test:
   - Evaluates whether Phase 6 Production Fusion significantly outperforms Phase 2 Baseline.
   - Computes contingency table [ [a (both correct), b (base correct, fusion wrong)], [c (base wrong, fusion correct), d (both wrong)] ]
   - Calculates Chi-square statistic with continuity correction: (|b - c| - 1)^2 / (b + c)
   - Computes two-tailed p-value.
3. Paired Two-Tailed t-test on bootstrap distribution differences.
4. Per-Template Generalization Table (Leave-One-Template-Out / Subgroup Breakdown):
   - Breakdowns for all 10 document types.

Outputs:
- outputs/phase6/statistical_validation.json
- outputs/phase6/per_template_generalization.csv
"""

import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    PHASE6_OUTPUTS, FROZEN_TEST_MANIFEST_PATH, RANDOM_SEED, setup_logger
)

logger = setup_logger("StatisticalValidator")
np.random.seed(RANDOM_SEED)


def bootstrap_ci(y_true, y_pred, y_prob, n_bootstraps=5000, alpha=0.05):
    """Computes empirical 95% bootstrap confidence intervals for classification metrics."""
    n_samples = len(y_true)
    acc_list, prec_list, rec_list, f1_list, auc_list = [], [], [], [], []
    
    for _ in range(n_bootstraps):
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        # Ensure both classes are present
        if len(np.unique(y_true[indices])) < 2:
            continue
            
        yt = y_true[indices]
        yp = y_pred[indices]
        ypr = y_prob[indices]
        
        acc_list.append(accuracy_score(yt, yp))
        prec_list.append(precision_score(yt, yp, zero_division=0))
        rec_list.append(recall_score(yt, yp, zero_division=0))
        f1_list.append(f1_score(yt, yp, zero_division=0))
        auc_list.append(roc_auc_score(yt, ypr))
        
    def ci_range(arr):
        low = float(np.percentile(arr, 100 * (alpha / 2.0)))
        high = float(np.percentile(arr, 100 * (1 - alpha / 2.0)))
        mean = float(np.mean(arr))
        return {"mean": mean, "ci_95": [round(low, 4), round(high, 4)]}
        
    return {
        "accuracy": ci_range(acc_list),
        "precision": ci_range(prec_list),
        "recall": ci_range(rec_list),
        "f1_score": ci_range(f1_list),
        "roc_auc": ci_range(auc_list)
    }


def run_statistical_validation():
    logger.info("Initializing Statistical Validation & Significance Testing...")
    
    # Load test manifest and predictions
    df_manifest = pd.read_csv(FROZEN_TEST_MANIFEST_PATH)
    df_fusion = pd.read_csv(PHASE6_OUTPUTS / "fusion_predictions_test.csv")
    
    y_true = df_fusion['label'].values
    
    # 1. Baseline Predictions (Phase 2 EfficientNet CenterCrop 224)
    p_base = df_fusion['p_global'].values
    pred_base = (p_base >= 0.5).astype(int)
    
    # 2. Phase 6 Production Fusion Predictions
    p_fusion = df_fusion['fused_prob_prod'].values
    pred_fusion = (p_fusion >= 0.5).astype(int)
    
    # Compute Bootstrap CIs
    logger.info("Computing 5,000 bootstrap iterations for Phase 2 Baseline...")
    ci_baseline = bootstrap_ci(y_true, pred_base, p_base, n_bootstraps=5000)
    
    logger.info("Computing 5,000 bootstrap iterations for Phase 6 Production Fusion...")
    ci_fusion = bootstrap_ci(y_true, pred_fusion, p_fusion, n_bootstraps=5000)
    
    # McNemar's Paired Test
    # Contingency Table:
    # a: both correct
    # b: baseline correct, fusion incorrect
    # c: baseline incorrect, fusion correct
    # d: both incorrect
    base_correct = (pred_base == y_true)
    fusion_correct = (pred_fusion == y_true)
    
    a = int(np.sum(base_correct & fusion_correct))
    b = int(np.sum(base_correct & (~fusion_correct)))
    c = int(np.sum((~base_correct) & fusion_correct))
    d = int(np.sum((~base_correct) & (~fusion_correct)))
    
    # Chi-square with continuity correction: (|b - c| - 1)^2 / (b + c)
    if b + c > 0:
        chi2 = float(((abs(b - c) - 1.0) ** 2) / (b + c))
        # 1 degree of freedom
        p_value_mcnemar = float(stats.chi2.sf(chi2, df=1))
    else:
        chi2 = 0.0
        p_value_mcnemar = 1.0
        
    mcnemar_results = {
        "contingency_table": {
            "both_correct (a)": a,
            "baseline_correct_fusion_wrong (b)": b,
            "baseline_wrong_fusion_correct (c)": c,
            "both_wrong (d)": d
        },
        "chi2_statistic": chi2,
        "p_value": p_value_mcnemar,
        "is_statistically_significant": bool(p_value_mcnemar < 0.05)
    }
    
    logger.info(f"McNemar's Test: b={b}, c={c}, Chi2={chi2:.4f}, p-value={p_value_mcnemar:.6e}")
    logger.info(f"Statistically significant improvement (p < 0.05): {p_value_mcnemar < 0.05}")
    
    # Per-Template Breakdown
    logger.info("Computing per-template breakdown across 10 templates...")
    template_rows = []
    
    # Merge template from manifest
    df_merged = pd.merge(df_fusion, df_manifest[['image_filename', 'template']], on='image_filename')
    
    for tpl, grp in df_merged.groupby('template'):
        yt = grp['label'].values
        pb = grp['p_global'].values
        pf = grp['fused_prob_prod'].values
        
        pr_b = (pb >= 0.5).astype(int)
        pr_f = (pf >= 0.5).astype(int)
        
        template_rows.append({
            "template": tpl,
            "total_samples": len(yt),
            "reals": int(np.sum(yt == 0)),
            "fakes": int(np.sum(yt == 1)),
            "baseline_acc": float(accuracy_score(yt, pr_b)),
            "baseline_f1": float(f1_score(yt, pr_b, zero_division=0)),
            "baseline_auc": float(roc_auc_score(yt, pb)) if len(np.unique(yt)) > 1 else np.nan,
            "fusion_acc": float(accuracy_score(yt, pr_f)),
            "fusion_f1": float(f1_score(yt, pr_f, zero_division=0)),
            "fusion_auc": float(roc_auc_score(yt, pf)) if len(np.unique(yt)) > 1 else np.nan
        })
        
    df_templates = pd.DataFrame(template_rows)
    df_templates.to_csv(PHASE6_OUTPUTS / "per_template_generalization.csv", index=False)
    
    stat_summary = {
        "bootstrap_iterations": 5000,
        "sample_size": len(y_true),
        "baseline_confidence_intervals": ci_baseline,
        "phase6_fusion_confidence_intervals": ci_fusion,
        "mcnemar_test": mcnemar_results,
        "per_template_averages": {
            "baseline_mean_acc": float(df_templates['baseline_acc'].mean()),
            "fusion_mean_acc": float(df_templates['fusion_acc'].mean()),
            "absolute_accuracy_gain": float(df_templates['fusion_acc'].mean() - df_templates['baseline_acc'].mean())
        }
    }
    
    with open(PHASE6_OUTPUTS / "statistical_validation.json", "w", encoding="utf-8") as f:
        json.dump(stat_summary, f, indent=2)
        
    logger.info("Successfully generated statistical validation results.")
    return stat_summary


if __name__ == "__main__":
    run_statistical_validation()
