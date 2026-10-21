# DocVerify — Final Validated Metric Freeze

**Project:** DocVerify — AI-Based Document Verification and Forgery Detection System  
**Audit Stage:** Phase 7 Step 13 (Final Metric Freeze & Scientific Baseline Integrity)  
**Date:** October 17, 2026  
**Auditor:** Senior ML, Computer Vision & Security Research Engineer  

---

## 1. Frozen Test Set Specifications
- **Manifest:** `outputs/phase6/frozen_test_manifest.csv`
- **Total Test Samples:** $N = 320$
- **Class Balance:** 151 Real Documents (47.2%), 169 Forged Documents (52.8%)
- **Identity Split:** 139 disjoint identity clusters (0 cluster overlap, 0 base document overlap with train or val)
- **Template Coverage:** All 10 national document templates (`alb_id`, `aze_passport`, `esp_id`, `est_id`, `fin_id`, `grc_passport`, `lva_passport`, `rus_internalpassport`, `srb_passport`, `svk_id`)

---

## 2. Definitive Benchmark Metrics

```
========================================================================================================================
CONFIGURATION                               MODE           ACCURACY   PRECISION   RECALL   F1-SCORE   ROC-AUC
========================================================================================================================
Phase 2 Global Baseline (EfficientNet-B0)   Autonomous      79.69%      82.91%    77.51%    80.12%    0.9112
Phase 6 Dedicated Patch Classifier (Oracle) Oracle GT       99.38%     100.00%    98.82%    99.40%    0.9999
Phase 6 Canonical Layout Prior Inspection   Autonomous      74.06%      70.09%    88.76%    78.33%    0.7515
Phase 6 Uniform Sliding Tile Inspection     Autonomous      68.75%      83.50%    50.89%    63.24%    0.8023
------------------------------------------------------------------------------------------------------------------------
Phase 6 Production Evidence Fusion (Default) Autonomous     91.88%      95.54%    88.76%    92.02%    0.9780
Phase 6 Strict Low-False-Alarm Operating Pt Autonomous     92.19%      96.75%    88.17%    92.26%    0.9780
========================================================================================================================
```

### Statistical Significance
- **Bootstrap 95% Confidence Intervals ($B=5{,}000$):**
  - Phase 2 Baseline: Accuracy **[75.31%, 84.06%]** | AUC **[0.8809, 0.9387]**
  - Phase 6 Production Fusion: Accuracy **[88.75%, 94.69%]** | AUC **[0.9627, 0.9897]**
  - *Distributions are strictly non-overlapping.*
- **McNemar's Paired Contingency Test:**
  - $\chi^2 = 32.09, \mathbf{p = 1.47 \times 10^{-8}}$ (Conclusive statistical superiority over Phase 2).

---

## 3. Strict Operating Point Notice

The **92.19% Accuracy** achieved under the **Strict Low-False-Alarm Threshold ($\tau = 0.5495$)** represents an operational point on the validation ROC curve tailored for banking/KYC onboarding (where false rejections of legitimate customers must be minimized). The primary scientifically reported default benchmark remains **91.88% Accuracy** ($\tau = 0.50$).

The **99.38% Accuracy** of the Dedicated Patch Model is an **Oracle Experiment** requiring ground-truth bounding boxes, demonstrating that native-scale character textures carry near-perfect manipulation signals, but is **NOT** the autonomous production accuracy.
