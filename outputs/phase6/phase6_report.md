# DocVerify — Phase 6 Research Validation, Robustness, Generalization & Scientific Evaluation Report

**Project:** DocVerify — AI-Based Document Verification and Forgery Detection System  
**Phase:** Phase 6 (Scientific Validation, Shortcut Auditing, Dedicated Patch Verification, Forensic Multi-Modal Evidence Fusion & Statistical Evaluation)  
**Author:** Senior Machine Learning, Computer Vision & Security Research Engineer  
**Date:** October 2026  
**Status:** **PHASE 6 COMPLETE — EXPERIMENTALLY VALIDATED & AUDITED**

---

## 1. Executive Summary

Phase 6 provides the definitive scientific foundation for **DocVerify**, resolving all potential data leakage vulnerabilities, auditing non-semantic shortcut learning, evaluating native-scale contextual patch verification, establishing independent forensic feature extraction (Error Level Analysis, sensor noise residuals, edge energy), formulating calibrated multi-modal evidence fusion, and proving statistical significance through extensive bootstrap resampling ($B=5{,}000$) and paired hypothesis testing ($p < 10^{-7}$).

Crucially, Phase 6 resolves the central scientific dilemma of Phase 4: the distinction between **Oracle-Assisted (ground-truth guided)** evaluations and **Production-Equivalent (fully autonomous, zero-ground-truth)** document verification. Under rigorous, zero-leakage, identity-disjoint conditions on a frozen test set ($N=320$), DocVerify achieves:
- **Baseline Global Classifier (Phase 2):** 79.69% Accuracy | 80.12% F1 | 0.9112 ROC-AUC
- **Autonomous Production Multi-Modal Fusion (Phase 6):** **91.88% Accuracy** | **92.02% F1** | **0.9780 ROC-AUC**
- **Strict Low-False-Alarm Threshold (0.5495):** **92.19% Accuracy** | **96.75% Precision** | **92.26% F1**
- **Statistical Significance (McNemar's Paired Test):** $\chi^2 = 32.09, p = 1.47 \times 10^{-8}$ (Overwhelmingly significant vs Baseline)
- **95% Bootstrap Confidence Interval:** **[88.75%, 94.69%]** Accuracy (vs Baseline [75.31%, 84.06%])

---

## 2. Integrity & Data Leakage Audit (Part 8)

A comprehensive forensic audit of all 2,222 SIDTD document images across training ($N=1{,}565$), validation ($N=337$), and frozen test ($N=320$) splits was conducted using cryptographic (MD5, SHA256) and perceptual hashing (Average, Perceptual, and Difference Hashes).

```
============================================================
DATA LEAKAGE AUDIT FINDINGS
============================================================
Total Documents Audited:               2,222
Identity Disjoint Clusters:            995
Cross-Split Base Document Overlap:     0 (0.00%)
Cross-Split Cluster Overlap:           0 (0.00%)
Cross-Split Cryptographic Collisions:  0 (0.00%)
Perceptual Collisions Across Splits:   0 (0.00%)
------------------------------------------------------------
Status: ZERO DATA LEAKAGE CONFIRMED
============================================================
```

All 320 frozen test images originate from identity clusters completely unobserved during model training and validation parameter estimation.

---

## 3. Shortcut & Bias Audit (Part 9)

To ensure the neural network is learning genuine physical and digital manipulation cues rather than trivial dataset generation artifacts, dedicated non-visual shortcut classifiers were evaluated on the frozen test set:

1. **Metadata-Only Classifier (Width, Height, Aspect Ratio, File Size):**
   - **Accuracy: 100.00% | ROC-AUC: 1.0000**
   - *Scientific Finding:* The raw SIDTD dataset exhibits synthetic file-container artifacts where synthetic generators save tampered images at slightly different byte encodings or canvas boundaries than real scans. If models are given raw file headers or unstandardized canvases, 100% accuracy can be achieved without looking at the document pixels.
2. **Color Mode Classifier (CMYK vs RGB):**
   - **Accuracy: 77.19% | Precision: 69.83% | Recall: 100.00%**
   - *Scientific Finding:* In the raw dataset, 50.3% of Real documents are encoded in CMYK color space, whereas 0.0% of Fake documents are CMYK. An adversary or naive classifier could exploit color space alone.
3. **Mitigation Strategy:**
   - Strict Pillow RGB conversion (`img.convert("RGB")`) strips color space headers.
   - Standardized uniform resizing and cropping discard canvas dimension shortcuts.
   - Patch-scale and pixel-level forensic analyses isolate local sensor noise rather than container metadata.

---

## 4. Preprocessing Geometry & Compression Ablation (Part 10 & 12)

| Preprocessing Pipeline | Input Size | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Failure / Cause |
|---|---|---|---|---|---|---|---|
| **A: CenterCrop 224 (Baseline)** | $224 \times 224$ | **79.69%** | 82.91% | 77.51% | 80.12% | 0.9112 | 38 FN, 27 FP (Phase 2 reference) |
| **B: Letterbox Padding 224** | $224 \times 224$ | **51.56%** | 55.30% | 43.20% | 48.50% | 0.5254 | Severe drop: Gray bars distort CNN features |
| **C: Letterbox Padding 384** | $384 \times 384$ | **60.62%** | 63.83% | 53.25% | 57.99% | 0.6521 | Partial recovery but aspect mismatch remains |
| **D: Standardized JPEG Q=85** | $224 \times 224$ | **73.44%** | 70.19% | 86.39% | 77.45% | 0.7938 | Drops ~6.25% acc; eliminates compression shortcut |

*Key Insight:* Standardizing all documents via uniform JPEG recompression at $Q=85$ removes residual compression shortcuts, dropping single-model global accuracy to 73.44%. This confirms why **multi-modal fusion** is mandatory for enterprise robustness.

---

## 5. Local Patch Verification & Production Tiling (Parts 13–16)

A dedicated local patch classifier (EfficientNet-B0) was trained strictly on native-resolution contextual crops ($2.5\times$ bounding window, $224\times224$ normalized).

### Local Inspection Benchmark (Frozen Test Set, $N=320$)
- **Oracle GT-Guided Patches:** **99.38% Accuracy** | **100.00% Precision** | **98.82% Recall** | **0.9999 ROC-AUC**  
  *Proof:* Microscopic character splicing and font inpainting boundaries possess distinct texture features easily identified when zoomed in at native scanner scale.
- **Production Canonical Priors (Mean Aggregation):** **74.06% Accuracy** | **70.09% Precision** | **88.76% Recall** | **0.7515 ROC-AUC**
- **Production Uniform Sliding Tiles (Mean Aggregation):** **68.75% Accuracy** | **83.50% Precision** | **50.89% Recall** | **0.8023 ROC-AUC**

---

## 6. Multi-Modal Evidence Fusion & Operational Decision Logic (Parts 28–33)

To overcome the limitations of isolated classifiers, DocVerify integrates 5 complementary evidence streams into a regularized logistic fusion engine, calibrated strictly on the validation split ($N=337$):
1. **Global CNN Forgery Probability ($P_{\text{global}}$):** Macro layout and document geometry ($w_1 = +3.066$).
2. **Local Contextual Patch Score ($P_{\text{patch}}$):** Canonical layout prior field inspection ($w_2 = +0.224$).
3. **Forensic Compression Residuals (ELA Mean & Inconsistency):** Compression discontinuity ($w_3 = -3.984, w_4 = -0.144$).
4. **Sensor Noise Inconsistency:** Spatial noise variance irregularity across quadrants ($w_5 = -0.905$).
5. **High-Frequency Edge Energy:** Laplacian sharpness mismatch between characters and background ($w_6 = +1.605$).
6. **Semantic OCR Evidence:** EasyOCR confidence and structural format anomaly flags ($w_7 = -0.423, w_8 = -0.424$).

### Operational KYC Decision Thresholds (Derived from Validation ROC)
- **Balanced Operating Point ($\tau = 0.5673$):** Accuracy 91.88%, Precision 96.73%, Recall 87.57%, F1 91.93%.
- **Conservative Anti-Fraud ($\tau = 0.6566$):** Accuracy 91.88%, Precision 99.31%, Recall 85.21%, F1 91.72%.
- **Strict Low-False-Alarm ($\tau = 0.5495$):** **Accuracy 92.19%**, **Precision 96.75%**, Recall 88.17%, **F1 92.26%**.

---

## 7. Master Experiment Benchmark Matrix (Part 41)

```
+------------+-----------------------+---------------------------------------+-------------+------------------+----------+-----------+--------+----------+---------+
| Exp ID     | Category              | Configuration                         | Input Res   | Assistance Mode  | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
+------------+-----------------------+---------------------------------------+-------------+------------------+----------+-----------+--------+----------+---------+
| EXP-01     | Global Baseline       | EfficientNet-B0 (CenterCrop)          | 224x224     | Autonomous       | 79.69%   | 82.91%    | 77.51% | 80.12%   | 0.9112  |
| EXP-02     | Preproc Ablation      | EfficientNet-B0 (Letterbox)           | 224x224     | Autonomous       | 51.56%   | 55.30%    | 43.20% | 48.50%   | 0.5254  |
| EXP-03     | Preproc Ablation      | EfficientNet-B0 (Standardized JPEG)   | 224x224     | Autonomous       | 73.44%   | 70.19%    | 86.39% | 77.45%   | 0.7938  |
| EXP-04     | Shortcut Audit        | Metadata Only (Size, Dim, Aspect)     | Header      | Non-Visual       | 100.00%  | 100.00%   | 100.0% | 100.00%  | 1.0000  |
| EXP-05     | Shortcut Audit        | Color Mode Only (CMYK vs RGB)         | Header      | Non-Visual       | 77.19%   | 69.83%    | 100.0% | 82.24%   | 0.7583  |
| EXP-06     | Local Patch           | Dedicated Patch Model (Oracle GT)     | Native Crop | Oracle GT Guided | 99.38%   | 100.00%   | 98.82% | 99.40%   | 0.9999  |
| EXP-07     | Local Patch           | Dedicated Patch Model (Canonical)     | Native Crop | Autonomous       | 74.06%   | 70.09%    | 88.76% | 78.33%   | 0.7515  |
| EXP-08     | Local Patch           | Dedicated Patch Model (Sliding Tiles) | 256x256     | Autonomous       | 68.75%   | 83.50%    | 50.89% | 63.24%   | 0.8023  |
| EXP-09     | Evidence Fusion       | Full Fusion (Oracle-Assisted Patch)   | Multi-Scale | Oracle GT Guided | 91.56%   | 94.94%    | 88.76% | 91.74%   | 0.9768  |
| EXP-10     | Evidence Fusion       | Full Production Fusion (Default 0.50) | Multi-Scale | Autonomous       | 91.88%   | 95.54%    | 88.76% | 92.02%   | 0.9780  |
| EXP-11     | Operational Decision  | Full Production Fusion (Strict 0.549) | Multi-Scale | Autonomous       | 92.19%   | 96.75%    | 88.17% | 92.26%   | 0.9780  |
+------------+-----------------------+---------------------------------------+-------------+------------------+----------+-----------+--------+----------+---------+
```

---

## 8. Statistical Validation & Hypothesis Testing (Part 34)

- **Bootstrap Confidence Intervals ($B = 5{,}000$):**
  - Phase 2 Baseline Accuracy: **79.67%** (95% CI: `[75.31%, 84.06%]`) | AUC: **0.9112** (`[0.8809, 0.9387]`)
  - Phase 6 Production Fusion Accuracy: **91.90%** (95% CI: `[88.75%, 94.69%]`) | AUC: **0.9781** (`[0.9627, 0.9897]`)
  - *Finding:* The 95% confidence intervals are completely non-overlapping, establishing unambiguous performance superiority.
- **McNemar's Paired Contingency Test:**
  - Both Correct: 252 | Baseline Correct, Fusion Wrong: 3 | Baseline Wrong, Fusion Correct: 42 | Both Wrong: 23
  - **$\chi^2 = 32.09$**
  - **$p\text{-value} = 1.47 \times 10^{-8}$**
  - *Conclusion:* Statistically significant improvement over Phase 2 baseline ($p \ll 0.001$).

---

## 9. Environmental Stress Testing & Degradation Curves (Part 37)

| Perturbation Type | Severity / Parameter | Test Accuracy | Test ROC-AUC | System Behavior |
|---|---|---|---|---|
| **JPEG Compression** | $Q=95$ | 75.62% | 0.8304 | Stable baseline |
| | $Q=85$ | 73.44% | 0.7938 | Standardized web compression |
| | $Q=70$ | 61.88% | 0.6195 | Moderate artifact suppression |
| | $Q=50$ | 60.94% | 0.6646 | Noticeable block boundary noise |
| | $Q=30$ | 54.69% | 0.5587 | Severe quantization blocking |
| **Gaussian Blur** | $\sigma=0.5$ | 74.38% | 0.8084 | Minimal degradation |
| | $\sigma=1.0$ | 59.38% | 0.6237 | High-frequency character edge loss |
| | $\sigma=2.0$ | 54.69% | 0.5960 | OCR and micro-textures degraded |
| **Sensor Noise** | $\sigma=5$ | 81.88% | 0.9153 | Robust, slight regularization boost |
| | $\sigma=15$ | 69.69% | 0.7861 | Mild noise resistance |
| | $\sigma=25$ | 65.00% | 0.7362 | High sensor grain |
| **Downscaling** | Scale 0.75 | 69.38% | 0.7791 | Good retention |
| | Scale 0.50 | 59.06% | 0.6452 | Loss of fine security fonts |
| | Scale 0.35 | 56.25% | 0.6161 | Heavy aliasing |

---

## 10. Failure Mode & Error Dissection (Parts 38–40)

- **Total Test Samples:** 320 (151 Genuine, 169 Forged)
- **True Positives (TP):** 150
- **True Negatives (TN):** 144
- **False Positives (FP):** 7 (False Alarm Rate: 4.64%)
- **False Negatives (FN):** 19 (Missed Forgery Rate: 11.24%)

### Root Cause Breakdown of Missed Forgeries (FN = 19):
1. **Subtle Single-Character Modification (10 cases / 52.6%):** Only a single digit in a birth date was altered with matched font weight, resulting in minimal global layout disturbance.
2. **Conservative Fusion Suppression (9 cases / 47.4%):** The dedicated local patch detected local tampering ($P_{\text{patch}} > 0.5$), but the global model registered low probability ($P_{\text{global}} < 0.2$), pulling the calibrated fused probability slightly below the 0.50 decision threshold.

### Root Cause Breakdown of False Alarms (FP = 7):
1. **OCR / Font Irregularities on Genuine Documents (7 cases / 100%):** Complex guilloche backgrounds and passport watermarks (e.g. `srb_passport`, `rus_internalpassport`) induced OCR character recognition confusions, triggering heuristic format anomaly flags on authentic samples.

---

## 11. Reproducibility & Phase 6 Conclusion

Phase 6 provides complete scientific reproducibility. Every result is generated through standalone scripts, documented in structured JSON outputs, and verifiable without external dependencies:
- Training Pipeline: `python scripts/train_patch_model.py`
- Tiled Benchmark: `python scripts/evaluate_tiled_inspection.py`
- Forensic Features: `python scripts/extract_forensic_features.py`
- Fusion Engine: `python scripts/fuse_evidence.py`
- Statistical Validation: `python scripts/statistical_validation.py`
- Stress Testing: `python scripts/stress_testing.py`
- Error Ledger: `python scripts/error_analysis.py`
- Benchmark Matrix: `python scripts/consolidate_phase6.py`

**Final Assessment:** The DocVerify multi-modal architecture demonstrates superior, statistically verified verification capability ($91.88\%$ accuracy, $0.9780$ AUC), successfully bridging the gap between theoretical research and autonomous production deployment.
