# DocVerify — Pre-Remediation Baseline Ledger (Phase 8 Step 1)

**Project:** DocVerify — AI-Based Document Verification and Forgery Detection System  
**Document Purpose:** Permanent Historical Record of Pre-Remediation Benchmarks, Model Provenance, and Phase 7A Audit Findings  
**Date of Freeze:** October 18, 2026  
**Status:** FROZEN & PRESERVED (Historical Record — Will NOT be overwritten)  

---

## 1. Explicit Scientific Declaration

> [!IMPORTANT]
> **HISTORICAL PRESERVATION DECLARATION:**  
> The 91.88% test accuracy result obtained in Phase 6 and finalized in Phase 7 is hereby preserved as a **Historical Pre-Remediation Benchmark** and will not be overwritten or discarded. From this point forward, it is formally designated as:  
> **"Pre-Remediation Phase 6/7 Benchmark"** (or *"Previous Phase 6/7 Benchmark — Pending Remediation"*).  
> All historical checkpoints, experiment logs, manifests, and PDF handover reports are permanently preserved. Phase 8 will establish a new, scientifically defensible benchmark following full methodological remediation.

---

## 2. Chronological Pre-Remediation Performance Evolution

| Milestone / Phase | Date | Methodological Summary | Test Acc | Precision | Recall | F1-Score | ROC-AUC | Missed Forgeries (FN) | False Alarms (FP) | Integrity Status |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Phase 2 Baseline** | Oct 2026 | Global EfficientNet-B0 trained on full-page resized images (256x256 -> 224x224) | 79.69% | 82.91% | 77.51% | 80.12% | 0.9112 | 38 / 169 | 27 / 151 | **Legitimate baseline; untouched** |
| **Phase 4 Ablation** | Oct 2026 | Multi-modal fusion with Global CNN + EasyOCR + Patch crops (2.5x window) | 86.88% | 85.47% | 90.53% | 87.93% | 0.9553 | 16 / 169 | 26 / 151 | **Oracle-assisted ablation (used GT boxes)** |
| **Phase 6 Benchmark** | Oct 2026 | Multi-modal fusion with Canonical Priors, ELA, and Forensic features | 91.88% | 95.54% | 88.76% | 92.02% | 0.9780 | 19 / 169 | 7 / 151 | **Pre-remediation benchmark (compromised by patch bug & feature mismatch)** |
| **Phase 7 Finalization**| Oct 2026 | System integration, enterprise UI, and production inference engine | 91.88% | 95.54% | 88.76% | 92.02% | 0.9780 | 19 / 169 | 7 / 151 | **Active production baseline awaiting Phase 8 remediation** |

---

## 3. Pre-Remediation Checkpoints & File Manifest

### A. Model Checkpoints
- **Phase 2 Global Model:** `models/phase2/best_model.pth` (MD5: Verified sound, frozen).
- **Phase 6 Patch Model:** `models/phase6/dedicated_patch_model.pth` (MD5: Preserved; compromised by bounding box parsing bug).
- **Phase 6 Fusion Weights:** `outputs/phase6/fusion_weights.json` (Preserved; coefficients dominated by `p_global` and `ela_mean`).

### B. Datasets and Split Manifests
- **Dataset Root:** `Dataset/` (SIDTD dataset: 10 templates, 2,222 images).
- **Master Manifest:** `outputs/dataset_manifest/dataset_manifest.csv` (1,565 Train, 337 Validation, 320 Test).
- **Frozen Test Manifest:** `outputs/phase6/frozen_test_manifest.csv` (320 samples: 151 Genuine, 169 Forged; exactly 0 identity overlap with train/val).

### C. Software Environment & Dependencies
- Python: 3.13.x
- PyTorch: 2.x (CPU execution mode)
- Torchvision: 0.20+
- EasyOCR: 1.7.2
- Scikit-Learn: 1.6+
- Pillow (PIL): 11.x
- OpenCV: 4.11+
- Streamlit: 1.42+
- ReportLab: 4.3+
- Random Seeds: 42 (Global deterministic seed)

---

## 4. Phase 7A Confirmed Issues Summary Requiring Phase 8 Remediation

1. **Bounding Box Parsing Bug:** `scripts/train_patch_model.py` checked `isinstance(b, list)` instead of `dict`, resulting in 100% fallback to `[0.35W, 0.45H, 0.25W, 0.08H]`.
2. **Invalid EXP-06 Claim:** Claimed 99.38% accuracy on native sensor-scale GT crops; invalidated by the parsing bug.
3. **Train/Test Fusion Feature Mismatch:** Validation `p_patch` came from Phase 4 oracle GT crops on Phase 2 model; Test `p_patch` came from Phase 6 canonical priors on dedicated patch model.
4. **JPEG / ELA Dataset Shortcut:** Genuine documents are larger (mean 801.9 KB) than synthetic fakes (mean 346.6 KB); `ela_mean` (-3.98) exploits this artifact.
5. **Template Identification Weakness:** Relies on filename substring hints (`alb_id_...`); aspect ratio fallback fails to distinguish ID-1 cards.
6. **Calibration Claim Discrepancy:** Documentation claimed isotonic calibration, but `IsotonicRegression` was never instantiated.
7. **Threshold Inversion:** Strict anti-fraud threshold (0.5495) was lower than conservative KYC threshold (0.6566).
8. **Grad-CAM Geometry Distortion:** Aspect ratio mismatch between 224x224 CenterCrop and 256x256 canvas stretched heatmap coordinates by 20–30%.
9. **Streamlit UI Stale State:** Previous verification results rendered under new document uploads.
10. **Temporary File Leak:** `temp_orig_...png` and `temp_cam_...png` accumulated on disk during PDF generation.

---

## 5. Phase 8 Remediation Objectives

Phase 8 will systematically resolve all 10 confirmed issues above, retrain only the affected components, reconstruct a unified evidence-generation pipeline, freeze a new benchmark on the untouched frozen test set, and document every metric transparently.
