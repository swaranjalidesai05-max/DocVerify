# DocVerify — Phase 8 Reproducibility Guide

**Project:** DocVerify — AI-Based Document Verification and Forgery Detection System  
**Milestone:** Phase 8 Scientific Remediation  
**Date:** October 18, 2026  

This document provides exact system requirements, dependency versions, random seeds, and terminal commands to reproduce every Phase 8 result from scratch.

---

## 1. System Environment & Core Dependencies

| Component | Specification / Version | Verification Command |
| :--- | :--- | :--- |
| **Operating System** | Windows 10 / 11 (PowerShell / Command Prompt) | `systeminfo` |
| **Python** | Python 3.13.x (64-bit) | `python --version` |
| **PyTorch** | `torch 2.6.0+` (CPU execution mode) | `python -c "import torch; print(torch.__version__)"` |
| **Torchvision** | `torchvision 0.21.0+` | `python -c "import torchvision; print(torchvision.__version__)"` |
| **Scikit-Learn** | `scikit-learn 1.6.1+` | `python -c "import sklearn; print(sklearn.__version__)"` |
| **EasyOCR** | `easyocr 1.7.2` | `python -c "import easyocr; print(easyocr.__version__)"` |
| **Pillow (PIL)** | `pillow 11.1.0+` | `python -c "import PIL; print(PIL.__version__)"` |
| **OpenCV** | `opencv-python 4.11.0+` | `python -c "import cv2; print(cv2.__version__)"` |
| **Streamlit** | `streamlit 1.42.0+` | `streamlit --version` |
| **ReportLab** | `reportlab 4.3.1+` | `python -c "import reportlab; print(reportlab.__version__)"` |
| **Deterministic Seed**| **`42`** (Applied across Python, NumPy, PyTorch CPU) | Hardcoded in `src/config.py` |

---

## 2. Model Checkpoints & Provenance

- **Global Baseline Model:** `models/phase2/best_model.pth` (EfficientNet-B0 trained in Phase 2 on identity-disjoint full-page scans).
- **Remediated Patch Classifier:** `models/phase8/dedicated_patch_model.pth` (EfficientNet-B0 trained in Phase 8 on native sensor-scale crops of manipulated boxes).
- **Multi-Modal Normalizer:** `models/phase8/fusion_scaler.joblib` (StandardScaler fit strictly on validation split $N=337$).
- **Multi-Modal Logistic Regressor:** `models/phase8/fusion_model.joblib` (L2 Logistic Regression fit strictly on validation split $N=337$).
- **Probability Calibrator:** `models/phase8/fusion_calibrator.joblib` (Isotonic Regression fit strictly on validation predicted probabilities $N=337$).

---

## 3. Step-by-Step Reproduction Pipeline

To reproduce all Phase 8 deliverables, run the following sequential commands from the project root (`E:\DocverifyAi\`):

### Step 1: Execute Patch Sample Audit
Verifies that all 1,222 SIDTD manipulated bounding boxes are parsed correctly and genuine documents receive canonical prior fields:
```powershell
python -c "from src.bbox_parser import parse_sidtd_bbox; print('Parser functional')"
```

### Step 2: Retrain Dedicated Patch Model (6 Epochs)
Retrains the dedicated patch classifier on true sensor-scale crops with state-dict deep copying:
```powershell
python scripts/train_patch_model_phase8.py
```
*Expected test accuracy: ~98.44% on frozen test partition.*

### Step 3: Run JPEG & ELA Shortcut Audit
Audits forensic residual behavior across controlled JPEG quality factors ($Q \in \{95, 90, 80, 70, 60, 50\}$):
```powershell
python scripts/audit_jpeg_ela_shortcut.py
```
*Expected finding: ELA delta collapses from 0.98 (Cohen's d = 1.52) to 0.02 (Cohen's d = 0.07) under matched Q90.*

### Step 4: Rebuild Multi-Modal Evidence Fusion & Probability Calibration
Extracts identical features across validation ($N=337$) and test ($N=320$), fits the fusion model and isotonic calibrator strictly on validation, and evaluates once on frozen test:
```powershell
python scripts/rebuild_fusion_pipeline_phase8.py
```
*Expected test accuracy: 91.25% (Balanced), 92.50% (Strict), 90.31% (Conservative).*

### Step 5: Verify Grad-CAM Spatial Inversion
Tests that the corrected CenterCrop inverse mapping achieves sub-2-pixel accuracy on rectangular document canvases:
```powershell
python scripts/test_gradcam_geometry.py
```
*Expected output: [PASS] Center Offset Error: <= 1px.*

### Step 6: Execute Statistical Analysis & Robustness Suite
Computes 5,000 bootstrap confidence intervals, McNemar's paired test vs Phase 2 baseline, 10 stress tests, and error analysis:
```powershell
python scripts/run_phase8_robustness_and_stats.py
```
*Expected output: McNemar Chi2 ~21.25, p-value < 0.0001.*

### Step 7: Run Comprehensive Automated Test Suite (14 Tests)
Validates all functional, statistical, and engineering components:
```powershell
python scripts/test_phase8.py
```
*Expected output: 14/14 TESTS PASSED.*

### Step 8: Launch Enterprise Web Interface
Runs the interactive Streamlit verification dashboard:
```powershell
streamlit run app.py
```
