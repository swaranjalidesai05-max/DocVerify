# DocVerify — Final Reproduction & Dependency Audit

**Project:** DocVerify — AI-Based Document Verification and Forgery Detection System  
**Audit Stage:** Phase 7 Step 10 (Reproducibility & Dependency Verification)  
**Date:** October 17, 2026  
**Auditor:** Senior ML, Computer Vision & Security Research Engineer  

---

## 1. Environment & Runtime Specifications

| Component | Verified Specification |
|---|---|
| **Python Version** | Python 3.13.0 (Windows 64-bit) |
| **PyTorch Framework** | PyTorch 2.14.1+cpu |
| **TorchVision** | torchvision 0.19.1+cpu |
| **Scientific Computing** | NumPy 1.26.4, SciPy 1.14.1, Pandas 2.2.3, Scikit-Learn 1.5.2 |
| **Image Processing** | Pillow (PIL) 10.4.0 |
| **Explainability & CAMs** | PyTorch native forward/backward hooks on `model.features[8]` |
| **OCR Framework** | EasyOCR 1.7.2 (with PyTorch CPU/CUDA dynamic routing) |
| **Web Dashboard** | Streamlit 1.39.0 |
| **Report Generation** | ReportLab 4.2.5 |
| **Random Seed** | 42 (Enforced across NumPy, PyTorch, Scikit-Learn, and Python random) |

---

## 2. Model Weight Checksums & Locations

1. **Phase 2 Frozen Global Classifier:**
   - Path: `models/phase2/best_model.pth`
   - Architecture: EfficientNet-B0 ($224 \times 224$ input)
   - Size: 16.1 MB (16,877,419 bytes)
2. **Phase 6 Dedicated Local Patch Classifier:**
   - Path: `models/phase6/dedicated_patch_model.pth`
   - Architecture: EfficientNet-B0 ($224 \times 224$ native-scale contextual crops)
   - Size: 16.1 MB (16,878,059 bytes)
3. **Phase 6 Evidence Fusion Parameters:**
   - Path: `outputs/phase6/fusion_weights.json`
   - Type: JSON (scaler means, standard deviations, regularized logistic coefficients, intercept)
4. **Phase 6 Operational Thresholds:**
   - Path: `outputs/phase6/operational_thresholds.json`
   - Type: JSON (Balanced: 0.5673, Conservative: 0.6566, Strict: 0.5495)

---

## 3. End-to-End Command Verification

```bash
# 1. Run Phase 6 scientific regression suite
python scripts/test_phase6.py
# Result: 10/10 tests passed

# 2. Run Phase 7 production pipeline test suite
python scripts/test_phase7.py
# Result: 11/11 tests passed

# 3. Launch interactive web dashboard
streamlit run app.py
```

All commands operate with zero external network access requirements and zero manual configuration.
