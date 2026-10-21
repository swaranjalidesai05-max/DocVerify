# DocVerify — Phase 8 Code Inventory

**Project:** DocVerify — AI-Based Document Verification and Forgery Detection System  
**Milestone:** Phase 8 Scientific Remediation  
**Date:** October 18, 2026  

---

## 1. Core Source Code Modules (`src/`)

| File Name | Purpose / Role | Inputs | Outputs | Dependencies | Phase Introduced / Modified |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `src/config.py` | Global project configuration, path resolution, and logging setup | None | Directory paths, ImageNet constants, Logger | `pathlib`, `logging` | Phase 1 (Preserved) |
| `src/bbox_parser.py` | Robust parser for SIDTD bounding box JSON structures (dicts and lists), clipping, validation | Raw bbox string/dict, image width/height, template ID | `[x, y, w, h]`, extraction status | `json`, `src.template_layouts` | **Phase 8 (NEW)** |
| `src/template_matcher.py` | Autonomous explainable document template identifier using geometric aspect ratio, scale, and OCR | PIL image, optional OCR text, optional hint | `template_id`, confidence, method, cues | `re`, `PIL`, `src.template_layouts` | **Phase 8 (NEW)** |
| `src/gradcam_corrected.py` | Mathematically exact inverse spatial mapping for Grad-CAM / HiRes-CAM on rectangular documents | Model, target layer, tensor, original dimensions | Heatmap numpy array matching original canvas | `torch`, `cv2`, `numpy` | **Phase 8 (NEW)** |
| `src/template_layouts.py` | Canonical bounding box layout coordinates and display names for 10 document schemas | Template ID | Normalized field bounding boxes, display names | Standard library | Phase 6 (Updated Phase 7/8) |
| `src/report_generator.py` | Audit-grade verification certificate PDF generator | Verification results dictionary, paths | Standalone verification PDF | `reportlab` | Phase 5 (Preserved) |
| `src/inference_engine.py` | Production-grade multi-modal verification inference engine | PIL image, operational policy, template override | Verification result dictionary with scores and heatmaps | `torch`, `PIL`, `joblib`, `src.*` | Phase 5 (Updated Phase 8) |

---

## 2. Machine Learning Checkpoints (`models/`)

| Checkpoint Path | Architecture | Training Provenance | Operational Status | Phase |
| :--- | :--- | :--- | :--- | :---: |
| `models/phase2/best_model.pth` | EfficientNet-B0 (Global Classifier) | Trained on full-page resized images (224x224) on Train split | **ACTIVE & SOUND** | Phase 2 |
| `models/phase6/dedicated_patch_model.pth` | EfficientNet-B0 (Patch Classifier) | Trained on static central window (Bugged fallback) | **PRESERVED HISTORICAL** | Phase 6 |
| `models/phase8/dedicated_patch_model.pth` | EfficientNet-B0 (Patch Classifier) | Trained on true sensor-scale crops of manipulated boxes | **ACTIVE & REMEDIATED** | **Phase 8** |
| `models/phase8/fusion_scaler.joblib` | StandardScaler (Feature Normalizer) | Fit strictly on validation multi-modal features (N=337) | **ACTIVE & REMEDIATED** | **Phase 8** |
| `models/phase8/fusion_model.joblib` | LogisticRegression (L2, C=0.5) | Fit strictly on validation scaled features (N=337) | **ACTIVE & REMEDIATED** | **Phase 8** |
| `models/phase8/fusion_calibrator.joblib` | IsotonicRegression (Probability Calibrator) | Fit strictly on validation predicted probabilities (N=337) | **ACTIVE & REMEDIATED** | **Phase 8** |

---

## 3. Key Pipeline & Evaluation Scripts (`scripts/`)

| Script Name | Purpose | Execution Output | Phase |
| :--- | :--- | :--- | :---: |
| `scripts/train_patch_model_phase8.py` | Retrains dedicated patch model on true manipulated bounding boxes with deepcopy checkpointing | `models/phase8/dedicated_patch_model.pth`, metrics JSON | **Phase 8** |
| `scripts/rebuild_fusion_pipeline_phase8.py` | Extracts standardized features, fits scaler, logistic regression, and isotonic calibrator | `outputs/phase8/fusion/*`, benchmark JSON | **Phase 8** |
| `scripts/audit_jpeg_ela_shortcut.py` | Executes controlled JPEG recompression stress audit across Q95-Q50 to evaluate ELA shortcut | `outputs/phase8/robustness/*` | **Phase 8** |
| `scripts/test_gradcam_geometry.py` | Validates inverse spatial mapping on synthetic rectangular canvases | `outputs/phase8/gradcam/*` | **Phase 8** |
| `scripts/run_phase8_robustness_and_stats.py` | Executes 5,000 bootstrap CIs, McNemar paired test, 10 stress tests, and error analysis | `outputs/phase8/stress_tests/*`, `evaluation/*` | **Phase 8** |
| `scripts/test_phase8.py` | Comprehensive 14-test functional, statistical, and engineering regression test suite | Console log, 14/14 PASS | **Phase 8** |
| `app.py` | Streamlit Enterprise Web Application with fixed session state and temporary file cleanup | Interactive UI at `localhost:8501` | Phase 7 / 8 |
