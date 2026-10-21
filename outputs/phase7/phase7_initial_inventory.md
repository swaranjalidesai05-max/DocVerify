# DocVerify — Phase 7 Initial Project Inventory & Status Audit

**Project:** DocVerify — AI-Based Document Verification and Forgery Detection System  
**Audit Stage:** Phase 7 Step 1 (Project Inventory & Baseline Traceability)  
**Date:** October 17, 2026  
**Auditor:** Senior ML, Computer Vision & Security Research Engineer  

---

## 1. Directory Structure & Key Subsystems

```
E:\DocverifyAi\
├── Dataset/                     # Read-only original SIDTD dataset (Images, Annotations)
├── archive/                     # Preserved archives
│   └── phase6_previous_attempt/ # 36 files from initial exploration, safely isolated
├── models/                      # Trained model checkpoints
│   ├── phase2/
│   │   ├── best_model.pth       # Frozen EfficientNet-B0 Global Classifier (Val AUC: 0.9112)
│   │   └── final_model.pth      # Epoch 15 checkpoint
│   └── phase6/
│       └── dedicated_patch_model.pth # Dedicated native-resolution local patch classifier
├── outputs/                     # Research artifacts & deliverables across all phases
│   ├── dataset_audit/           # Phase 1: Audit report, summary, and distribution plots
│   ├── dataset_manifest/        # Master dataset_manifest.csv (2,222 samples, 995 clusters)
│   ├── phase2/                  # Phase 2: Global classification curves, confusion matrix
│   ├── phase3/                  # Phase 3: Grad-CAM, HiRes-CAM, Pointing game, IoU metrics
│   ├── phase4/                  # Phase 4: OCR evaluation, oracle patch verification, fusion
│   ├── phase5/                  # Phase 5: Streamlit integration test runs and sample reports
│   ├── phase6/                  # Phase 6: Frozen test benchmarks, statistical validation, CIs
│   │   ├── DocVerify_Phase6_Handover_Report.pdf
│   │   ├── calibrated_fusion_results.json
│   │   ├── error_analysis.csv & summary.json
│   │   ├── experiment_matrix.csv
│   │   ├── forensic_features_test.csv & val.csv
│   │   ├── frozen_test_manifest.csv (N=320)
│   │   ├── fusion_predictions_test.csv & fusion_weights.json
│   │   ├── leakage_audit.json & oracle_leakage_audit.md
│   │   ├── operational_thresholds.json
│   │   ├── patch_model_metrics.json
│   │   ├── per_template_generalization.csv
│   │   ├── phase6_report.md & phase6_results.json
│   │   ├── preprocessing_ablation.json
│   │   ├── reproduction_guide.md
│   │   ├── robustness_curves.csv & stress_testing.json
│   │   ├── shortcut_results.json
│   │   ├── statistical_validation.json
│   │   └── tiled_inspection_predictions.csv & results.json
│   ├── phase7/                  # Phase 7: Active validation outputs
│   └── project_archive/         # Consolidated historical code & documentation
├── scripts/                     # Executable pipelines and benchmark scripts (31 files)
├── src/                         # Core library modules
│   ├── config.py                # Centralized dynamic path and parameter configuration
│   ├── inference_engine.py      # Production inference orchestrator
│   ├── report_generator.py      # Forensic PDF verification report builder (ReportLab)
│   └── template_layouts.py      # Canonical layout geometry priors for 10 SIDTD templates
├── app.py                       # Streamlit web application frontend
├── requirements.txt             # Project dependencies
└── README.md                    # Project overview documentation
```

---

## 2. Core Source Code Modules

| File | Primary Responsibility | Current Pipeline Version |
|---|---|---|
| `app.py` | Interactive Web Dashboard (Streamlit) | Phase 5 UI / connects to `DocVerifyInferenceEngine` |
| `src/config.py` | Dynamic Path & Parameter Management | Phase 6 Dynamic (`Path(__file__).resolve().parent.parent`) |
| `src/inference_engine.py` | Complete End-to-End Inference Pipeline | Phase 5 Integration Baseline — Needs Phase 6 Alignment |
| `src/report_generator.py` | Professional Forensic PDF Report Builder | Phase 5 Verification Report Generator |
| `src/template_layouts.py` | Canonical Geometry Priors for 10 Templates | Phase 4/5 Canonical layout bounding box definitions |

---

## 3. Active Checkpoint Inventory

1. **Global Macro Classifier:** `models/phase2/best_model.pth`
   - Architecture: EfficientNet-B0
   - Weights: ImageNet-1K pretrained + fine-tuned on SIDTD train split
   - Input: $224 \times 224$ normalized RGB
   - Size: 16.1 MB
2. **Dedicated Local Patch Classifier:** `models/phase6/dedicated_patch_model.pth`
   - Architecture: EfficientNet-B0 (Dedicated micro-texture classifier)
   - Input: $224 \times 224$ native-resolution contextual crops
   - Size: 16.1 MB
3. **Calibrated Fusion Weights:** `outputs/phase6/fusion_weights.json`
   - Architecture: Regularized Logistic Regression + StandardScaler parameters
   - Fit strictly on Validation Split ($N=337$)

---

## 4. Initial Audit Observations & Key Phase 7 Priorities

1. **Production Entry Point Alignment:** `src/inference_engine.py` was originally constructed during Phase 5 and relied on Phase 4 evidence fusion (`p_global, p_patch, avg_ocr_conf, text_mismatch, format_anomaly`). It must be formally audited and upgraded to use the validated **Phase 6 multi-modal fusion architecture** (`p_global`, dedicated `dedicated_patch_model.pth` local patch score, ELA mean, ELA inconsistency, noise residual inconsistency, edge variance, OCR confidence, format anomaly).
2. **Template Identification Robustness:** Ensure fallback handling does not depend on filename hints for arbitrary user uploads.
3. **Absence of Ground Truth Dependency:** Verify that no ground-truth annotations, test labels, or oracle bounding boxes are ever touched during real-time inference.
