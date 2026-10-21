# DocVerify: AI-Based Document Verification and Forgery Detection System

DocVerify is a research-grade, multi-modal AI system engineered to detect manipulations, splices, character inpainting, and micro-text tampering in national identity documents (passports, national ID cards, driver licenses) across diverse jurisdictions.

---

## 🌟 System Architecture & Research Innovations

The system resolves the fundamental **micro-forgery downsampling bottleneck** (where downsampling high-resolution identity cards to standard 224×224 CNN resolution collapses single-character manipulations to sub-pixel noise) through multi-modal evidence fusion:

```
                      Uploaded Document Image (JPEG / PNG / PDF Scan)
                                             │
                                             ▼
                             [Step 1: Image Standardization]
                             • Pillow safe RGB conversion (strips CMYK shortcut)
                             • Native dimension caching (W x H)
                                             │
                                             ▼
                          [Step 2: Template Identification]
                          • Geometric aspect ratio matching against 10 canonical templates
                          • Confidence scoring & graceful unknown-template fallback
                                             │
                   ┌─────────────────────────┴─────────────────────────┐
                   ▼                                                   ▼
     [Branch A: Macro Visual Stream]                    [Branch B: Micro Local Stream]
     • Resize (256) -> CenterCrop (224)                 • Canonical layout prior crops (x2.5 window)
     • EfficientNet-B0 Global Classifier                • Dedicated Patch Classifier (Phase 6)
     • Grad-CAM & HiRes-CAM Heatmaps                    • Native sensor scale texture analysis
     • Signal: P_global (forgery prob)                  • Signal: P_patch (mean & max field score)
                   │                                                   │
                   └─────────────────────────┬─────────────────────────┘
                                             │
                   ┌─────────────────────────┴─────────────────────────┐
                   ▼                                                   ▼
     [Branch C: Forensic Anomaly Stream]                [Branch D: Semantic OCR Stream]
     • Error Level Analysis (ELA, Q=90)                 • Contrast enhancement & upscale (h >= 48px)
     • Spatial quadrant inconsistency                   • EasyOCR text extraction
     • Median filter noise residual variance            • Alphanumeric & date syntactic regex audit
     • Laplacian high-pass edge energy                  • Signals: OCR confidence & Format anomaly
                   │                                                   │
                   └─────────────────────────┬─────────────────────────┘
                                             │
                                             ▼
                       [Step 3: Calibrated Multi-Modal Evidence Fusion]
                       • 8-dimensional feature vector:
                         [P_global, P_patch, ELA_mean, ELA_incon, Noise_incon, Edge_var, OCR_conf, Anomaly]
                       • Standardized via Phase 6 validation-derived scaler
                       • Regularized logistic regression combination (w_1 ... w_8)
                                             │
                                             ▼
                          [Step 4: Operational Decision Logic]
                          • Evaluates fused score against calibrated thresholds:
                            - Balanced Mode (tau = 0.5673)
                            - Strict Low-False-Alarm Mode (tau = 0.5495)
                          • Multi-tiered risk mapping:
                            - P_fused >= tau -> HIGH RISK / SUSPICIOUS (Red)
                            - (tau - 0.15) <= P_fused < tau -> MEDIUM RISK / REVIEW (Amber)
                            - P_fused < (tau - 0.15) -> LOW RISK / AUTHENTIC (Green)
                                             │
                                             ▼
                      [Step 5: Explainability & Certificate Generation]
                      • Grad-CAM / HiRes-CAM attention overlay backprojected to native scan
                      • Field-by-field OCR & patch score ledger
                      • ReportLab automated PDF verification certificate generation
```

---

## 📊 Definitive Untouched Test Benchmark (N=320)

Evaluated strictly on an identity-disjoint, out-of-distribution frozen test split ($N=320$: 151 Real, 169 Fake across 139 disjoint clusters):

| Evaluation Metric | Phase 2 Baseline (Global CNN) | Phase 6 Production Fusion (Default tau=0.50) | Phase 6 Strict Operating Point (tau=0.5495) | Statistical Significance (McNemar) |
| :--- | :---: | :---: | :---: | :---: |
| **Accuracy** | 79.69% | **91.88%** | **92.19%** | $\chi^2 = 32.09, \mathbf{p = 1.47 \times 10^{-8}}$ |
| **Precision** | 82.91% | **95.54%** | **96.75%** | Statistically Superior |
| **Recall (Tampering Detection)** | 77.51% | **88.76%** | **88.17%** | Statistically Superior |
| **F1-Score** | 80.12% | **92.02%** | **92.26%** | Statistically Superior |
| **ROC-AUC** | 0.9112 | **0.9780** | **0.9780** | Statistically Superior |
| **95% Bootstrap CI** | [75.31%, 84.06%] | **[88.75%, 94.69%]** | — | Non-Overlapping |
| **Assistance Mode** | Autonomous | **Autonomous (Zero GT)** | **Autonomous (Zero GT)** | Production-Ready |

*Oracle Experiment Note:* Dedicated local patch verification with ground-truth bounding box crops achieves **99.38% Accuracy (0.9999 AUC)**, confirming that native-scale character textures carry near-flawless forgery artifacts, but this oracle configuration is not used in autonomous production.

---

## 🚀 Quick Start Guide

### 1. Requirements & Environment
Ensure Python 3.10+ is installed:
```bash
git clone https://github.com/DocVerify/DocVerify.git
cd DocVerify
pip install -r requirements.txt
```

### 2. Launch the Interactive Web Application
```bash
streamlit run app.py
```
Open `http://localhost:8501` in your browser.

### 3. Run the Automated Test Suite & Regression Checks
```bash
# Run Phase 6 Scientific Validation Regression Suite (10/10 tests)
python scripts/test_phase6.py

# Run Phase 7 Production Pipeline Regression Suite (11/11 tests)
python scripts/test_phase7.py
```

---

## 📁 Repository Structure

```
DocVerifyAi/
├── app.py                             # Interactive Streamlit web dashboard
├── requirements.txt                   # Dependency specifications
├── README.md                          # Project overview and technical documentation
├── models/
│   ├── phase2/best_model.pth          # Frozen EfficientNet-B0 global baseline checkpoint
│   └── phase6/dedicated_patch_model.pth # Dedicated native-resolution local patch classifier
├── src/
│   ├── config.py                      # Centralized dynamic path and parameter configuration
│   ├── inference_engine.py            # Unified end-to-end multi-modal inference pipeline
│   ├── template_layouts.py            # Canonical layout proportions for 10 document types
│   └── report_generator.py            # ReportLab automated PDF certificate compiler
├── scripts/
│   ├── test_phase6.py                 # Phase 6 scientific validation regression suite
│   ├── test_phase7.py                 # Phase 7 production pipeline regression suite
│   ├── train_patch_model.py           # Dedicated local patch model training pipeline
│   ├── evaluate_tiled_inspection.py   # Production-equivalent tiled inspection benchmark
│   ├── extract_forensic_features.py   # Error Level Analysis, noise residual, and edge extractor
│   ├── fuse_evidence.py               # Calibrated multi-modal evidence fusion engine
│   ├── statistical_validation.py      # Bootstrap CIs and McNemar paired hypothesis testing
│   ├── stress_testing.py              # Environmental robustness degradation curves
│   └── error_analysis.py              # Failure mode and forensic error ledger
└── outputs/
    ├── phase6/                        # Phase 6 research benchmarks, matrix, and reports
    └── phase7/                        # Phase 7 production audit and verification certificates
```

---

## ⚖️ Legal & Compliance Disclaimer

DocVerify is an academic research prototype and AI-assisted screening system. It is designed to assist human compliance analysts and document examiners by flagging anomalies and calculating risk scores. It does not constitute a legally binding certification of identity document authenticity.
