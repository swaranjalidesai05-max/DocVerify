# DocVerify — Final Production System Architecture

**Project:** DocVerify — AI-Based Document Verification and Forgery Detection System  
**Audit Stage:** Phase 7 Step 12 (Final Production Architecture Document)  
**Date:** October 17, 2026  
**Auditor:** Senior ML, Computer Vision & Security Research Engineer  

---

## 1. End-to-End Information Flow

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

## 2. Mathematical Formulation of Multi-Modal Fusion

Given input document $D$, the multi-modal evidence feature vector $\mathbf{x} \in \mathbb{R}^8$ is constructed:
$$\mathbf{x} = \begin{bmatrix} P_{\text{global}} \\ P_{\text{patch}} \\ \text{ELA}_{\text{mean}} \\ \text{ELA}_{\text{inconsistency}} \\ \text{Noise}_{\text{inconsistency}} \\ \text{Edge}_{\text{variance}} \\ \text{OCR}_{\text{confidence}} \\ \text{Format}_{\text{anomaly}} \end{bmatrix}$$

Standardized using validation split statistics $\boldsymbol{\mu}, \boldsymbol{\sigma}$:
$$\mathbf{z} = \frac{\mathbf{x} - \boldsymbol{\mu}}{\boldsymbol{\sigma}}$$

Evaluated through regularized logistic weights $\boldsymbol{\beta}$ and intercept $\beta_0$:
$$\text{logit} = \beta_0 + \boldsymbol{\beta}^T \mathbf{z}$$
$$P_{\text{fused}} = \sigma(\text{logit}) = \frac{1}{1 + e^{-\text{logit}}}$$

### Weight Values:
- Intercept $\beta_0 = -0.2146$
- $\beta_{\text{global}} = +3.0660$ (Dominant macro visual signal)
- $\beta_{\text{patch}} = +0.2240$ (Local micro-texture confirmation)
- $\beta_{\text{ELA\_mean}} = -3.9836$ (Compression discrepancy sensitivity)
- $\beta_{\text{ELA\_incon}} = -0.1443$
- $\beta_{\text{Noise\_incon}} = -0.9050$
- $\beta_{\text{Edge\_var}} = +1.6052$ (High-frequency character boundary energy)
- $\beta_{\text{OCR\_conf}} = -0.4227$
- $\beta_{\text{Format\_anom}} = -0.4240$

---

## 3. Production Deliverable Structure

- **Interactive UI:** `app.py` (Streamlit web dashboard)
- **Inference Library:** `src/inference_engine.py` (Zero external dependencies outside PyTorch, PIL, EasyOCR, ReportLab)
- **PDF Certificate Engine:** `src/report_generator.py` (Client-ready forensic certificates)
- **Canonical Schemas:** `src/template_layouts.py` (10 European national ID and passport schemas)
