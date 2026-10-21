# DocVerify — Phase 9 UI/UX Refinement: Pre-Implementation Audit (Before)

**Project:** DocVerify — AI-Based Passport & Identity Document Verification & Forgery Detection System  
**Audit Date:** October 19, 2026  
**Auditor:** Senior Full-Stack AI & UX Systems Architect  
**Objective:** Independent audit and plan of action for resolving UI/UX visual defects, unstyled Streamlit artifacts, terminology exposure, and usability flaws.  

---

## 1. Current UI Problems Identified

An exhaustive inspection of the running application (`app.py`, `src/ui_components.py`, `src/report_generator.py`) and user interaction flows identified the following confirmed issues:

### A. Unexplained Radio-Button Artifacts (Floating Dots)
* **Observed Defect:** Red/black circular dots appeared floating without context near the document intake card.
* **Root Cause:** In `app.py` (lines 97–102), `st.radio("Intake Mode", ["Upload Document File", "Select Benchmark Test Set Document"], horizontal=True, label_visibility="collapsed")` was used. Hiding the label via `label_visibility="collapsed"` stripped the semantic title while leaving the unstyled circular radio inputs floating on the canvas.
* **Impact:** Confuses normal users; looks like an accidental or broken input control.

### B. Excessive Vertical Whitespace & Disjointed Spacing
* **Observed Defect:** Vast empty areas between controls, massive gaps under settings, and unbalanced column heights.
* **Root Cause:**
  1. Compounding margins: Streamlit default element margins (`1rem`) compounded with `.dv-card { margin-bottom: 1.25rem; }`, plus hardcoded `<div style='height: 12px;'></div>` and `<div style='height: 14px;'></div>` elements scattered throughout `app.py`.
  2. Column asymmetry: In `col_upload, col_settings = st.columns([1.1, 0.9], gap="large")`, the upload box is substantially taller than the two selectboxes, leaving an empty void on the right side.
  3. Previews hidden inside an `st.expander`, which pushed the primary action buttons off the fold.

### C. Default / Unstyled Streamlit Widgets
* **Observed Defect:** Dropdowns and selectboxes lacked custom styling, displaying standard browser borders, accidental text-selection cursors, and font mismatches with the custom Plus Jakarta Sans theme.
* **Root Cause:** `src/ui_components.py` provided CSS for cards, verdict banners, and metric tiles, but omitted rules for `div[data-baseweb="select"]`, `.stSelectbox`, and `.stFileUploader`.

### D. Exposure of Internal Technical Terminology
* **Observed Defect:** Users were exposed to internal technical terms such as:
  - "Youden's J Optimal"
  - Internal template keys (`alb_id`, `srb_passport`, `esp_id`)
  - "p_global_fake", "max_p_patch", "ela_inconsistency"
  - "Certified PDF" (legally inappropriate for an academic AI decision-support tool)
* **Root Cause:** Absence of a centralized, user-facing translation layer for policies, templates, and forensic metrics.

### E. Unclear Document Intake & Preview Flow
* **Observed Defect:** Uploading an image did not display an immediate, clear document card; instead, it placed the preview inside a collapsible expander. If a user wished to change the document, there was no intuitive "Remove / Change Document" action.
* **Root Cause:** Lack of a dedicated Section 3 "Document Preview" card with clear metadata and reset action.

### F. Progress Feedback Ambiguity
* **Observed Defect:** During inference, a generic card displayed "Executing Multi-Modal Forensic Verification..." without communicating clear, discrete processing stages.
* **Root Cause:** Static text block rather than a structured multi-stage progression tracker reflecting genuine backend execution stages.

---

## 2. Files Involved in Refinement

1. `app.py`: Main application layout, intake flow, state management, verdict presentation, evidence tabs, report download.
2. `src/ui_components.py`: Enterprise design system, CSS styling tokens, BaseWeb selectbox styling, user-friendly labels, policy definitions.
3. `src/report_generator.py`: PDF report title and disclaimer alignment (replacing "Certified" with "AI-Assisted Verification Report").
4. `src/template_layouts.py`: Human-readable country and document type mapping verification.

---

## 3. Components That Will NOT Be Modified (Strict Protection)

The following core components are **100% FROZEN** and will strictly NOT be altered:
- Global classifier checkpoint: `models/phase2/best_model.pth`
- Dedicated patch model checkpoint: `models/phase8/dedicated_patch_model.pth`
- Fusion weights and scaler: `outputs/phase8/fusion/fusion_weights.json`
- Calibrator model: `models/phase8/fusion_calibrator.joblib`
- Operational thresholds: `outputs/phase8/fusion/operational_thresholds.json`
- Feature schema and 7-D vector extraction: `src/inference_engine.py`
- Core Grad-CAM algorithm: `src/gradcam_corrected.py`
- SIDTD dataset files: `Dataset/`
- Test suite and assertions: `scripts/test_phase8.py`

---

## 4. Proposed Structural Refinements

```
┌────────────────────────────────────────────────────────────────────────┐
│ HEADER: DocVerify | AI-Powered Passport & Identity Document Verification│
│ Status: ● System Ready                                                 │
├───────────────────────────────────┬────────────────────────────────────┤
│ SECTION 1: UPLOAD YOUR DOCUMENT   │ SECTION 2: VERIFICATION SETTINGS   │
│ - Dominant drag & drop zone       │ - Verification Policy:             │
│ - Supported: JPG, PNG, WEBP, BMP  │   • Balanced (Recommended)         │
│ - Max: 200 MB                     │   • High Recall                    │
│ - Optional sample benchmark picker│   • Strict Anti-Fraud              │
│                                   │ - Document Type:                   │
│                                   │   • Automatic Detection            │
│                                   │   • Human-readable country types   │
│                                   │ - Primary Action: Verify Document  │
├───────────────────────────────────┴────────────────────────────────────┤
│ SECTION 3: DOCUMENT PREVIEW (Active upon upload)                       │
│ - High-res preview image | Filename, Dimensions, Detected Type         │
│ - [Change / Remove Document] button                                    │
├────────────────────────────────────────────────────────────────────────┤
│ SECTION 4: REAL-TIME VERIFICATION ANALYSIS STAGES                      │
│ ✓ Image standardized | ✓ Layout matched | ✓ Visual authenticity...     │
├────────────────────────────────────────────────────────────────────────┤
│ SECTION 5 & 6: VERIFICATION RESULT & TAMPERING RISK                    │
│ - Unambiguous Verdict: VERIFIED GENUINE / REVIEW REQUIRED / FORGERY    │
│ - Prominent Tampering Risk: e.g. "27% Estimated Tampering Risk"        │
├────────────────────────────────────────────────────────────────────────┤
│ SECTION 7: MULTI-MODAL EVIDENCE CARDS (5 Core Dimensions)              │
│ [Visual Analysis] [Field Inspection] [Text & Typography]               │
│ [Forensic Consistency] [Evidence Fusion]                               │
├────────────────────────────────────────────────────────────────────────┤
│ SECTION 8–11: DETAILED EVIDENCE TABS                                   │
│ Tab 1: Visual Explainability (Responsible Grad-CAM explanation)        │
│ Tab 2: Field-Level Inspection (High-res field crops & risk status)     │
│ Tab 3: Text & Typography (EasyOCR transcriptions, format checks)       │
│ Tab 4: Forensic Consistency (ELA & Noise spatial variance)             │
│ Tab 5: AI-Assisted Verification Report (PDF Download)                  │
├────────────────────────────────────────────────────────────────────────┤
│ SECTION 12: TECHNICAL / RESEARCH DETAILS (Collapsible Expander)        │
│ SECTION 13: SUBORDINATE DECISION-SUPPORT DISCLAIMER FOOTER             │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 5. Verification Plan

Following code implementation:
1. Verify elimination of radio-button dots.
2. Verify BaseWeb selectbox styling and typography consistency.
3. Test upload, preview, and verification across authentic and forged documents.
4. Test state clearing on document switch.
5. Verify PDF report generation and download.
6. Re-run `scripts/test_phase8.py` to confirm 14/14 automated tests pass with zero regressions.
