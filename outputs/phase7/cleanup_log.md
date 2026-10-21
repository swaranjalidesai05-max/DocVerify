# DocVerify — Phase 7 Codebase Cleanup Log

**Project:** DocVerify — AI-Based Document Verification and Forgery Detection System  
**Audit Stage:** Phase 7 Step 11 (Safe Codebase Cleanup & Deprecation Ledger)  
**Date:** October 17, 2026  
**Auditor:** Senior ML, Computer Vision & Security Research Engineer  

---

## 1. Principles of Safe Cleanup
1. **Preserve All Research History:** Do not delete historical scripts, reports, or models from Phase 1 through Phase 6.
2. **Preserve Dataset Integrity:** Never modify `Dataset/`.
3. **Deprecate Only Misleading / Oracle Production Logic:** Only eliminate code that actively risks connecting ground-truth annotations or outdated heuristics to user-facing inference.

---

## 2. Action Ledger

| File / Component | Action | Reason | Replacement / Active Status |
|---|---|---|---|
| `src/inference_engine.py` (Phase 5 Version) | **UPGRADED** | Used Phase 4 feature vector and lacked native Phase 6 dedicated patch model and forensic features | Upgraded to load `models/phase6/dedicated_patch_model.pth`, extract ELA/noise/edge forensics, and load `outputs/phase6/fusion_weights.json` |
| `app.py` (Engine instantiation) | **UPDATED** | Initialized `DocVerifyInferenceEngine` with deprecated `evidence_csv_path` | Updated to instantiate with Phase 6 model paths, fusion weights, and thresholds |
| `archive/phase6_previous_attempt/` (36 files) | **PRESERVED IN ARCHIVE** | Exploratory predecessor; kept for historical transparency | Preserved intact in archive directory |
| `outputs/phase7/test_debug.pdf` | **DELETED** | Scratch file created during PDF margin test | Replaced by permanent verified PDF reports in `outputs/phase7/sample_reports/` |
| Historical Phase 1–6 output directories | **PRESERVED IN FULL** | Core academic evidence | Preserved intact |
