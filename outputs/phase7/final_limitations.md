# DocVerify — System Limitations & Boundary Constraints

**Project:** DocVerify — AI-Based Document Verification and Forgery Detection System  
**Audit Stage:** Phase 7 Step 12 (Honest Limitations & Boundary Conditions)  
**Date:** October 17, 2026  
**Auditor:** Senior ML, Computer Vision & Security Research Engineer  

---

## 1. Experimental & Dataset Limitations

1. **SIDTD Dataset Scope:** The primary empirical evaluation is grounded on the Synthetic Identity Document Tampering Dataset (SIDTD). While SIDTD contains 2,222 documents across 10 European national ID and passport templates, cross-dataset generalization to completely unseen continents or physical driver's licenses has not been evaluated.
2. **Dataset Generation Shortcuts:** As discovered in Phase 6, raw SIDTD files carry non-semantic metadata artifacts (100% accuracy on file size/dimensions alone, and 77.2% on CMYK headers). While DocVerify actively mitigates these via standardized RGB loading and patch-scale verification, real-world deployment requires ongoing container sanitation.
3. **High-Fidelity Micro-Inpainting:** Single-character substitutions where font weight, kerning, and color perfectly match the host document (accounting for 52.6% of remaining false negatives) can evade global CNN detection unless captured by the local patch prior.

---

## 2. Engineering & Operational Limitations

1. **OCR Reliability on Complex Backgrounds:** In documents with intricate guilloche patterns or dense security watermarks (e.g. Russian internal passports and Serbian passports), OCR character confusion can trigger false format anomaly flags (accounting for 100% of false alarms).
2. **Template Identification Ambiguity:** When an uploaded document does not match any of the 10 canonical templates and exhibits an irregular aspect ratio (e.g. non-standard cropping or camera skew), the system falls back to central horizontal bands, which may not align with custom field coordinates.
3. **Hardware Execution Speed:** On CPU-only environments, end-to-end multi-modal inference averages ~3.44 seconds per document (primarily dominated by OCR at ~1.96s and CAM backpropagation at ~0.31s). Deployment on GPU or neural accelerators reduces this latency below 300ms.

---

## 3. Legal & Compliance Disclaimer

**DocVerify is an AI-assisted decision-support system, NOT a legal certification authority.** Results reflect statistical pattern recognition and forensic anomaly estimation. The system cannot guarantee 100% fraud prevention and must be utilized in conjunction with human compliance officers and physical document security inspections.
