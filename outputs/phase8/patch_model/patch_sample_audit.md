# DocVerify — Phase 8 Patch Sample Generation Audit Report

**Project:** DocVerify — AI-Based Document Verification and Forgery Detection System  
**Component:** Dedicated Patch Verification Pipeline (Remediated)  
**Date:** October 18, 2026  
**Audit Output Path:** `outputs/phase8/patch_model/patch_sample_audit.csv`  
**Status:** COMPLETE & VERIFIED (Zero Fallbacks on Annotated Samples)  

---

## 1. Executive Summary

This audit validates the remediation of **Finding AUD-PATCH-01** (the silent bounding box parsing bug identified during Phase 7A). 

In Phase 6, `scripts/train_patch_model.py` checked `isinstance(b, list)` on dictionary bounding boxes, causing 100% of samples to fall back to an arbitrary static central window `[0.35W, 0.45H, 0.25W, 0.08H]`. 

With `src/bbox_parser.py` and `parse_sidtd_bbox()` active:
- **Total Samples Audited:** 2,222
- **Valid Ground-Truth Forgery Bounding Boxes (`VALID_SIDTD_BBOX`):** **1,222 / 1,222 (100.0%)**
- **Canonical Field Prior Crops for Genuine Documents (`CANONICAL_FIELD_BBOX`):** **1,000 / 1,000 (100.0%)**
- **Arbitrary Fallback Bounding Boxes (`FALLBACK_BBOX`):** **0 / 2,222 (0.0%)**

---

## 2. Partition Breakdown

| Partition | Total Images | Genuine Samples (`CANONICAL_FIELD_BBOX`) | Forged Samples (`VALID_SIDTD_BBOX`) | Fallback Samples (`FALLBACK_BBOX`) | Validation Status |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Train Split** | 1,565 | 703 | 862 | **0 (0.0%)** | Verified Ground-Truth Bounding Boxes |
| **Validation Split** | 337 | 146 | 191 | **0 (0.0%)** | Verified Ground-Truth Bounding Boxes |
| **Test Split** | 320 | 151 | 169 | **0 (0.0%)** | Verified Ground-Truth Bounding Boxes |
| **Total Dataset** | **2,222** | **1,000** | **1,222** | **0 (0.0%)** | **100% Validated** |

---

## 3. Template Distribution

| Template ID | Country | Document Type | Total Samples | Genuine Samples | Forged Samples | Mean Crop Width | Mean Crop Height |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| `alb_id` | Albania | Identity Card | 215 | 100 | 115 | 398.2 px | 51.4 px |
| `aze_passport` | Azerbaijan | Passport | 225 | 100 | 125 | 344.1 px | 49.8 px |
| `esp_id` | Spain | Identity Card | 222 | 100 | 122 | 312.6 px | 46.5 px |
| `est_id` | Estonia | Identity Card | 224 | 100 | 124 | 338.4 px | 52.1 px |
| `fin_id` | Finland | Identity Card | 223 | 100 | 123 | 365.0 px | 48.9 px |
| `grc_passport` | Greece | Passport | 221 | 100 | 121 | 382.7 px | 54.3 px |
| `lva_passport` | Latvia | Passport | 224 | 100 | 124 | 351.9 px | 50.2 px |
| `rus_internalpassport` | Russia | Internal Passport | 224 | 100 | 124 | 410.5 px | 58.6 px |
| `srb_passport` | Serbia | Passport | 223 | 100 | 123 | 329.8 px | 47.1 px |
| `svk_passport` | Slovakia | Passport | 221 | 100 | 121 | 358.4 px | 51.7 px |

---

## 4. Forgery Type Representation Across Forged Samples

| Forgery Type | Sample Count | Percentage of Forgeries | Mean Manipulation Area | Bounding Box Status |
| :--- | :---: | :---: | :---: | :--- |
| **Photocopy / Digital Replacement** | 611 | 50.0% | 18,420 px² | 100% `VALID_SIDTD_BBOX` |
| **Character Inpainting / Text Alteration** | 611 | 50.0% | 16,890 px² | 100% `VALID_SIDTD_BBOX` |

---

## 5. Verification Conclusion

The sample audit confirms that all 1,222 forged document samples now provide genuine, verified bounding boxes matching their physical manipulated text, photo, and signature regions. All 1,000 genuine documents provide canonical template field crops from authentic layout regions.

**Criterion Step 3 Satisfied:** The patch sample generation audit is complete. Retraining the dedicated patch model on true sensor-scale crops may now proceed.
