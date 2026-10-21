# DocVerify — Phase 8 Dedicated Patch Model Evaluation Report

**Project:** DocVerify — AI-Based Document Verification and Forgery Detection System  
**Model:** Remediated Dedicated Patch Classifier (`models/phase8/dedicated_patch_model.pth`)  
**Backbone:** EfficientNet-B0 (ImageNet-1K pretrained weights)  
**Date:** October 18, 2026  
**Status:** VALIDATED ON FROZEN TEST SET (N=320)  

---

## 1. Executive Summary & Remediation Verdict

During the Phase 7A Independent Technical Audit, **Finding AUD-PATCH-01** confirmed that the Phase 6 patch model was trained exclusively on an arbitrary static central rectangular crop (`[0.35W, 0.45H, 0.25W, 0.08H]`) due to a Python dictionary/list parsing mismatch. As a result:

> [!WARNING]
> **HISTORICAL INVALIDATION DECLARATION:**  
> The previous EXP-06 benchmark metric of **99.38% test accuracy** is hereby formally classified as:  
> **"INVALIDATED / SUPERSEDED DUE TO PATCH-BOX PARSING ERROR"**  
> That metric reflected classification of a static background window rather than actual manipulated regions. It must not be cited as a valid benchmark of forgery detection capability.

In Phase 8, `src/bbox_parser.py` was deployed to parse 100% of the true SIDTD ground-truth manipulated bounding boxes (`VALID_SIDTD_BBOX`) across all 1,222 forged documents. Contextual crops (2.0x scale) were extracted at native sensor resolution. The model was trained strictly on the training partition ($N=1,565$) and selected via validation F1 ($N=337$) with true state-dict deep copying (`copy.deepcopy`), resolving AUD-PATCH-03.

---

## 2. Comparison: Historical Invalidated Model vs. Remediated Patch Model

| Metric | Historical Phase 6 Patch Model (EXP-06) | Remediated Phase 8 Patch Model | Status / Delta |
| :--- | :---: | :---: | :--- |
| **Bounding Box Source** | Static central window fallback (Bug) | True SIDTD Ground-Truth Bounding Boxes | **Remediated (100% genuine annotations)** |
| **Checkpoint Integrity** | Shallow copy (`.copy()`), parameter mutation | True deep copy (`copy.deepcopy()`) | **Remediated (Clean parameter isolation)** |
| **Test Accuracy** | *99.38% (Invalidated)* | **98.44%** | -0.94% (Honest scientific evaluation) |
| **Test Precision** | *99.41% (Invalidated)* | **100.00%** | +0.59% (Zero false alarms on genuine patches) |
| **Test Recall** | *99.41% (Invalidated)* | **97.04%** | -2.37% (5 missed micro-forgeries out of 169) |
| **Test F1-Score** | *99.41% (Invalidated)* | **98.50%** | -0.91% |
| **Test ROC-AUC** | *0.9997 (Invalidated)* | **0.9967** | High discriminative capacity |
| **Missed Forgeries (FN)** | *1 / 169 (Invalidated)* | **5 / 169** | Accurately reflects difficult micro-edits |
| **False Alarms (FP)** | *1 / 151 (Invalidated)* | **0 / 151** | Zero authentic patches flagged as forged |

---

## 3. Confusion Matrix Breakdown (Frozen Test Set $N=320$)

```
                       Predicted Genuine    Predicted Forged
Actual Genuine (N=151):      151 (TN)              0 (FP)
Actual Forged  (N=169):        5 (FN)            164 (TP)
```

- **True Negatives (TN):** 151 / 151 (100.0% specificity)
- **False Positives (FP):** 0 / 151 (0.0% false alarm rate)
- **False Negatives (FN):** 5 / 169 (2.96% missed forgery rate)
- **True Positives (TP):** 164 / 169 (97.04% sensitivity)

---

## 4. Per-Template Test Performance

| Template ID | Country | Document Type | Test Samples | Accuracy | Notes |
| :--- | :--- | :--- | :---: | :---: | :--- |
| `alb_id` | Albania | Identity Card | 31 | 100.0% | Clean character edge detection |
| `aze_passport` | Azerbaijan | Passport | 32 | 96.88% | 1 subtle numeric date replacement missed |
| `esp_id` | Spain | Identity Card | 32 | 100.0% | Excellent MRZ and photo border distinction |
| `est_id` | Estonia | Identity Card | 32 | 96.88% | 1 single-glyph alteration missed |
| `fin_id` | Finland | Identity Card | 32 | 100.0% | Robust against fine guilloche background |
| `grc_passport` | Greece | Passport | 32 | 96.88% | 1 faint stamp overlay misclassified |
| `lva_passport` | Latvia | Passport | 32 | 100.0% | Robust separation |
| `rus_internalpassport` | Russia | Internal Passport | 33 | 96.97% | 1 Cyrillic serial digit alteration missed |
| `srb_passport` | Serbia | Passport | 32 | 100.0% | Perfect classification |
| `svk_passport` | Slovakia | Passport | 32 | 96.88% | 1 subtle birthdate alteration missed |

---

## 5. Performance by Forgery Mechanism

| Forgery Type | Test Samples | Correctly Detected | Detection Recall | Failure Characteristics |
| :--- | :---: | :---: | :---: | :--- |
| **Photocopy / Digital Replacement** | 85 | 84 | **98.82%** | High contrast border artifacts easily captured |
| **Character Inpainting / Text Alteration** | 84 | 80 | **95.24%** | Subtle 1-character font matching represents all 4 text FNs |

---

## 6. Scientific Conclusion

The remediated dedicated patch classifier provides genuine, sensor-scale verification of manipulated text fields. While the resulting test accuracy (98.44%) is slightly lower than the invalidated 99.38% static-crop artifact, it is **empirically valid, reproducible, and grounded in ground-truth forgery geometry**.
