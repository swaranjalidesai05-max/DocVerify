# DOCVERIFY: Phase 2 AI Baseline Report
**Project:** DocVerify — AI-Based Document Verification and Forgery Detection System  
**Phase:** Phase 2 — Dataset Preparation, Identity-Disjoint Splitting, and AI Baseline Classification  
**Date:** 2026-10-17  
**Author:** Senior Machine Learning & Computer Vision Engineer  

---

## Executive Summary

Phase 2 successfully established the first scientifically validated, identity-leakage-free AI baseline for the DocVerify project. 

The predefined SIDTD split suffered from over 93% document identity leakage. In Phase 2, a complete **connected identity-cluster splitting architecture** was engineered. It groups all physical documents involved in inpainting, splicing, and crop-and-replace manipulations into disjoint identity clusters, guaranteeing **0.0% identity leakage** between training, validation, and testing.

Using this verified split, an ImageNet-pretrained **EfficientNet-B0** classifier was fine-tuned with in-memory CMYK-to-RGB conversion, conservative document augmentations, and training-derived class weighting.

### Key Results on Untouched Test Set (Unseen Document Identities):
- **Test Accuracy:** **79.69%**
- **Test Precision:** **82.91%**
- **Test Recall (Sensitivity):** **77.51%**
- **Test F1-Score:** **80.12%**
- **Test ROC-AUC:** **0.9112**
- **Confusion Matrix:** True Negative (TN) = 124, False Positive (FP) = 27, False Negative (FN) = 38, True Positive (TP) = 131.
- **Identity Leakage:** **Strictly 0 (0.00%)**.

---

## 1. Physical Dataset & Manifest Compilation

A complete manifest was compiled at `outputs/dataset_manifest/dataset_manifest.csv` referencing all 2,222 physical images on disk:
- **Total Physical Images:** 2,222
  - **Real Images:** 1,000 (`label = 0`, `reals`)
  - **Fake Images:** 1,222 (`label = 1`, `fakes`)
- **Metadata Fields:** `image_path`, `absolute_path`, `label`, `base_document_id`, `identity_cluster_id`, `template`, `country`, `doc_type`, `forgery_type`, `manipulated_field`, `bounding_box_json`, `width`, `height`, `color_mode`, `md5_hash`, `is_exact_duplicate`, `split`.

---

## 2. Base Document Identity & Identity-Disjoint Splitting Methodology

### The Connected Component Identity Challenge
In SIDTD, manipulations are not purely isolated:
1. `Inpaint_and_Rewrite` operations modify a single source document (e.g., `alb_id_00.jpg` $\rightarrow$ `alb_id_00_fake_6_25.jpg`).
2. `Crop_and_Replace` operations splice a field from a donor document (`second_src`) onto a recipient document (`src`). For instance, `alb_id_23_fake_6_106.jpg` crops text from donor `alb_id_23.jpg` and pastes it onto recipient `alb_id_87.jpg`.

If `alb_id_23` were placed in Train while `alb_id_87` were placed in Test, visual features from the donor document would leak into the test set.

### The Solution: Graph Connected-Component Clustering
We modeled every document and manipulation relationship as an undirected graph:
- **Nodes:** Physical document identities (1,000 real document instances).
- **Edges:** Bi-directional relationships between recipient (`src`), donor (`second_src`), and filename identity prefixes.
- **Result:** Exactly **927 isolated identity clusters** were formed (865 single-document clusters, 62 multi-document spliced clusters).

### Stratified Group Assignment
The 927 clusters were grouped by document template (`alb_id`, `aze_passport`, etc.) and partitioned using a fixed random seed (`seed=42`) into:
- **Train Split:** 649 clusters (1,565 images: 703 real, 862 fake) — **70.4%**
- **Validation Split:** 139 clusters (337 images: 146 real, 191 fake) — **15.2%**
- **Test Split:** 139 clusters (320 images: 151 real, 169 fake) — **14.4%**

### Leakage Verification Guarantee:
- `Train ∩ Val` base identities: **0**
- `Train ∩ Test` base identities: **0**
- `Val ∩ Test` base identities: **0**
- `Train ∩ Val ∩ Test` image hash collisions: **0**

### Duplicate Image Handling:
The single exact duplicate pair discovered in Phase 1 (`grc_passport_96_fake_4_101.jpg` and `grc_passport_96_fake_4_42.jpg`, sharing MD5 `405c441a...`) was tracked and placed together in the **Validation** split. Zero duplicate leakage exists across splits.

---

## 3. Data Preprocessing & RGB Pipeline

1. **In-Memory RGB Conversion:**  
   Phase 1 identified that 500 real documents are encoded in CMYK color space. Standard OpenCV loaders silently misinterpret CMYK buffers. The custom `SIDTDDataset` pipeline enforces:
   ```python
   with Image.open(path) as img:
       img_rgb = img.convert("RGB")
   ```
2. **Preprocessing Normalization:**  
   Images are resized to $256 \times 256$, center-cropped to $224 \times 224$, and normalized using standard ImageNet parameters ($\mu = [0.485, 0.456, 0.406]$, $\sigma = [0.229, 0.224, 0.225]$).
3. **Conservative Document Augmentation (Train only):**  
   Aggressive transforms (cutout, heavy shear, strong blurring) destroy subtle inpainting edge boundaries. We applied conservative, document-safe augmentations:
   - Random horizontal flip ($p=0.5$)
   - Micro-rotation ($\pm 4^\circ$)
   - Subtle ColorJitter (brightness $\pm 0.08$, contrast $\pm 0.08$)
   - Validation and Test preprocessing remained strictly deterministic.

---

## 4. Model Architecture & Training Configuration

- **Backbone:** `EfficientNet-B0` (ImageNet-pretrained).
- **Classification Head:** Linear projection layer (`in_features=1280`, `out_features=2`).
- **Trainable Parameters:** 4,010,110 (all layers fine-tuned end-to-end).
- **Batch Size:** 32.
- **Optimizer:** AdamW ($\text{lr}=10^{-4}$, $\text{weight\_decay}=10^{-4}$).
- **Scheduler:** CosineAnnealingLR ($T_{\max}=10$).
- **Loss Function:** CrossEntropyLoss with class weights derived strictly from the **training split only**:
  - $w_{\text{real}} = 1.1131$, $w_{\text{fake}} = 0.9078$.
- **Checkpointing:** Monitored per-epoch validation F1-score; the optimal checkpoint was automatically preserved at `models/phase2/best_model.pth`.

---

## 5. Experimental Results & Test Evaluation

The selected best model (from Epoch 9, with Validation F1 = 0.8356) was evaluated **ONCE** on the untouched Test set containing 320 unseen document identities.

### Benchmark Metrics:
| Metric | Test Set Score |
| :--- | :--- |
| **Accuracy** | **79.69%** (255 / 320) |
| **Precision** | **82.91%** |
| **Recall (Sensitivity)** | **77.51%** |
| **F1-Score** | **80.12%** |
| **ROC-AUC** | **0.9112** |
| **False Positive Rate (FPR)** | **17.88%** (27 / 151) |
| **False Negative Rate (FNR)** | **22.49%** (38 / 169) |

### Confusion Matrix Breakdown:
- **True Negative (TN):** 124 (Real documents correctly verified as genuine)
- **False Positive (FP):** 27 (Real documents misclassified as fake)
- **False Negative (FN):** 38 (Forged documents misclassified as real)
- **True Positive (TP):** 131 (Forged documents correctly detected)

### Per-Class Performance:
- **Real Document Accuracy:** **82.12%** (124 / 151)
- **Fake Document Accuracy:** **77.51%** (131 / 169)

---

## 6. Granular Breakdown: Forgery Types & Document Templates

### A. Performance by Forgery Type:
| Forgery Type | Test Count | Recall (Detection Rate) | Accuracy |
| :--- | :--- | :--- | :--- |
| **Inpaint_and_Rewrite** | 145 | **78.62%** | 78.62% |
| **Crop_and_Replace** | 24 | **70.83%** | 70.83% |

*Interpretation:* Splicing attacks (`Crop_and_Replace`) proved more difficult than inpainting for the baseline classifier. In `Crop_and_Replace`, donor text inherits genuine camera noise and sensor characteristics, whereas `Inpaint_and_Rewrite` introduces subtle synthetic typography and localized interpolation smoothing that the CNN captures more effectively.

### B. Performance by Document Template:
| Document Template | Test Samples | Accuracy | Observation |
| :--- | :--- | :--- | :--- |
| `est_id` | 29 | **100.0%** | Pristine separation on clean identity layouts |
| `fin_id` | 35 | **100.0%** | Flawless detection across real and fake |
| `svk_id` | 25 | **96.00%** | Near-perfect classification |
| `aze_passport` | 35 | **88.57%** | Strong discrimination |
| `esp_id` | 36 | **83.33%** | High reliability |
| `grc_passport` | 34 | **73.53%** | Moderate complexity on passport textures |
| `lva_passport` | 21 | **71.43%** | Moderate complexity |
| `srb_passport` | 34 | **70.59%** | Moderate complexity |
| `rus_internalpassport` | 29 | **65.52%** | Complex Guilloche background patterns |
| `alb_id` | 42 | **54.76%** | High error rate; dark background and noise |

---

## 7. Error Analysis & Limitations

Representative error cases were saved to `outputs/phase2/error_analysis/`:

1. **False Negatives (Missed Forgeries, e.g., `FN_alb_id_03_fake_6_69.jpg`):**
   - **Root Cause:** Micro-inpainting on compact fields (such as a 1-character gender modification or subtle 2-digit expiry date change). Downsampling high-resolution identity documents ($2167 \times 1525$) to $224 \times 224$ compresses small text fields into a handful of pixels, diminishing global CNN sensitivity to micro-manipulations.
2. **False Positives (Genuine Flags as Fake, e.g., `FP_alb_id_11.jpg`):**
   - **Root Cause:** Heavy high-frequency background patterns (Guilloche lines, security micro-print, laminated grain) that simulate inpainting noise artifacts to a global convolutional kernel.

---

## 8. Artifacts & Generated Files

All deliverables are organized outside the dataset directory:

1. **Scripts:**
   - [`scripts/build_manifest.py`](file:///E:/DocverifyAi/scripts/build_manifest.py) — Manifest & identity-disjoint clustering pipeline.
   - [`scripts/verify_split.py`](file:///E:/DocverifyAi/scripts/verify_split.py) — Mathematical split integrity & zero-leakage verification.
   - [`scripts/train_phase2.py`](file:///E:/DocverifyAi/scripts/train_phase2.py) — PyTorch training, validation, checkpointing, and evaluation script.
2. **Manifest:**
   - [`outputs/dataset_manifest/dataset_manifest.csv`](file:///E:/DocverifyAi/outputs/dataset_manifest/dataset_manifest.csv) (2,222 verified rows).
3. **Trained Checkpoints:**
   - [`models/phase2/best_model.pth`](file:///E:/DocverifyAi/models/phase2/best_model.pth) (Saved at Epoch 9, Val F1 = 0.8356).
   - [`models/phase2/final_model.pth`](file:///E:/DocverifyAi/models/phase2/final_model.pth).
4. **Machine-Readable Metrics:**
   - [`outputs/phase2/phase2_results.json`](file:///E:/DocverifyAi/outputs/phase2/phase2_results.json).
   - [`outputs/phase2/experiment_config.json`](file:///E:/DocverifyAi/outputs/phase2/experiment_config.json).
5. **Visualizations:**
   - [`training_curves.png`](file:///E:/DocverifyAi/outputs/phase2/visualizations/training_curves.png)
   - [`confusion_matrix.png`](file:///E:/DocverifyAi/outputs/phase2/visualizations/confusion_matrix.png)
   - [`roc_curve.png`](file:///E:/DocverifyAi/outputs/phase2/visualizations/roc_curve.png)
6. **Error Analysis Samples:**
   - [`outputs/phase2/error_analysis/`](file:///E:/DocverifyAi/outputs/phase2/error_analysis/)

---

## 9. Recommendations for Phase 3

The findings from Phase 2 clearly identify the roadmap for Phase 3:
1. **Explainability & Localization (Grad-CAM):**  
   Because global classification achieved **0.9112 ROC-AUC**, generating visual explanation heatmaps via Grad-CAM / HiRes-CAM on the EfficientNet convolutional blocks will expose which regions triggered forgery flags.
2. **Ground-Truth Localization Evaluation:**  
   Phase 1 and 2 established exact bounding-box coordinates for all 1,222 fake samples. Phase 3 can quantitatively evaluate heatmap localization accuracy against ground-truth boxes using Pointing Game and Intersection over Union (IoU) metrics.
3. **Patch-Level or High-Resolution Crop Ensembles:**  
   To resolve the False Negatives caused by $224 \times 224$ global downsampling, Phase 3 can evaluate local crop/patch analysis for suspected text fields.
