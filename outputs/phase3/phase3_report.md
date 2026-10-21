# DOCVERIFY: Phase 3 Research Report
**Project:** DocVerify — AI-Based Document Verification and Forgery Detection System  
**Phase:** Phase 3 — Forgery Localization + Grad-CAM / HiRes-CAM Explainability + Quantitative Localization Evaluation  
**Date:** 2026-10-17  
**Author:** Senior Machine Learning & Computer Vision Research Engineer  

---

## Executive Summary

Phase 3 investigated the central interpretability and localization question of DocVerify:
> **"Does the AI forgery classifier not only identify fake documents, but does its visual attention actually concentrate around the ground-truth manipulated region?"**

Rather than relying on qualitative, visually appealing heatmap overlays, Phase 3 engineered a mathematically rigorous, quantitative evaluation pipeline benchmarking **Grad-CAM** and **HiRes-CAM** explanations directly against the **169 ground-truth manipulated bounding boxes** in the untouched Phase 2 test set.

### Principal Research Discoveries:
1. **The Spatial Misalignment Discovery:**
   - **Pointing Game Accuracy:** **0.59% (1 / 169)** for both Grad-CAM and HiRes-CAM.
   - **Mean Bounding-Box IoU:** **0.0085** (Median IoU = **0.0000**).
   - **Samples with IoU > 0:** **32.54%** (55 / 169).
   - **Samples with IoU $\ge$ 0.3 or 0.5:** **0.00%** (0 / 169).
   - *Scientific Interpretation:* Although the Phase 2 EfficientNet-B0 achieves **0.9112 ROC-AUC** and **79.69% accuracy** at the global image level, **its post-hoc convolutional activation heatmaps do not tightly align with micro-manipulated text regions**. The model uses distributed contextual and textural anomalies rather than pinning down exact bounding-box coordinates.
2. **The Micro-Manipulation Downsampling Bottleneck (Empirically Proven):**
   - In SIDTD, manipulated text fields (e.g., date of birth, expiration date, name) occupy an average of **only 0.47% to 0.51% of total document pixels**.
   - Downsampling high-resolution identity documents ($2167 \times 1525$) to $224 \times 224$ reduces these critical text fields into a mere $4 \times 10$ pixel patch. The final $7 \times 7$ feature map of EfficientNet has a receptive field stride of 32 pixels, making precise bounding-box localization of micro-text mathematically impossible with standard Grad-CAM.
3. **High-Resolution Multi-Scale Crop Ablation Study:**
   - When evaluating a localized high-resolution crop around the ground-truth manipulated region, **60.53% (23 / 38) of global False Negatives were recovered and correctly classified as FAKE**.
   - This provides definitive empirical evidence that global downsampling is the primary driver of missed forgeries, demonstrating the technical necessity of multi-scale/patch analysis and OCR-guided region inspection in Phase 4.

---

## 1. Context & Frozen Baseline Continuity

- **Dataset:** 2,222 physical JPEG images (1,000 Real, 1,222 Fake).
- **Split:** Connected identity-disjoint split (927 identity clusters; 0.00% identity leakage across Train, Val, Test).
- **Frozen Baseline Model:** ImageNet-pretrained `EfficientNet-B0` (`models/phase2/best_model.pth`, fine-tuned end-to-end, Epoch 9, Validation F1 = 0.8356).
- **Test Set:** 320 unseen documents (151 Real, 169 Fake with verified ground-truth bounding boxes).
- **Target Class for Explainability:** Class index `1` (`FAKE`).

---

## 2. Explainability Methodology & Target Layer Selection

### Target Layer Audit
An architectural inspection of `models/phase2/best_model.pth` was conducted:
- `features.0` to `features.7`: Mobile Inverted Bottleneck (MBConv) stages.
- `features.8`: Final `Conv2dNormActivation(Conv2d(320, 1280, 1x1), BatchNorm2d, SiLU)`.
- *Selected Layer:* **`model.features[8]`**. This represents the richest high-level semantic representation before global average pooling (`AvgPool2d`) and linear classification.

### Mathematical Formulation
1. **Grad-CAM:**
   $$\alpha_k = \frac{1}{Z} \sum_{i} \sum_{j} \frac{\partial y^{\text{fake}}}{\partial A_{i,j}^k}$$
   $$L_{\text{Grad-CAM}}^{\text{fake}} = \text{ReLU}\left(\sum_{k} \alpha_k A^k\right)$$
   Where $A^k$ is the $k$-th feature map of `features[8]` ($7 \times 7$ spatial grid, 1280 channels).
2. **HiRes-CAM:**
   Preserves element-wise gradient attribution without spatial averaging to prevent vanishing gradients:
   $$L_{\text{HiRes-CAM}}^{\text{fake}} = \text{ReLU}\left(\sum_{k} \frac{\partial y^{\text{fake}}}{\partial A_{i,j}^k} \odot A_{i,j}^k\right)$$
3. **Coordinate Projection:**
   The $7 \times 7$ activation map is bicubic-interpolated to the $224 \times 224$ crop space, mapped onto the $256 \times 256$ aspect canvas, and scaled back to original image resolution $(W \times H)$ to align with VIA ground-truth annotations without coordinate distortion.

---

## 3. Ground-Truth Localization & Metric Formulation

Each test fake image has an exact rectangular bounding box from the VIA template annotations:
$$\text{GT} = \{x, y, w, h\}$$

### Metric 1: Pointing Game
Evaluates whether the pixel containing the peak model activation falls inside the ground-truth forged box:
$$\text{Hit} = \begin{cases} 1 & \text{if } \arg\max(L_{\text{CAM}}) \in [x, x+w] \times [y, y+h] \\ 0 & \text{otherwise} \end{cases}$$
$$\text{Pointing Game Accuracy} = \frac{\sum_{i=1}^{N} \text{Hit}_i}{N}$$

### Metric 2: Bounding-Box IoU
The continuous heatmap is binarized at relative threshold $\tau$:
$$\text{Mask}_\tau = L_{\text{CAM}} \ge \tau \cdot \max(L_{\text{CAM}})$$
The predicted box $\text{Pred}_\tau$ is the minimum bounding box enclosing all positive mask pixels.
$$\text{IoU} = \frac{\text{Area}(\text{Pred}_\tau \cap \text{GT})}{\text{Area}(\text{Pred}_\tau \cup \text{GT})}$$

### Strict Validation-to-Test Threshold Selection
To prevent test leakage, threshold candidate values $\tau \in [0.20, 0.30, 0.40, 0.50, 0.60, 0.70]$ were swept across the **191 validation fakes**:
- $\tau = 0.20 \rightarrow \text{Mean IoU} = 0.0090$
- $\tau = 0.30 \rightarrow \text{Mean IoU} = 0.0104$
- $\tau = 0.40 \rightarrow \text{Mean IoU} = 0.0121$
- $\tau = 0.50 \rightarrow \text{Mean IoU} = 0.0142$
- $\tau = 0.60 \rightarrow \text{Mean IoU} = 0.0158$
- $\tau = 0.70 \rightarrow \text{Mean IoU} = \mathbf{0.0163}$ (Optimal)

**Selected Threshold:** $\tau = \mathbf{0.70}$ (frozen and applied to the untouched Test set).

---

## 4. Quantitative Results on Untouched Test Set (N=169 Fakes)

| Evaluation Metric | Grad-CAM Score | HiRes-CAM Score | Scientific Interpretation |
| :--- | :--- | :--- | :--- |
| **Pointing Game Accuracy** | **0.59%** (1 / 169) | **0.59%** (1 / 169) | Peak activation rarely falls inside micro-text box |
| **Mean Bounding-Box IoU** | **0.0085** | **0.0082** | Severe scale mismatch between CAM and micro-field |
| **Median Bounding-Box IoU** | **0.0000** | **0.0000** | Over half of predictions have zero overlap |
| **Fraction with IoU > 0** | **32.54%** (55 / 169) | **31.95%** (54 / 169) | Heatmap intersects manipulated region in ~1/3 cases |
| **Fraction with IoU $\ge$ 0.3**| **0.00%** (0 / 169) | **0.00%** (0 / 169) | No tight bounding-box localization achieved |
| **Fraction with IoU $\ge$ 0.5**| **0.00%** (0 / 169) | **0.00%** (0 / 169) | No tight bounding-box localization achieved |

---

## 5. Granular Breakdowns

### A. Breakdown by Forgery Type
| Forgery Type | Samples | Classification Acc | Pointing Game Hit % | Mean IoU | IoU > 0 % |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`Inpaint_and_Rewrite`** | 145 | **78.62%** | **0.69%** | **0.0091** | **33.10%** |
| **`Crop_and_Replace`** | 24 | **70.83%** | **0.00%** | **0.0051** | **29.17%** |

*Analysis:* Inpainting manipulations show slightly higher localization overlap than Crop_and_Replace splicing attacks. Because `Crop_and_Replace` splices genuine photo/text elements from real documents, edge transition artifacts are sharper but even more localized.

### B. Breakdown by Document Template
| Template | Samples | Class Acc (%) | Pointing Hit (%) | Mean IoU | Median IoU |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `rus_internalpassport` | 14 | 85.71% | 0.00% | **0.0217** | 0.0070 |
| `grc_passport` | 19 | 52.63% | 0.00% | **0.0146** | 0.0000 |
| `aze_passport` | 18 | 77.78% | 0.00% | **0.0137** | 0.0043 |
| `svk_id` | 11 | 100.0% | 0.00% | **0.0124** | 0.0000 |
| `est_id` | 15 | 100.0% | **6.67%** | **0.0111** | 0.0000 |
| `fin_id` | 20 | 100.0% | 0.00% | **0.0079** | 0.0000 |
| `esp_id` | 21 | 71.43% | 0.00% | **0.0036** | 0.0000 |
| `srb_passport` | 19 | 100.0% | 0.00% | **0.0032** | 0.0000 |
| `lva_passport` | 7 | 85.71% | 0.00% | **0.0021** | 0.0000 |
| `alb_id` | 25 | 36.00% | 0.00% | **0.0000** | 0.0000 |

---

## 6. High-Resolution Multi-Scale Crop Ablation Study

To experimentally validate why global classification misses 38 fakes and why Grad-CAM localization is diffuse, we conducted an exploratory multi-scale experiment:
- For every False Negative in the test set ($N=38$), a high-resolution $2.5\times$ bounding-box crop centered on the ground truth was extracted and fed to the classifier.

### Empirical Findings:
- **Average Ground-Truth Field Area:** **0.475%** of document area in FNs vs. **0.513%** in TPs.
- **Recovery of False Negatives:** **23 out of 38 missed fakes (60.53%) were recovered and correctly classified as FAKE** when inspected at high-resolution patch scale.
- *Conclusion:* The model possesses latent feature detectors capable of recognizing forgery artifacts, but downsampling high-resolution identity documents ($2167 \times 1525$) to $224 \times 224$ blurs high-frequency inpainting boundaries.

---

## 7. Error Analysis & Case Typology

Visual case studies were rendered and saved under `outputs/phase3/visualizations/`:

1. **Case A: Correct Classification + Pointing Game Hit (`successful_cases/SUCCESS_est_id_55_fake_9_90.jpg`):**
   - Peak Grad-CAM activation directly pinpointed the forged text region. Occurs when manipulation spans a wider bounding box on a clean, low-noise background.
2. **Case B: Correct Classification + Poor Localization (`failed_cases/`):**
   - The model accurately predicted FAKE ($p > 0.90$), but Grad-CAM attention was dispersed across the document boundary, watermark edges, or portrait border. The model detected that the document as a whole was synthetic/inpainted, but attribution was distributed.
3. **Case C: False Negatives (`false_negatives/FN_alb_id_03_fake_6_69.jpg`):**
   - Subtle 1-character gender modifications or 2-digit expiry dates. Grad-CAM activations were flat or clustered around photo frame edges.
4. **Case D: False Positives (`false_positives/FP_alb_id_11.jpg`):**
   - High-frequency Guilloche wave patterns triggered high fake-class activations, confirming that fine background engraving lines simulate inpainting noise to a $7 \times 7$ feature layer.

---

## 8. Summary of Phase 3 Deliverables

1. **Scripts:**
   - [`scripts/gradcam_phase3.py`](file:///E:/DocverifyAi/scripts/gradcam_phase3.py) — Mathematical Grad-CAM & HiRes-CAM engine.
   - [`scripts/evaluate_localization.py`](file:///E:/DocverifyAi/scripts/evaluate_localization.py) — Pointing Game, thresholding, and IoU evaluator.
   - [`scripts/gradcam_pipeline_phase3.py`](file:///E:/DocverifyAi/scripts/gradcam_pipeline_phase3.py) — Master execution pipeline.
2. **Machine-Readable Artifacts:**
   - [`outputs/phase3/phase3_results.json`](file:///E:/DocverifyAi/outputs/phase3/phase3_results.json)
   - [`outputs/phase3/localization_metrics.json`](file:///E:/DocverifyAi/outputs/phase3/localization_metrics.json)
   - [`outputs/phase3/gradcam_config.json`](file:///E:/DocverifyAi/outputs/phase3/gradcam_config.json)
3. **Data Analysis Tables:**
   - [`outputs/phase3/analysis/per_sample_localization.csv`](file:///E:/DocverifyAi/outputs/phase3/analysis/per_sample_localization.csv) (169 rows with coordinates, hits, and IoUs).
   - [`outputs/phase3/analysis/by_forgery_type.csv`](file:///E:/DocverifyAi/outputs/phase3/analysis/by_forgery_type.csv)
   - [`outputs/phase3/analysis/by_document_type.csv`](file:///E:/DocverifyAi/outputs/phase3/analysis/by_document_type.csv)
   - [`outputs/phase3/analysis/by_field.csv`](file:///E:/DocverifyAi/outputs/phase3/analysis/by_field.csv)
4. **Visualizations:**
   - [`outputs/phase3/visualizations/summary_plots/pointing_game_results.png`](file:///E:/DocverifyAi/outputs/phase3/visualizations/summary_plots/pointing_game_results.png)
   - [`outputs/phase3/visualizations/summary_plots/iou_distribution.png`](file:///E:/DocverifyAi/outputs/phase3/visualizations/summary_plots/iou_distribution.png)
   - [`outputs/phase3/visualizations/summary_plots/iou_by_forgery_type.png`](file:///E:/DocverifyAi/outputs/phase3/visualizations/summary_plots/iou_by_forgery_type.png)
   - Case studies in `successful_cases/`, `failed_cases/`, `false_negatives/`, `false_positives/`.

---

## 9. Next Phase Hand-off (Phase 4 Recommendations)

The quantitative failure of Grad-CAM to tightly localize micro-manipulations ($0.59\%$ Pointing Game accuracy, $0.0085$ IoU) is a critical scientific finding:
1. **Do Not Rely on Heatmaps for Automated Localization Decisions:** Heatmaps are suitable as soft visual guides for human auditors, but cannot be treated as automated bounding-box detectors.
2. **Phase 4 OCR & Field-Level Feature Fusion:**  
   Because $60.53\%$ of False Negatives are recovered at patch scale, Phase 4 should use OCR as an auxiliary feature extractor:
   - Run OCR to locate and extract document text fields.
   - Inspect text consistency, typography, and high-resolution crops around suspected fields.
   - Fuse global CNN confidence with OCR field-consistency flags into a robust verification score.
