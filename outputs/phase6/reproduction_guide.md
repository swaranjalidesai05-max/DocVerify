# DocVerify — Phase 6 Complete Reproduction Guide

This guide describes how to reproduce every experiment, benchmark, statistical test, and artifact in Phase 6 from scratch.

## Environment Prerequisites

- Python 3.10+ (Current runtime: Python 3.13)
- PyTorch (CPU or CUDA)
- torchvision
- scikit-learn
- pandas, numpy, scipy
- Pillow
- reportlab (for PDF documentation)
- EasyOCR (optional for live text extraction; historical features cached)

## Directory Layout

```
E:\DocverifyAi\
├── Dataset/                     # Original SIDTD dataset (read-only)
├── models/
│   ├── phase2/best_model.pth    # Frozen global baseline checkpoint
│   └── phase6/dedicated_patch_model.pth # Dedicated local patch checkpoint
├── outputs/
│   ├── dataset_manifest/dataset_manifest.csv # Master dataset manifest (2,222 records)
│   ├── phase4/evidence/per_sample_evidence.csv # Phase 4 evidence
│   └── phase6/                  # Phase 6 outputs & benchmarks
└── scripts/                     # Standalone Phase 6 experiment scripts
```

## Step-by-Step Reproduction Pipeline

### Step 1: Create Frozen Test Split Manifest
```bash
python scripts/create_frozen_test_manifest.py
```
*Output:* `outputs/phase6/frozen_test_manifest.csv` (320 samples: 151 Real, 169 Fake, 139 disjoint clusters, all 10 templates).

### Step 2: Run Cryptographic & Perceptual Data Leakage Audit
```bash
python scripts/leakage_audit.py
```
*Output:* `outputs/phase6/oracle_leakage_audit.md`, `outputs/phase6/leakage_audit.json`. Confirms 0 duplicate records, 0 cross-split cluster leaks, 0 hash collisions.

### Step 3: Run Non-Semantic Shortcut / Bias Audit
```bash
python scripts/shortcut_audit.py
```
*Output:* `outputs/phase6/shortcut_results.json`. Audits metadata, CMYK color space, and basic color statistics.

### Step 4: Evaluate Preprocessing & Geometry Vulnerability
```bash
python scripts/evaluate_preprocessing.py
```
*Output:* `outputs/phase6/preprocessing_ablation.json`. Compares CenterCrop 224, Letterbox 224, Letterbox 384, and Standardized JPEG Q=85 recompression.

### Step 5: Train Dedicated Local Patch Verification Model
```bash
python scripts/train_patch_model.py
```
*Output:* `models/phase6/dedicated_patch_model.pth`, `outputs/phase6/patch_model_metrics.json`. Evaluates native contextual patch classifier (EfficientNet-B0).

### Step 6: Evaluate Production-Equivalent Tiled & Canonical Prior Inspection
```bash
python scripts/evaluate_tiled_inspection.py
```
*Output:* `outputs/phase6/tiled_inspection_results.json`, `outputs/phase6/tiled_inspection_predictions.csv`. Tests canonical prior field inspection and sliding-window tiling (MAX, MEAN, TOP-3 aggregation) without ground-truth boxes.

### Step 7: Extract Independent Forensic Features (ELA, Sensor Noise, Edge Energy)
```bash
python scripts/extract_forensic_features.py
```
*Output:* `outputs/phase6/forensic_features_val.csv`, `outputs/phase6/forensic_features_test.csv`, `outputs/phase6/forensic_feature_distributions.json`.

### Step 8: Multi-Modal Evidence Fusion & Operational Calibration
```bash
python scripts/fuse_evidence.py
```
*Output:* `outputs/phase6/fusion_weights.json`, `outputs/phase6/calibrated_fusion_results.json`, `outputs/phase6/fusion_predictions_test.csv`, `outputs/phase6/operational_thresholds.json`.

### Step 9: Statistical Validation & 5,000 Bootstrap Resampling
```bash
python scripts/statistical_validation.py
```
*Output:* `outputs/phase6/statistical_validation.json`, `outputs/phase6/per_template_generalization.csv`. Computes 95% bootstrap CIs, McNemar's paired test ($\chi^2 = 32.09, p = 1.47 \times 10^{-8}$), and per-template breakdowns.

### Step 10: Environmental Stress Testing (Robustness Curves)
```bash
python scripts/stress_testing.py
```
*Output:* `outputs/phase6/robustness_stress_testing.json`, `outputs/phase6/robustness_curves.csv`. Evaluates JPEG compression ($Q \in [30..95]$), blur ($\sigma \in [0.5..3.0]$), noise ($\sigma \in [5..25]$), and scaling.

### Step 11: Failure Mode & Forensic Error Analysis
```bash
python scripts/error_analysis.py
```
*Output:* `outputs/phase6/error_analysis.csv`, `outputs/phase6/error_analysis_summary.json`. Catalogs all 19 False Negatives and 7 False Positives with root causes.

### Step 12: Consolidate Master Benchmark Matrix
```bash
python scripts/consolidate_phase6.py
```
*Output:* `outputs/phase6/experiment_matrix.csv`, `outputs/phase6/phase6_results.json`.
