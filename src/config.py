#!/usr/bin/env python3
"""
DocVerify - Centralized Project Configuration
Author: Senior ML, Computer Vision & Full-Stack AI Research Engineer

Provides unified, dynamic path resolution, model checkpoints, threshold constants,
template parameters, and logging configurations across all modules.
Eliminates all hardcoded filesystem paths.
"""

import os
from pathlib import Path
import logging

# Dynamically locate project root (2 levels up from src/config.py)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Dataset paths
DATASET_DIR = PROJECT_ROOT / "Dataset"
TEMPLATE_DIR = DATASET_DIR / "template"
IMAGES_DIR = TEMPLATE_DIR / "Images"
ANNOTATIONS_DIR = TEMPLATE_DIR / "Annotations"

# Manifest and outputs paths
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
MANIFEST_DIR = OUTPUTS_DIR / "dataset_manifest"
MANIFEST_PATH = MANIFEST_DIR / "dataset_manifest.csv"

# Phase directories
PHASE1_OUTPUTS = OUTPUTS_DIR / "dataset_audit"
PHASE2_OUTPUTS = OUTPUTS_DIR / "phase2"
PHASE3_OUTPUTS = OUTPUTS_DIR / "phase3"
PHASE4_OUTPUTS = OUTPUTS_DIR / "phase4"
PHASE5_OUTPUTS = OUTPUTS_DIR / "phase5"
PHASE6_OUTPUTS = OUTPUTS_DIR / "phase6"

# Model paths
MODELS_DIR = PROJECT_ROOT / "models"
PHASE2_MODEL_PATH = MODELS_DIR / "phase2" / "best_model.pth"
PHASE6_MODELS_DIR = MODELS_DIR / "phase6"
PATCH_MODEL_PATH = PHASE6_MODELS_DIR / "dedicated_patch_model.pth"

# Phase 6 Artifact Paths
FROZEN_TEST_MANIFEST_PATH = PHASE6_OUTPUTS / "frozen_test_manifest.csv"
LEAKAGE_AUDIT_REPORT = PHASE6_OUTPUTS / "oracle_leakage_audit.md"
SHORTCUT_RESULTS_JSON = PHASE6_OUTPUTS / "shortcut_results.json"
EXPERIMENT_CONFIG_PATH = PHASE6_OUTPUTS / "experiment_config.json"
EXPERIMENT_MATRIX_PATH = PHASE6_OUTPUTS / "experiment_matrix.csv"
PHASE6_RESULTS_JSON = PHASE6_OUTPUTS / "phase6_results.json"
PHASE6_REPORT_MD = PHASE6_OUTPUTS / "phase6_report.md"
PHASE6_HANDOVER_PDF = PHASE6_OUTPUTS / "DocVerify_Phase6_Handover_Report.pdf"
REPRODUCTION_GUIDE_MD = PHASE6_OUTPUTS / "reproduction_guide.md"

# Random Seeds
RANDOM_SEED = 42

# Image Processing Standards
TARGET_IMAGE_SIZE = (224, 224)
HIGH_RES_SIZE = (384, 384)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Logging Configuration
def setup_logger(name: str = "DocVerify", log_file: Path = None, level=logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(level)
        formatter = logging.Formatter("[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        
        # Stream handler
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        
        # File handler if specified
        if log_file:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            fh = logging.FileHandler(str(log_file), encoding="utf-8")
            fh.setFormatter(formatter)
            logger.addHandler(fh)
            
    return logger
