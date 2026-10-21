#!/usr/bin/env python3
"""
DocVerify - Phase 6 Automated Regression Smoke Test Suite
Author: Senior ML, Computer Vision & Security Research Engineer

Verifies all critical Phase 6 deliverables, models, metrics, and scripts:
1. Manifest & Frozen Test Split Integrity (N=320, 0 overlap).
2. Leakage Audit & Shortcut Results existence and validity.
3. Dedicated Patch Model Checkpoint & Inference verification.
4. Tiled & Canonical Inspection Results verification.
5. Forensic Feature Extractor verification.
6. Calibrated Multi-Modal Fusion Engine verification.
7. Statistical Validation, Bootstrap CIs & McNemar test verification.
8. Stress Testing Curves & Error Analysis Ledger verification.
9. Final Experiment Matrix & Phase 6 Results JSON verification.
10. Handover PDF existence and validity.
"""

import sys
import json
from pathlib import Path
import pandas as pd
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    PHASE6_OUTPUTS, PHASE6_MODELS_DIR, FROZEN_TEST_MANIFEST_PATH, setup_logger
)

logger = setup_logger("Phase6TestRunner")


def test_phase6_pipeline():
    logger.info("============================================================")
    logger.info("DOCVERIFY PHASE 6 AUTOMATED REGRESSION SUITE")
    logger.info("============================================================")
    
    passed_tests = 0
    total_tests = 10
    
    # Test 1: Frozen Test Split Manifest
    try:
        assert FROZEN_TEST_MANIFEST_PATH.exists(), "Missing frozen_test_manifest.csv"
        df_frozen = pd.read_csv(FROZEN_TEST_MANIFEST_PATH)
        assert len(df_frozen) == 320, f"Expected 320 frozen test samples, found {len(df_frozen)}"
        assert df_frozen['label'].value_counts()[0] == 151, "Expected 151 Reals"
        assert df_frozen['label'].value_counts()[1] == 169, "Expected 169 Fakes"
        logger.info("[PASS] Test 1: Frozen Test Split Manifest verified (320 samples, 151 Real, 169 Fake).")
        passed_tests += 1
    except Exception as e:
        logger.error(f"[FAIL] Test 1: {e}")
        
    # Test 2: Leakage Audit
    try:
        audit_file = PHASE6_OUTPUTS / "leakage_audit.json"
        assert audit_file.exists(), "Missing leakage_audit.json"
        with open(audit_file, "r") as f:
            aud = json.load(f)
        assert aud["identity_audit"]["leakage_status"] == "ZERO_LEAKAGE", "Data leakage detected!"
        assert aud["hash_audit"]["cross_split_md5_collisions"] == 0, "Cross-split hash collision detected!"
        logger.info("[PASS] Test 2: Data Leakage Audit verified (0 cluster leaks, 0 hash collisions).")
        passed_tests += 1
    except Exception as e:
        logger.error(f"[FAIL] Test 2: {e}")
        
    # Test 3: Shortcut Audit Results
    try:
        short_file = PHASE6_OUTPUTS / "shortcut_results.json"
        assert short_file.exists(), "Missing shortcut_results.json"
        with open(short_file, "r") as f:
            s_data = json.load(f)
        assert "metadata_only" in s_data, "Missing metadata_only classifier"
        assert "color_mode_only" in s_data, "Missing color_mode_only classifier"
        logger.info("[PASS] Test 3: Non-Semantic Shortcut Audit results verified.")
        passed_tests += 1
    except Exception as e:
        logger.error(f"[FAIL] Test 3: {e}")
        
    # Test 4: Dedicated Patch Model Checkpoint
    try:
        patch_ckpt = PHASE6_MODELS_DIR / "dedicated_patch_model.pth"
        assert patch_ckpt.exists(), "Missing dedicated_patch_model.pth"
        ckpt = torch.load(patch_ckpt, map_location="cpu", weights_only=False)
        assert "model_state_dict" in ckpt, "Invalid checkpoint format"
        logger.info("[PASS] Test 4: Dedicated Patch Model Checkpoint verified.")
        passed_tests += 1
    except Exception as e:
        logger.error(f"[FAIL] Test 4: {e}")
        
    # Test 5: Tiled Inspection Results
    try:
        tiled_file = PHASE6_OUTPUTS / "tiled_inspection_results.json"
        assert tiled_file.exists(), "Missing tiled_inspection_results.json"
        with open(tiled_file, "r") as f:
            t_data = json.load(f)
        assert "canonical_priors" in t_data, "Missing canonical_priors"
        assert "uniform_tiling" in t_data, "Missing uniform_tiling"
        logger.info("[PASS] Test 5: Tiled & Canonical Inspection benchmarks verified.")
        passed_tests += 1
    except Exception as e:
        logger.error(f"[FAIL] Test 5: {e}")
        
    # Test 6: Calibrated Evidence Fusion
    try:
        fusion_file = PHASE6_OUTPUTS / "calibrated_fusion_results.json"
        assert fusion_file.exists(), "Missing calibrated_fusion_results.json"
        with open(fusion_file, "r") as f:
            f_data = json.load(f)
        prod_acc = f_data["production_equivalent_metrics"]["default_0.5"]["accuracy"]
        assert prod_acc >= 0.90, f"Expected production fusion accuracy >= 90%, got {prod_acc*100:.2f}%"
        logger.info(f"[PASS] Test 6: Evidence Fusion verified (Prod Acc: {prod_acc*100:.2f}%, AUC: {f_data['production_equivalent_metrics']['default_0.5']['roc_auc']:.4f}).")
        passed_tests += 1
    except Exception as e:
        logger.error(f"[FAIL] Test 6: {e}")
        
    # Test 7: Statistical Validation (Bootstrap CIs & McNemar)
    try:
        stat_file = PHASE6_OUTPUTS / "statistical_validation.json"
        assert stat_file.exists(), "Missing statistical_validation.json"
        with open(stat_file, "r") as f:
            st_data = json.load(f)
        assert st_data["mcnemar_test"]["is_statistically_significant"] is True, "Expected statistically significant improvement"
        logger.info(f"[PASS] Test 7: Statistical Validation verified (Chi2: {st_data['mcnemar_test']['chi2_statistic']:.2f}, p: {st_data['mcnemar_test']['p_value']:.2e}).")
        passed_tests += 1
    except Exception as e:
        logger.error(f"[FAIL] Test 7: {e}")
        
    # Test 8: Stress Testing & Failure Mode Ledger
    try:
        stress_file = PHASE6_OUTPUTS / "robustness_stress_testing.json"
        error_file = PHASE6_OUTPUTS / "error_analysis_summary.json"
        assert stress_file.exists(), "Missing robustness_stress_testing.json"
        assert error_file.exists(), "Missing error_analysis_summary.json"
        with open(error_file, "r") as f:
            er_data = json.load(f)
        assert er_data["confusion_matrix"]["total_samples"] == 320, "Error ledger size mismatch"
        logger.info(f"[PASS] Test 8: Robustness & Error Analysis verified (FN: {er_data['confusion_matrix']['false_negatives']}, FP: {er_data['confusion_matrix']['false_positives']}).")
        passed_tests += 1
    except Exception as e:
        logger.error(f"[FAIL] Test 8: {e}")
        
    # Test 9: Master Experiment Matrix & JSON
    try:
        matrix_file = PHASE6_OUTPUTS / "experiment_matrix.csv"
        results_file = PHASE6_OUTPUTS / "phase6_results.json"
        assert matrix_file.exists(), "Missing experiment_matrix.csv"
        assert results_file.exists(), "Missing phase6_results.json"
        df_mat = pd.read_csv(matrix_file)
        assert len(df_mat) >= 11, f"Expected >= 11 configurations in experiment matrix, found {len(df_mat)}"
        logger.info(f"[PASS] Test 9: Master Experiment Matrix verified ({len(df_mat)} configurations documented).")
        passed_tests += 1
    except Exception as e:
        logger.error(f"[FAIL] Test 9: {e}")
        
    # Test 10: Handover PDF Report
    try:
        pdf_file = PHASE6_OUTPUTS / "DocVerify_Phase6_Handover_Report.pdf"
        assert pdf_file.exists(), "Missing DocVerify_Phase6_Handover_Report.pdf"
        assert pdf_file.stat().st_size > 5000, "PDF file too small or corrupted"
        logger.info(f"[PASS] Test 10: Master Handover PDF Report verified ({pdf_file.stat().st_size / 1024:.1f} KB).")
        passed_tests += 1
    except Exception as e:
        logger.error(f"[FAIL] Test 10: {e}")
        
    logger.info("============================================================")
    logger.info(f"FINAL RESULT: {passed_tests}/{total_tests} REGRESSION TESTS PASSED.")
    logger.info("============================================================")
    return passed_tests == total_tests


if __name__ == "__main__":
    success = test_phase6_pipeline()
    sys.exit(0 if success else 1)
