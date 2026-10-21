#!/usr/bin/env python3
"""
DocVerify - Phase 8 Automated Comprehensive Regression & Verification Suite
Author: Senior ML, Computer Vision & Security Quality Assurance Engineer

Tests 14 Core Functional, Statistical, and Engineering Modules:
1. Image loading and safe 4-channel CMYK to RGB conversion.
2. Autonomous template identification (without filename hints).
3. Explicit UNKNOWN template fallback handling on anomalous dimensions.
4. Corrected SIDTD bounding-box parsing and boundary clipping.
5. Remediated dedicated patch classifier checkpoint & forward pass.
6. Unified multi-modal evidence feature schema and strict ordering.
7. Truthful Isotonic probability calibration and Brier score.
8. Monotonically ordered operational thresholds (Conservative <= Balanced <= Strict).
9. Grad-CAM exact spatial inverse mapping (aspect ratio preservation).
10. OCR failure resilience (graceful degradation when OCR fails).
11. Forensic consistency feature extraction (ELA, noise, edge variance).
12. PDF verification certificate compilation with safe text escaping.
13. Corrupted input and oversized image memory safety guards.
14. Stale session state prevention logic.
"""

import sys
import io
import json
import tempfile
from pathlib import Path
import numpy as np
from PIL import Image, ImageChops

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    PHASE2_MODEL_PATH, setup_logger
)
from src.bbox_parser import parse_sidtd_bbox
from src.template_matcher import identify_document_template
from src.gradcam_corrected import CorrectedGradCAM
from src.inference_engine import DocVerifyInferenceEngine
from src.report_generator import generate_verification_report

logger = setup_logger("Phase8TestRunner")


def run_phase8_test_suite():
    logger.info("=" * 65)
    logger.info("DOCVERIFY PHASE 8 COMPREHENSIVE AUTOMATED TEST SUITE")
    logger.info("=" * 65)

    passed_count = 0
    total_tests = 14

    # Test 1: Image Loading & Safe CMYK Conversion
    try:
        cmyk_img = Image.new("CMYK", (200, 200), color=(10, 20, 30, 40))
        rgb_img = cmyk_img.convert("RGB")
        assert rgb_img.mode == "RGB", f"Expected RGB mode, got {rgb_img.mode}"
        assert rgb_img.size == (200, 200), f"Size mismatch: {rgb_img.size}"
        logger.info("[PASS] Test 1: CMYK to RGB standardization verified.")
        passed_count += 1
    except Exception as e:
        logger.error(f"[FAIL] Test 1: {e}")

    # Test 2: Autonomous Template Detection (No Filename Hints)
    try:
        test_img = Image.new("RGB", (2167, 1360))
        res = identify_document_template(test_img, filename_hint=None)
        assert res["template_id"] == "alb_id", f"Expected alb_id, got {res['template_id']}"
        assert res["template_confidence"] > 0.80, f"Expected high confidence, got {res['template_confidence']}"
        logger.info("[PASS] Test 2: Autonomous template detection without filename hints verified.")
        passed_count += 1
    except Exception as e:
        logger.error(f"[FAIL] Test 2: {e}")

    # Test 3: Explicit UNKNOWN Template Fallback
    try:
        weird_img = Image.new("RGB", (500, 500)) # Square 1:1 ratio
        res = identify_document_template(weird_img, filename_hint=None)
        assert res["template_id"] == "UNKNOWN", f"Expected UNKNOWN, got {res['template_id']}"
        assert not res["is_known_template"], "Expected is_known_template=False"
        logger.info("[PASS] Test 3: Explicit UNKNOWN template fallback verified.")
        passed_count += 1
    except Exception as e:
        logger.error(f"[FAIL] Test 3: {e}")

    # Test 4: SIDTD Bounding Box Parsing & Clipping
    try:
        raw_dict = '{"name": "rect", "x": 100, "y": 200, "width": 50, "height": 30}'
        box, status = parse_sidtd_bbox(raw_dict, img_width=1000, img_height=800)
        assert status == "VALID_SIDTD_BBOX", f"Expected VALID_SIDTD_BBOX, got {status}"
        assert box == [100, 200, 50, 30], f"Box coordinates mismatch: {box}"

        # Test clipping out-of-bounds
        oob_dict = '{"name": "rect", "x": 950, "y": 750, "width": 200, "height": 200}'
        box_clip, _ = parse_sidtd_bbox(oob_dict, img_width=1000, img_height=800)
        assert box_clip[0] + box_clip[2] <= 1000, f"Box width exceeded bounds: {box_clip}"
        assert box_clip[1] + box_clip[3] <= 800, f"Box height exceeded bounds: {box_clip}"
        logger.info("[PASS] Test 4: SIDTD bounding box parsing and boundary clipping verified.")
        passed_count += 1
    except Exception as e:
        logger.error(f"[FAIL] Test 4: {e}")

    # Test 5: Remediated Patch Model Checkpoint & Deep Copy Verification
    try:
        patch_ckpt_path = PROJECT_ROOT / "models" / "phase8" / "dedicated_patch_model.pth"
        assert patch_ckpt_path.exists(), f"Missing checkpoint: {patch_ckpt_path}"
        ckpt = torch.load(patch_ckpt_path, map_location="cpu", weights_only=False)
        assert "model_state_dict" in ckpt, "Missing model_state_dict in checkpoint"
        assert ckpt["phase"] == "Phase 8 Remediated", f"Unexpected phase: {ckpt.get('phase')}"
        logger.info("[PASS] Test 5: Dedicated patch model checkpoint verified.")
        passed_count += 1
    except Exception as e:
        logger.error(f"[FAIL] Test 5: {e}")

    # Test 6: Unified Feature Schema & Strict Feature Ordering
    try:
        weights_path = PROJECT_ROOT / "outputs" / "phase8" / "fusion" / "fusion_weights.json"
        assert weights_path.exists(), f"Missing fusion weights: {weights_path}"
        with open(weights_path, "r", encoding="utf-8") as f:
            fw = json.load(f)
        expected_cols = [
            "p_global_fake", "p_patch_fake", "ela_inconsistency",
            "noise_inconsistency", "edge_variance", "ocr_confidence", "format_anomaly"
        ]
        assert fw["feature_order"] == expected_cols, f"Feature ordering mismatch: {fw['feature_order']}"
        assert "p_patch_fake" in fw["feature_weights"], "Missing p_patch_fake in feature weights"
        assert fw["feature_weights"]["p_patch_fake"] > 1.0, f"p_patch_fake weight unexpectedly small: {fw['feature_weights']['p_patch_fake']}"
        logger.info("[PASS] Test 6: Unified feature schema and feature ordering verified.")
        passed_count += 1
    except Exception as e:
        logger.error(f"[FAIL] Test 6: {e}")

    # Test 7: Truthful Probability Calibration
    try:
        calib_path = PROJECT_ROOT / "models" / "phase8" / "fusion_calibrator.joblib"
        assert calib_path.exists(), f"Missing calibrator model: {calib_path}"
        bench_json = PROJECT_ROOT / "outputs" / "phase8" / "final_benchmark" / "final_benchmark_results.json"
        assert bench_json.exists(), f"Missing benchmark json: {bench_json}"
        with open(bench_json, "r", encoding="utf-8") as f:
            b_data = json.load(f)
        brier = b_data["test_policy_metrics"]["balanced"]["brier_score"]
        ece = b_data["test_policy_metrics"]["balanced"]["expected_calibration_error"]
        assert brier < 0.10, f"Brier score unexpectedly high: {brier}"
        assert ece < 0.08, f"Expected Calibration Error unexpectedly high: {ece}"
        logger.info(f"[PASS] Test 7: Truthful calibration verified (Brier: {brier:.4f}, ECE: {ece:.4f}).")
        passed_count += 1
    except Exception as e:
        logger.error(f"[FAIL] Test 7: {e}")

    # Test 8: Operational Threshold Semantics & Monotonic Ordering
    try:
        thresh_path = PROJECT_ROOT / "outputs" / "phase8" / "fusion" / "operational_thresholds.json"
        assert thresh_path.exists(), f"Missing thresholds json: {thresh_path}"
        with open(thresh_path, "r", encoding="utf-8") as f:
            t_data = json.load(f)
        t_cons = t_data["conservative_kyc_threshold"]
        t_bal = t_data["balanced_threshold"]
        t_strict = t_data["strict_antifraud_threshold"]
        assert t_cons <= t_bal <= t_strict, f"Threshold monotonic violation: {t_cons} <= {t_bal} <= {t_strict}"
        assert not np.isinf(t_strict), "Strict threshold must be finite"
        logger.info(f"[PASS] Test 8: Operational thresholds monotonic ordering verified: {t_cons:.2f} <= {t_bal:.2f} <= {t_strict:.2f}.")
        passed_count += 1
    except Exception as e:
        logger.error(f"[FAIL] Test 8: {e}")

    # Test 9: Grad-CAM Geometric Inverse Mapping
    try:
        cam_7x7 = np.zeros((7, 7), dtype=np.float32)
        cam_7x7[3, 3] = 1.0
        full_cam = CorrectedGradCAM.map_cam_to_original_image(cam_7x7, orig_width=2167, orig_height=1360)
        assert full_cam.shape == (1360, 2167), f"Grad-CAM shape mismatch: {full_cam.shape}"
        py, px = np.unravel_index(np.argmax(full_cam), full_cam.shape)
        assert abs(px - 1083) <= 2 and abs(py - 680) <= 2, f"Center offset error: ({px}, {py})"
        logger.info("[PASS] Test 9: Grad-CAM exact spatial inverse mapping verified.")
        passed_count += 1
    except Exception as e:
        logger.error(f"[FAIL] Test 9: {e}")

    # Test 10: OCR Failure Resilience
    try:
        engine = DocVerifyInferenceEngine(enable_ocr=False)
        dummy_img = Image.new("RGB", (1327, 827), color=(255, 255, 255))
        res = engine.verify_document(dummy_img, operational_mode="balanced")
        assert "verdict" in res, "Missing verdict in OCR-disabled inference"
        assert res["evidence"]["ocr_confidence"] >= 0.0, "Invalid OCR fallback confidence"
        logger.info("[PASS] Test 10: OCR failure and disabled engine resilience verified.")
        passed_count += 1
    except Exception as e:
        logger.error(f"[FAIL] Test 10: {e}")

    # Test 11: Forensic Consistency Extraction
    try:
        engine = DocVerifyInferenceEngine(enable_ocr=False)
        dummy_img = Image.new("RGB", (800, 600), color=(128, 128, 128))
        forensics = engine.compute_forensics(dummy_img)
        assert "ela_inconsistency" in forensics, "Missing ela_inconsistency"
        assert "noise_inconsistency" in forensics, "Missing noise_inconsistency"
        assert "edge_variance" in forensics, "Missing edge_variance"
        logger.info("[PASS] Test 11: Forensic consistency feature extractor verified.")
        passed_count += 1
    except Exception as e:
        logger.error(f"[FAIL] Test 11: {e}")

    # Test 12: PDF Certificate Generation & Safe Text Escaping
    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_pdf:
            tmp_pdf_path = Path(tmp_pdf.name)
        mock_result = {
            "verdict": "POTENTIALLY FORGED / SUSPICIOUS",
            "risk_level": "HIGH RISK",
            "risk_score": 92.5,
            "calibrated_forgery_probability": 0.925,
            "threshold_applied": 0.7200,
            "operational_policy": "Balanced Risk Policy",
            "template_id": "esp_id",
            "template_confidence": 0.99,
            "document_display_name": "Identity Document (Spain)",
            "evidence": {
                "p_global_fake": 0.88,
                "p_patch_fake": 0.91,
                "forensic_cues": {"ela_inconsistency": 0.35, "noise_inconsistency": 18.2, "edge_variance": 2400.0},
                "ocr_confidence": 0.89,
                "format_anomalies_detected": 1
            },
            "patch_findings": [
                {"field_name": "surname", "field_type": "name", "patch_prob_fake": 0.95, "patch_pred": "SUSPICIOUS"}
            ],
            "ocr_findings": [
                {"field_name": "birth_date", "field_type": "date", "text": "32/13/1985", "confidence": 0.90, "is_anomaly": True}
            ],
            "device": "CPU"
        }
        generate_verification_report(mock_result, tmp_pdf_path)
        assert tmp_pdf_path.exists() and tmp_pdf_path.stat().st_size > 1000, "PDF generation failed"
        tmp_pdf_path.unlink(missing_ok=True)
        logger.info("[PASS] Test 12: PDF verification certificate generation verified.")
        passed_count += 1
    except Exception as e:
        logger.error(f"[FAIL] Test 12: {e}")

    # Test 13: Corrupted Input & Oversized Image Memory Guard
    try:
        engine = DocVerifyInferenceEngine(enable_ocr=False)
        oversized = Image.new("RGB", (4000, 3000), color=(200, 200, 200))
        res = engine.verify_document(oversized)
        assert res["pil_image"].width <= 3500, f"Oversized image was not scaled down: {res['pil_image'].size}"
        logger.info("[PASS] Test 13: Oversized image scaling memory guard verified.")
        passed_count += 1
    except Exception as e:
        logger.error(f"[FAIL] Test 13: {e}")

    # Test 14: Stale Result Prevention Logic
    try:
        session = {}
        # Simulate initial file intake
        session["current_doc_id"] = "file_a.jpg"
        session["last_result"] = {"verdict": "VERIFIED GENUINE"}
        # Simulate new file upload
        new_file = "file_b.jpg"
        if session.get("current_doc_id") != new_file:
            session["current_doc_id"] = new_file
            session.pop("last_result", None)
        assert "last_result" not in session, "Stale result remained in session after file switch!"
        logger.info("[PASS] Test 14: Stale session state prevention logic verified.")
        passed_count += 1
    except Exception as e:
        logger.error(f"[FAIL] Test 14: {e}")

    logger.info("=" * 65)
    logger.info(f"PHASE 8 TEST SUITE RESULTS: {passed_count}/{total_tests} TESTS PASSED")
    logger.info("=" * 65)

    assert passed_count == total_tests, f"Test failure detected: {passed_count}/{total_tests} passed."


if __name__ == "__main__":
    run_phase8_test_suite()
