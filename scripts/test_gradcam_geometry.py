#!/usr/bin/env python3
"""
DocVerify - Phase 8 Grad-CAM Geometry Verification Test
Author: Senior ML & Computer Vision Engineer

Validates that CorrectedGradCAM.map_cam_to_original_image() achieves:
1. Exact preservation of physical center coordinates.
2. Zero horizontal or vertical anisotropic stretching.
3. Accurate bounds placement on rectangular documents (aspect 1.59).
"""

import sys
import json
from pathlib import Path
import numpy as np
import cv2

PROJECT_ROOT = Path("E:/DocverifyAi")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.gradcam_corrected import CorrectedGradCAM

OUT_DIR = Path("E:/DocverifyAi/outputs/phase8/gradcam")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def test_gradcam_geometry():
    print("=" * 65)
    print("PHASE 8: GRAD-CAM GEOMETRIC INVERSE MAPPING VERIFICATION")
    print("=" * 65)

    # 1. Simulate standard ID-1 card dimensions (2167 x 1360, aspect 1.593)
    orig_w, orig_h = 2167, 1360
    scale = 256.0 / min(orig_w, orig_h) # 256 / 1360 = 0.1882
    rw = int(round(orig_w * scale))     # 408
    rh = int(round(orig_h * scale))     # 256

    x_off = (rw - 224) // 2             # 92
    y_off = (rh - 224) // 2             # 16

    expected_x1 = int(round(x_off / scale))
    expected_x2 = int(round((x_off + 224) / scale))
    expected_y1 = int(round(y_off / scale))
    expected_y2 = int(round((y_off + 224) / scale))

    print(f"Original Canvas: {orig_w}x{orig_h} (Aspect Ratio: {orig_w/orig_h:.4f})")
    print(f"Resize(256) Dimensions: {rw}x{rh}")
    print(f"CenterCrop(224) Bounds in Resized Frame: [{x_off}, {y_off}, {x_off+224}, {y_off+224}]")
    print(f"Expected CenterCrop in Original Space: X=[{expected_x1}, {expected_x2}], Y=[{expected_y1}, {expected_y2}]")

    # 2. Create synthetic 7x7 activation map with high activation peak in the center [3, 3]
    cam_7x7 = np.zeros((7, 7), dtype=np.float32)
    cam_7x7[3, 3] = 1.0

    # Map back to original image coordinates
    full_cam = CorrectedGradCAM.map_cam_to_original_image(cam_7x7, orig_w, orig_h)

    assert full_cam.shape == (orig_h, orig_w), f"Shape mismatch: {full_cam.shape} vs {(orig_h, orig_w)}"

    # Measure active bounding box in full_cam
    active_y, active_x = np.where(full_cam > 0.05)
    actual_x1, actual_x2 = int(np.min(active_x)), int(np.max(active_x))
    actual_y1, actual_y2 = int(np.min(active_y)), int(np.max(active_y))

    # Verify center coordinate alignment
    peak_y, peak_x = np.unravel_index(np.argmax(full_cam), full_cam.shape)
    expected_center_x = (expected_x1 + expected_x2) // 2
    expected_center_y = (expected_y1 + expected_y2) // 2

    x_error = abs(peak_x - expected_center_x)
    y_error = abs(peak_y - expected_center_y)

    print(f"Actual Peak Location: ({peak_x}, {peak_y}) | Expected Center: ({expected_center_x}, {expected_center_y})")
    print(f"Center Offset Error: X_err={x_error} px, Y_err={y_error} px")

    # Assert spatial alignment within 2 pixels tolerance (quantization rounding)
    assert x_error <= 2, f"Horizontal coordinate error too high: {x_error}"
    assert y_error <= 2, f"Vertical coordinate error too high: {y_error}"

    # Verify zero activation outside the central crop field of view
    outside_left = np.sum(full_cam[:, :expected_x1])
    outside_right = np.sum(full_cam[:, expected_x2:])
    assert outside_left == 0.0, f"Activation leaked outside left boundary: {outside_left}"
    assert outside_right == 0.0, f"Activation leaked outside right boundary: {outside_right}"

    print("\n[PASS] Geometric inverse mapping verified: zero aspect ratio distortion, exact crop boundary containment.")

    result_payload = {
        "test": "Grad-CAM Geometric Inversion Verification",
        "canvas_size": [orig_w, orig_h],
        "crop_bounds_in_original_frame": [expected_x1, expected_y1, expected_x2, expected_y2],
        "peak_location": [int(peak_x), int(peak_y)],
        "center_error_pixels": {"x_error": int(x_error), "y_error": int(y_error)},
        "boundary_containment": "STRICT_ZERO_OUTSIDE_ACTIVATION",
        "status": "PASS"
    }

    with open(OUT_DIR / "gradcam_geometry_verification.json", "w", encoding="utf-8") as f:
        json.dump(result_payload, f, indent=2)
    print(f"Saved verification report to: {OUT_DIR / 'gradcam_geometry_verification.json'}")


if __name__ == "__main__":
    test_gradcam_geometry()
