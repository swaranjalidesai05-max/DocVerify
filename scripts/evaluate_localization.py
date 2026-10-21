#!/usr/bin/env python3
"""
DocVerify - Phase 3 Localization Evaluation Engine
Author: Senior ML & Computer Vision Research Engineer

Metrics Implemented:
1. Pointing Game:
   - Hit = 1 if argmax(CAM) falls inside the ground-truth manipulated bounding box [x_min, y_min, x_max, y_max].
   - Hit = 0 otherwise.
   - Pointing Game Accuracy = Sum(Hits) / N.
2. Bounding-Box IoU:
   - Binary mask = CAM >= threshold * max(CAM).
   - Minimum enclosing bounding box of the thresholded activation region.
   - IoU = Area(Pred ∩ GT) / Area(Pred ∪ GT).
   - Computes: Mean IoU, Median IoU, Fraction(IoU > 0), IoU >= 0.3, IoU >= 0.5.
3. Multi-Threshold Sweep on Validation Set:
   - Evaluates thresholds [0.20, 0.30, 0.40, 0.50, 0.60, 0.70].
   - Selects optimal validation threshold and evaluates frozen on Test Set.
"""

from typing import Dict, Any, Tuple, List, Optional
import json
import numpy as np


def compute_pointing_game(
    cam_orig: np.ndarray,
    gt_bbox: Dict[str, Any]
) -> Tuple[int, Tuple[int, int]]:
    """
    Computes Pointing Game hit/miss on original image coordinate space.
    gt_bbox contains: {'x': int, 'y': int, 'width': int, 'height': int}
    """
    max_idx = np.unravel_index(np.argmax(cam_orig), cam_orig.shape)
    max_y, max_x = int(max_idx[0]), int(max_idx[1])
    
    gx = gt_bbox.get("x", 0)
    gy = gt_bbox.get("y", 0)
    gw = gt_bbox.get("width", 0)
    gh = gt_bbox.get("height", 0)
    
    in_box = (gx <= max_x <= gx + gw) and (gy <= max_y <= gy + gh)
    hit = 1 if in_box else 0
    return hit, (max_x, max_y)


def extract_pred_bbox(
    cam_orig: np.ndarray,
    threshold: float = 0.5
) -> Tuple[Optional[Tuple[int, int, int, int]], float]:
    """
    Thresholds CAM and computes minimum enclosing bounding box [x_min, y_min, x_max, y_max].
    Returns (pred_bbox, mask_area_ratio).
    """
    max_val = np.max(cam_orig)
    if max_val <= 0:
        return None, 0.0
        
    mask = cam_orig >= (threshold * max_val)
    if not np.any(mask):
        return None, 0.0
        
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    
    # [x_min, y_min, x_max, y_max]
    pred_bbox = (int(cmin), int(rmin), int(cmax), int(rmax))
    mask_area_ratio = float(np.sum(mask) / (cam_orig.shape[0] * cam_orig.shape[1]))
    return pred_bbox, mask_area_ratio


def compute_iou(
    pred_bbox: Optional[Tuple[int, int, int, int]],
    gt_bbox: Dict[str, Any]
) -> float:
    """
    Computes standard 2D Intersection over Union between pred [x1, y1, x2, y2]
    and gt {'x': x, 'y': y, 'width': w, 'height': h}.
    """
    if pred_bbox is None:
        return 0.0
        
    px1, py1, px2, py2 = pred_bbox
    gx1 = gt_bbox.get("x", 0)
    gy1 = gt_bbox.get("y", 0)
    gx2 = gx1 + gt_bbox.get("width", 0)
    gy2 = gy1 + gt_bbox.get("height", 0)
    
    # Intersection coordinates
    ix1 = max(px1, gx1)
    iy1 = max(py1, gy1)
    ix2 = min(px2, gx2)
    iy2 = min(py2, gy2)
    
    inter_w = max(0, ix2 - ix1)
    inter_h = max(0, iy2 - iy1)
    inter_area = inter_w * inter_h
    
    if inter_area == 0:
        return 0.0
        
    pred_area = (px2 - px1) * (py2 - py1)
    gt_area = (gx2 - gx1) * (gy2 - gy1)
    union_area = pred_area + gt_area - inter_area
    
    if union_area <= 0:
        return 0.0
        
    return float(inter_area / union_area)
