#!/usr/bin/env python3
"""
DocVerify - Phase 8 Deterministic & Explainable Template Identification Engine
Author: Senior ML & Computer Vision Research Engineer

Remediation Focus:
1. Resolves AUD-TPL-01: Does NOT rely on filename substring hints.
2. Resolves AUD-TPL-02: Combines geometric aspect ratio, native sensor scale, and OCR keyword anchors.
3. Explicit UNKNOWN state: Returns 'UNKNOWN' with safe generic fallback if distance > threshold.
"""

import re
from typing import Dict, Any, Optional, Tuple, List
from PIL import Image

from src.template_layouts import CANONICAL_TEMPLATE_LAYOUTS, get_template_display_name

TEMPLATE_SPECS: Dict[str, Dict[str, Any]] = {
    "alb_id": {
        "native_w": 2167, "native_h": 1360, "aspect_ratio": 1.5934,
        "doc_type": "Identity Card", "country": "Albania",
        "keywords": ["shqiperise", "leternjoftim", "republika", "albania", "alb", "shtetesia"]
    },
    "aze_passport": {
        "native_w": 2167, "native_h": 1525, "aspect_ratio": 1.4210,
        "doc_type": "Passport", "country": "Azerbaijan",
        "keywords": ["azerbaycan", "respublikasi", "pasport", "aze", "azerbaijan"]
    },
    "esp_id": {
        "native_w": 1327, "native_h": 827, "aspect_ratio": 1.6046,
        "doc_type": "Identity Card", "country": "Spain",
        "keywords": ["espana", "nacional", "identidad", "dni", "esp", "documento", "apellidos"]
    },
    "est_id": {
        "native_w": 929, "native_h": 580, "aspect_ratio": 1.6017,
        "doc_type": "Identity Card", "country": "Estonia",
        "keywords": ["eesti", "isikutunnistus", "estonia", "est", "kodakondsus", "synniaeg"]
    },
    "fin_id": {
        "native_w": 931, "native_h": 584, "aspect_ratio": 1.5942,
        "doc_type": "Identity Card", "country": "Finland",
        "keywords": ["suomi", "finland", "henkilokortti", "identitetskort", "fin", "syntymaaika"]
    },
    "grc_passport": {
        "native_w": 1136, "native_h": 803, "aspect_ratio": 1.4147,
        "doc_type": "Passport", "country": "Greece",
        "keywords": ["hellenic", "republic", "passport", "grc", "elliniki", "diamoni"]
    },
    "lva_passport": {
        "native_w": 2167, "native_h": 1521, "aspect_ratio": 1.4247,
        "doc_type": "Passport", "country": "Latvia",
        "keywords": ["latvijas", "republika", "pase", "passport", "lva", "pilsona"]
    },
    "rus_internalpassport": {
        "native_w": 1519, "native_h": 1069, "aspect_ratio": 1.4210,
        "doc_type": "Internal Passport", "country": "Russia",
        "keywords": ["паспорт", "россия", "российская", "федерация", "rus", "россии"]
    },
    "srb_passport": {
        "native_w": 2163, "native_h": 1488, "aspect_ratio": 1.4536,
        "doc_type": "Passport", "country": "Serbia",
        "keywords": ["srbija", "pasos", "passport", "srb", "република", "србија", "пасош"]
    },
    "svk_id": {
        "native_w": 1134, "native_h": 716, "aspect_ratio": 1.5838,
        "doc_type": "Identity Card", "country": "Slovakia",
        "keywords": ["slovenska", "republika", "obciansky", "preukaz", "svk", "slovakia"]
    }
}


def identify_document_template(
    img: Image.Image,
    ocr_text: Optional[str] = None,
    filename_hint: Optional[str] = None,
    max_distance_threshold: float = 0.35
) -> Dict[str, Any]:
    """
    Identifies document template using geometric aspect ratio, native sensor scale,
    and OCR keyword anchors. Never relies on filename hints.

    Returns:
        dict with template_id, template_confidence, identification_method, display_name, matched_cues.
    """
    w, h = img.size
    aspect_ratio = float(w) / max(1.0, float(h))

    cleaned_ocr = ""
    if ocr_text:
        cleaned_ocr = re.sub(r"[^\w\s]", " ", ocr_text.lower())

    best_tpl = None
    min_dist = float("inf")
    best_cues = []
    ocr_boosted_tpl = None

    # Check for direct OCR keyword matches first
    if cleaned_ocr:
        for tpl_id, meta in TEMPLATE_SPECS.items():
            matches = [kw for kw in meta["keywords"] if kw in cleaned_ocr]
            if len(matches) >= 2:
                ocr_boosted_tpl = tpl_id
                best_cues.append(f"Strong OCR anchor match: {matches}")
                break

    for tpl_id, meta in TEMPLATE_SPECS.items():
        ar_dist = abs(aspect_ratio - meta["aspect_ratio"]) / meta["aspect_ratio"]
        w_dist = abs(w - meta["native_w"]) / float(meta["native_w"])
        h_dist = abs(h - meta["native_h"]) / float(meta["native_h"])

        # Combined distance metric: aspect ratio weighted 2.0x, dimensional scale 1.0x each
        total_dist = 2.0 * ar_dist + w_dist + h_dist

        # If OCR matched this template, apply significant bonus
        if ocr_boosted_tpl == tpl_id:
            total_dist *= 0.25

        if total_dist < min_dist:
            min_dist = total_dist
            best_tpl = tpl_id

    # Compute confidence in [0.0, 1.0] from distance
    # A distance of 0.0 -> 0.99 confidence; distance of 0.20 -> 0.60; distance >= 0.35 -> UNKNOWN
    confidence = max(0.0, min(0.99, 1.0 - (min_dist / 0.50)))

    if min_dist <= max_distance_threshold and best_tpl is not None:
        cues = [
            f"Matched schema {best_tpl} (AR: {aspect_ratio:.3f}, Canvas: {w}x{h})",
            f"Geometric distance: {min_dist:.4f}"
        ] + best_cues
        method = "OCR_AND_GEOMETRIC_CONSENSUS" if ocr_boosted_tpl == best_tpl else "GEOMETRIC_SCALE_MATCH"

        return {
            "template_id": best_tpl,
            "template_confidence": round(confidence, 4),
            "identification_method": method,
            "display_name": get_template_display_name(best_tpl),
            "matched_cues": cues,
            "is_known_template": True
        }
    else:
        # Distance exceeds threshold -> Explicit UNKNOWN state
        return {
            "template_id": "UNKNOWN",
            "template_confidence": round(confidence, 4),
            "identification_method": "UNKNOWN_LOW_CONFIDENCE",
            "display_name": "Standard Identity Document (Generic Schema)",
            "matched_cues": [f"Geometric distance {min_dist:.4f} exceeded threshold {max_distance_threshold:.2f}"],
            "is_known_template": False
        }
