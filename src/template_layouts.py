#!/usr/bin/env python3
"""
DocVerify - Canonical Template Field Layout Database
Author: Senior ML & Computer Vision Research Engineer

Purpose:
Stores the canonical bounding box layout proportions and field schemas
derived from the 10 SIDTD document templates. During production deployment,
when a user uploads a document and the template is identified (e.g. alb_id),
these canonical coordinates generate automatic candidate field crops without
needing ground-truth manipulation coordinates.
"""

from typing import Dict, Any, List

# User-friendly display names mapping for UI presentation
TEMPLATE_DISPLAY_NAMES: Dict[str, str] = {
    "alb_id": "Identity Document",
    "aze_passport": "Passport",
    "esp_id": "Identity Document",
    "est_id": "Identity Document",
    "fin_id": "Identity Document",
    "grc_passport": "Passport",
    "lva_passport": "Passport",
    "rus_internalpassport": "Internal Passport",
    "srb_passport": "Passport",
    "svk_id": "Identity Document",
    "svk_passport": "Passport",
}

def get_template_display_name(template_id: str) -> str:
    """Returns human-readable document type name for UI while preserving internal ID."""
    return TEMPLATE_DISPLAY_NAMES.get(template_id, "Identity Document")

# Normalized bounding box coordinates [x_ratio, y_ratio, w_ratio, h_ratio]
CANONICAL_TEMPLATE_LAYOUTS: Dict[str, Dict[str, Any]] = {
    "alb_id": {
        "country": "Albania",
        "doc_type": "Identity Card",
        "aspect_ratio": 1.593, # 2167 x 1360
        "fields": {
            "surname": {"norm_bbox": [0.324, 0.178, 0.088, 0.040], "type": "name"},
            "name": {"norm_bbox": [0.325, 0.268, 0.081, 0.035], "type": "name"},
            "nationality": {"norm_bbox": [0.325, 0.362, 0.267, 0.038], "type": "text"},
            "birth_date": {"norm_bbox": [0.327, 0.535, 0.166, 0.038], "type": "date"},
            "expiry_date": {"norm_bbox": [0.672, 0.622, 0.183, 0.032], "type": "date"},
            "id_number": {"norm_bbox": [0.670, 0.713, 0.192, 0.032], "type": "alphanumeric"},
            "number": {"norm_bbox": [0.671, 0.361, 0.159, 0.038], "type": "numeric"}
        }
    },
    "aze_passport": {
        "country": "Azerbaijan",
        "doc_type": "Passport",
        "aspect_ratio": 1.421, # 1519 x 1069
        "fields": {
            "surname": {"norm_bbox": [0.342, 0.243, 0.185, 0.045], "type": "name"},
            "name": {"norm_bbox": [0.342, 0.337, 0.185, 0.045], "type": "name"},
            "nationality": {"norm_bbox": [0.342, 0.430, 0.150, 0.045], "type": "text"},
            "birth_date": {"norm_bbox": [0.342, 0.524, 0.150, 0.045], "type": "date"},
            "expiry_date": {"norm_bbox": [0.553, 0.617, 0.150, 0.045], "type": "date"},
            "number": {"norm_bbox": [0.737, 0.150, 0.184, 0.045], "type": "alphanumeric"}
        }
    },
    "esp_id": {
        "country": "Spain",
        "doc_type": "DNI / National ID",
        "aspect_ratio": 1.605, # 1327 x 827
        "fields": {
            "surname": {"norm_bbox": [0.301, 0.181, 0.226, 0.048], "type": "name"},
            "name": {"norm_bbox": [0.301, 0.278, 0.188, 0.048], "type": "name"},
            "birth_date": {"norm_bbox": [0.301, 0.472, 0.151, 0.048], "type": "date"},
            "expiry_date": {"norm_bbox": [0.528, 0.568, 0.151, 0.048], "type": "date"},
            "id_number": {"norm_bbox": [0.603, 0.181, 0.226, 0.048], "type": "alphanumeric"}
        }
    },
    "est_id": {
        "country": "Estonia",
        "doc_type": "Identity Card",
        "aspect_ratio": 1.594, # 931 x 584
        "fields": {
            "surname": {"norm_bbox": [0.290, 0.188, 0.258, 0.051], "type": "name"},
            "name": {"norm_bbox": [0.290, 0.291, 0.258, 0.051], "type": "name"},
            "birth_date": {"norm_bbox": [0.290, 0.497, 0.172, 0.051], "type": "date"},
            "expiry_date": {"norm_bbox": [0.537, 0.600, 0.172, 0.051], "type": "date"},
            "id_number": {"norm_bbox": [0.537, 0.702, 0.215, 0.051], "type": "alphanumeric"}
        }
    },
    "fin_id": {
        "country": "Finland",
        "doc_type": "Identity Card",
        "aspect_ratio": 1.602, # 929 x 580
        "fields": {
            "surname": {"norm_bbox": [0.312, 0.190, 0.237, 0.052], "type": "name"},
            "name": {"norm_bbox": [0.312, 0.293, 0.215, 0.052], "type": "name"},
            "birth_date": {"norm_bbox": [0.312, 0.499, 0.172, 0.052], "type": "date"},
            "expiry_date": {"norm_bbox": [0.538, 0.603, 0.172, 0.052], "type": "date"},
            "id_number": {"norm_bbox": [0.538, 0.707, 0.215, 0.052], "type": "alphanumeric"}
        }
    },
    "grc_passport": {
        "country": "Greece",
        "doc_type": "Passport",
        "aspect_ratio": 1.421, # 2167 x 1525
        "fields": {
            "surname": {"norm_bbox": [0.346, 0.230, 0.231, 0.046], "type": "name"},
            "name": {"norm_bbox": [0.346, 0.328, 0.231, 0.046], "type": "name"},
            "nationality": {"norm_bbox": [0.346, 0.426, 0.185, 0.046], "type": "text"},
            "birth_date": {"norm_bbox": [0.346, 0.525, 0.185, 0.046], "type": "date"},
            "expiry_date": {"norm_bbox": [0.554, 0.623, 0.185, 0.046], "type": "date"},
            "number": {"norm_bbox": [0.738, 0.131, 0.185, 0.046], "type": "alphanumeric"}
        }
    },
    "lva_passport": {
        "country": "Latvia",
        "doc_type": "Passport",
        "aspect_ratio": 1.425, # 2167 x 1521
        "fields": {
            "surname": {"norm_bbox": [0.332, 0.237, 0.231, 0.046], "type": "name"},
            "name": {"norm_bbox": [0.332, 0.335, 0.231, 0.046], "type": "name"},
            "birth_date": {"norm_bbox": [0.332, 0.533, 0.185, 0.046], "type": "date"},
            "expiry_date": {"norm_bbox": [0.554, 0.631, 0.185, 0.046], "type": "date"},
            "number": {"norm_bbox": [0.738, 0.138, 0.185, 0.046], "type": "alphanumeric"}
        }
    },
    "rus_internalpassport": {
        "country": "Russia",
        "doc_type": "Internal Passport",
        "aspect_ratio": 1.421, # 1519 x 1069
        "fields": {
            "surname": {"norm_bbox": [0.450, 0.100, 0.350, 0.060], "type": "name"},
            "name": {"norm_bbox": [0.450, 0.255, 0.350, 0.065], "type": "name"},
            "patronymic": {"norm_bbox": [0.450, 0.330, 0.350, 0.065], "type": "name"},
            "sex": {"norm_bbox": [0.360, 0.415, 0.120, 0.055], "type": "text"},
            "birth_date": {"norm_bbox": [0.580, 0.410, 0.250, 0.060], "type": "date"},
            "birth_place": {"norm_bbox": [0.410, 0.485, 0.400, 0.135], "type": "text"},
            "number": {"norm_bbox": [0.925, 0.200, 0.060, 0.600], "type": "numeric", "rotate": 90}
        }
    },
    "srb_passport": {
        "country": "Serbia",
        "doc_type": "Passport",
        "aspect_ratio": 1.421, # 2167 x 1525
        "fields": {
            "surname": {"norm_bbox": [0.346, 0.246, 0.231, 0.046], "type": "name"},
            "name": {"norm_bbox": [0.346, 0.344, 0.231, 0.046], "type": "name"},
            "birth_date": {"norm_bbox": [0.346, 0.541, 0.185, 0.046], "type": "date"},
            "expiry_date": {"norm_bbox": [0.554, 0.639, 0.185, 0.046], "type": "date"},
            "number": {"norm_bbox": [0.738, 0.148, 0.185, 0.046], "type": "alphanumeric"}
        }
    },
    "svk_id": {
        "country": "Slovakia",
        "doc_type": "Identity Card",
        "aspect_ratio": 1.591, # 929 x 584
        "fields": {
            "surname": {"norm_bbox": [0.301, 0.188, 0.237, 0.051], "type": "name"},
            "name": {"norm_bbox": [0.301, 0.291, 0.215, 0.051], "type": "name"},
            "birth_date": {"norm_bbox": [0.301, 0.497, 0.172, 0.051], "type": "date"},
            "expiry_date": {"norm_bbox": [0.538, 0.600, 0.172, 0.051], "type": "date"},
            "id_number": {"norm_bbox": [0.538, 0.702, 0.215, 0.051], "type": "alphanumeric"}
        }
    }
}
