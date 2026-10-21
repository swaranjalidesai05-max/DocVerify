#!/usr/bin/env python3
"""
DocVerify - Phase 4 High-Resolution Patch Verification Engine
Author: Senior ML & Computer Vision Research Engineer

Purpose:
1. Takes a full document image and candidate field coordinates (x, y, w, h).
2. Extracts a localized high-resolution crop (with contextual padding).
3. Applies model-compatible preprocessing (Resize 256, CenterCrop 224, ImageNet norm)
   to ensure the manipulated field occupies a large, high-frequency spatial scale
   rather than being downsampled to sub-pixel noise.
4. Queries the frozen Phase 2 EfficientNet-B0 classifier to obtain P_patch(FAKE).
5. Compares P_patch against P_global and computes recovery rates for False Negatives.
"""

from typing import Tuple, Dict, Any, Optional
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms


class PatchVerificationEngine:
    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        
        self.eval_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def extract_patch(
        self,
        pil_image: Image.Image,
        bbox: Dict[str, Any],
        context_multiplier: float = 2.5,
        min_dim: int = 128
    ) -> Tuple[Image.Image, Tuple[int, int, int, int]]:
        """
        Extracts high-resolution contextual crop centered on bbox.
        bbox contains: {'x': int, 'y': int, 'width': int, 'height': int}
        """
        orig_w, orig_h = pil_image.size
        gx = bbox.get("x", 0)
        gy = bbox.get("y", 0)
        gw = bbox.get("width", 0)
        gh = bbox.get("height", 0)
        
        cx, cy = gx + gw / 2.0, gy + gh / 2.0
        crop_dim = max(int(max(gw, gh) * context_multiplier), min_dim)
        
        crop_x1 = max(0, int(cx - crop_dim / 2.0))
        crop_y1 = max(0, int(cy - crop_dim / 2.0))
        crop_x2 = min(orig_w, crop_x1 + crop_dim)
        crop_y2 = min(orig_h, crop_y1 + crop_dim)
        
        crop_patch = pil_image.crop((crop_x1, crop_y1, crop_x2, crop_y2))
        return crop_patch, (crop_x1, crop_y1, crop_x2, crop_y2)

    def evaluate_patch(
        self,
        patch_image: Image.Image
    ) -> Dict[str, Any]:
        """Runs the frozen classifier on the high-resolution patch."""
        tensor = self.eval_transform(patch_image.convert("RGB")).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            
        prob_real = float(probs[0])
        prob_fake = float(probs[1])
        pred_class = int(np.argmax(probs))
        
        return {
            "patch_prob_real": round(prob_real, 4),
            "patch_prob_fake": round(prob_fake, 4),
            "patch_pred_class": pred_class
        }
