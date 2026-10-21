#!/usr/bin/env python3
"""
DocVerify - Phase 8 Mathematically Correct Grad-CAM Spatial Inverse Mapping
Author: Senior ML & Computer Vision Research Engineer

Remediation Focus:
Resolves AUD-CAM-01: Corrects spatial coordinate mapping between model input
(Resize(256) -> CenterCrop(224)) and original rectangular document coordinates.

Guarantees:
1. Exact preservation of physical aspect ratio (no horizontal or vertical stretching).
2. Precise inverse mapping of CenterCrop offsets.
3. Heatmap placed at the exact physical pixel coordinates evaluated by the CNN layers.
"""

from typing import Tuple, Optional
import numpy as np
import cv2
import torch
import torch.nn as nn
from PIL import Image


class CorrectedGradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._hook_handles = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        h1 = self.target_layer.register_forward_hook(forward_hook)
        h2 = self.target_layer.register_full_backward_hook(backward_hook)
        self._hook_handles = [h1, h2]

    def remove_hooks(self):
        for h in self._hook_handles:
            h.remove()

    def generate_cam(
        self,
        input_tensor: torch.Tensor,
        target_class: int = 1
    ) -> np.ndarray:
        """
        Computes 7x7 Grad-CAM activation map for the input tensor.
        """
        self.model.eval()
        self.model.zero_grad()

        output = self.model(input_tensor)
        score = output[0, target_class]
        score.backward()

        # Pool gradients across spatial channels (Global Average Pooling)
        weights = torch.mean(self.gradients, dim=[2, 3], keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1).squeeze(0)
        cam = torch.clamp(cam, min=0.0).cpu().numpy()

        # Normalize to [0, 1]
        max_val = np.max(cam)
        if max_val > 0:
            cam = cam / max_val
        return cam

    @staticmethod
    def map_cam_to_original_image(
        cam_2d: np.ndarray,
        orig_width: int,
        orig_height: int
    ) -> np.ndarray:
        """
        Mathematically exact inverse mapping from 7x7 / 224x224 crop back to original image space.

        Pipeline Inversion:
        1. Resize(256): scales shorter edge to 256.
           scale s = 256.0 / min(orig_w, orig_h)
           rw = round(orig_w * s), rh = round(orig_h * s)
        2. CenterCrop(224):
           x_off = (rw - 224) // 2
           y_off = (rh - 224) // 2
        3. Inverse map crop coordinates to original image pixels:
           x1 = int(x_off / s)
           y1 = int(y_off / s)
           x2 = min(orig_width, int((x_off + 224) / s))
           y2 = min(orig_height, int((y_off + 224) / s))
        """
        orig_w = max(1, int(orig_width))
        orig_h = max(1, int(orig_height))

        # Scale factor used by torchvision.transforms.Resize(256)
        scale = 256.0 / min(orig_w, orig_h)
        rw = int(round(orig_w * scale))
        rh = int(round(orig_h * scale))

        # CenterCrop(224) offsets in resized frame
        x_off = max(0, (rw - 224) // 2)
        y_off = max(0, (rh - 224) // 2)

        # Inverse project to original image pixels
        x1 = max(0, int(round(x_off / scale)))
        y1 = max(0, int(round(y_off / scale)))
        x2 = min(orig_w, int(round((x_off + 224) / scale)))
        y2 = min(orig_h, int(round((y_off + 224) / scale)))

        crop_w = max(1, x2 - x1)
        crop_h = max(1, y2 - y1)

        # Resize CAM to exact unscaled crop rectangle
        cam_resized_crop = cv2.resize(cam_2d, (crop_w, crop_h), interpolation=cv2.INTER_LINEAR)

        # Place onto full-sized zero-canvas
        full_canvas = np.zeros((orig_h, orig_w), dtype=np.float32)
        full_canvas[y1:y2, x1:x2] = cam_resized_crop[:(y2 - y1), :(x2 - x1)]

        return full_canvas
