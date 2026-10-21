#!/usr/bin/env python3
"""
DocVerify - Phase 3 Grad-CAM & HiRes-CAM Explainability Module
Author: Senior ML & Computer Vision Research Engineer

Mathematical Implementation:
1. Grad-CAM:
   - Forward pass to target layer features.8 (Conv2dNormActivation, 1280 channels).
   - Target class score y^c (c = FAKE = 1).
   - Gradient d(y^c) / d(A^k).
   - Importance weight alpha_k = (1 / Z) * sum_{i, j} (d(y^c) / d(A_{i,j}^k)).
   - L_{Grad-CAM}^c = ReLU(sum_k alpha_k * A^k).
2. HiRes-CAM:
   - Preserves pixel-level element-wise attribution without spatial pooling:
     L_{HiRes-CAM}^c = ReLU(sum_k (d(y^c)/d(A_{i,j}^k) * A_{i,j}^k)).
3. Heatmap normalization & geometric coordinate alignment:
   - Bicubic interpolation of 7x7 activation to 224x224 (preprocessed space)
     and inverse-mapped to original image resolution (W x H) matching ground-truth VIA bounding boxes.
"""

from typing import Tuple, Dict, Any, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms


class ExplainabilityEngine:
    def __init__(self, model: nn.Module, target_layer: nn.Module, device: torch.device):
        self.model = model.to(device)
        self.model.eval()
        self.target_layer = target_layer
        self.device = device
        
        self.activations = []
        self.gradients = []
        
        # Register hooks
        self._fwd_handle = self.target_layer.register_forward_hook(self._forward_hook)
        self._bwd_handle = self.target_layer.register_full_backward_hook(self._backward_hook)
        
        self.imagenet_mean = [0.485, 0.456, 0.406]
        self.imagenet_std = [0.229, 0.224, 0.225]
        
        self.eval_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.imagenet_mean, std=self.imagenet_std)
        ])

    def _forward_hook(self, module, inp, outp):
        self.activations.append(outp)

    def _backward_hook(self, module, grad_in, grad_out):
        self.gradients.append(grad_out[0])

    def clear(self):
        self.activations.clear()
        self.gradients.clear()

    def generate_cams(
        self,
        pil_image: Image.Image,
        target_class: int = 1
    ) -> Dict[str, Any]:
        """
        Executes forward + backward pass and computes both Grad-CAM and HiRes-CAM heatmaps.
        Returns maps at:
        - 224x224 (model input tensor space)
        - W x H (original image space)
        """
        self.clear()
        orig_w, orig_h = pil_image.size
        im_rgb = pil_image.convert("RGB")
        tensor = self.eval_transform(im_rgb).unsqueeze(0).to(self.device)
        
        # Forward pass
        logits = self.model(tensor)
        probs = torch.softmax(logits, dim=1).detach().cpu().numpy()[0]
        pred_class = int(np.argmax(probs))
        
        # Target score
        self.model.zero_grad()
        score = logits[0, target_class]
        score.backward(retain_graph=True)
        
        act = self.activations[0]    # Shape: (1, 1280, 7, 7)
        grad = self.gradients[0]     # Shape: (1, 1280, 7, 7)
        
        # 1. Grad-CAM calculation
        weights = torch.mean(grad, dim=(2, 3), keepdim=True) # (1, 1280, 1, 1)
        grad_cam_7x7 = torch.sum(weights * act, dim=1, keepdim=True) # (1, 1, 7, 7)
        grad_cam_7x7 = F.relu(grad_cam_7x7)
        
        # 2. HiRes-CAM calculation
        hires_cam_7x7 = torch.sum(grad * act, dim=1, keepdim=True) # (1, 1, 7, 7)
        hires_cam_7x7 = F.relu(hires_cam_7x7)
        
        # Interpolate to 224x224 (preprocessed crop space)
        gc_224 = F.interpolate(grad_cam_7x7, size=(224, 224), mode="bicubic", align_corners=False)
        hr_224 = F.interpolate(hires_cam_7x7, size=(224, 224), mode="bicubic", align_corners=False)
        
        gc_224_np = gc_224.squeeze().detach().cpu().numpy()
        hr_224_np = hr_224.squeeze().detach().cpu().numpy()
        
        # Normalize to [0, 1]
        def norm_map(m):
            mx = np.max(m)
            mn = np.min(m)
            if mx - mn > 1e-8:
                return (m - mn) / (mx - mn)
            return np.zeros_like(m)
            
        gc_224_norm = norm_map(gc_224_np)
        hr_224_norm = norm_map(hr_224_np)
        
        # Map back to original image resolution (orig_w, orig_h)
        # Note: Preprocessing was Resize to 256 along shorter/proportional edge followed by CenterCrop(224).
        # To project precisely to original image pixels:
        # Step a: Place 224x224 in center of 256x256 canvas
        # Step b: Resize 256x256 to (orig_w, orig_h)
        canvas_256 = np.zeros((256, 256), dtype=np.float32)
        start_y = (256 - 224) // 2
        start_x = (256 - 224) // 2
        canvas_256[start_y:start_y+224, start_x:start_x+224] = gc_224_norm
        
        # Interpolate canvas to original image size
        gc_orig = np.array(
            Image.fromarray(canvas_256).resize((orig_w, orig_h), resample=Image.Resampling.BICUBIC)
        )
        gc_orig_norm = norm_map(gc_orig)
        
        # Same for HiRes-CAM
        hr_canvas_256 = np.zeros((256, 256), dtype=np.float32)
        hr_canvas_256[start_y:start_y+224, start_x:start_x+224] = hr_224_norm
        hr_orig = np.array(
            Image.fromarray(hr_canvas_256).resize((orig_w, orig_h), resample=Image.Resampling.BICUBIC)
        )
        hr_orig_norm = norm_map(hr_orig)
        
        return {
            "logits": logits.detach().cpu().numpy()[0],
            "probs": probs,
            "prob_real": float(probs[0]),
            "prob_fake": float(probs[1]),
            "pred_class": pred_class,
            "target_class": target_class,
            "gradcam_224": gc_224_norm,
            "gradcam_orig": gc_orig_norm,
            "hirescam_224": hr_224_norm,
            "hirescam_orig": hr_orig_norm,
            "orig_size": (orig_w, orig_h)
        }

    def close(self):
        self._fwd_handle.remove()
        self._bwd_handle.remove()
