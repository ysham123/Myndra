"""Saliency map generation for medical image analysis."""

import torch
import numpy as np
from PIL import Image
import os
import io
import base64
from typing import Optional

def simple_saliency(
    input_tensor: torch.Tensor,
    score: torch.Tensor,
    out_png: str,
    apply_colormap: bool = False,
) -> str:
    """Generate input-gradient saliency heatmap and return as base64 string.
    
    Args:
        input_tensor: Input image tensor with shape (B, C, H, W)
        score: Model output score to compute gradients from
        out_png: (Deprecated) File path - kept for API compatibility
        apply_colormap: If True, apply hot colormap to grayscale saliency
    
    Returns:
        Base64 encoded PNG string of the heatmap
    """
    # Ensure gradients are enabled
    if not input_tensor.requires_grad:
        input_tensor.requires_grad_(True)
    
    # Compute gradients
    if input_tensor.grad is not None:
        input_tensor.grad.zero_()
        
    try:
        score.backward(retain_graph=True)
    except RuntimeError as e:
        raise RuntimeError(f"Gradient computation failed: {e}")
    
    if input_tensor.grad is None:
        raise ValueError("Gradient is None - ensure tensor is part of computation graph")
    
    # Extract and normalize saliency map
    sal = input_tensor.grad.abs().mean(dim=1)[0]
    sal = sal / (sal.max() + 1e-8)  # Normalize to [0, 1]
    
    # Convert to numpy and scale
    sal_np = (sal.detach().cpu().numpy() * 255).astype(np.uint8)
    
    # Create PIL image
    img = Image.fromarray(sal_np, mode='L')
    
    # Apply colormap if requested
    if apply_colormap:
        img = img.convert('RGB')
        pixels = np.array(img)
        colored = np.zeros((*pixels.shape[:2], 3), dtype=np.uint8)
        # Red-heavy "Hot" map for clinical attention
        colored[:, :, 0] = pixels[:, :, 0]  # Red
        colored[:, :, 1] = (pixels[:, :, 0] * 0.2).astype(np.uint8) # Little Green
        colored[:, :, 2] = 0 # No Blue
        img = Image.fromarray(colored)
    
    # Convert to Base64
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    return f"data:image/png;base64,{img_str}"
