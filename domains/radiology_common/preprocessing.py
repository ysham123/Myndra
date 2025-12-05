"""Preprocessing utilities for chest X-ray images."""

from PIL import Image
import numpy as np
import torch
from pathlib import Path
from typing import Optional

def load_cxr(
    image_path: str,
    size: int = 224,
    mean: float = 0.5,
    std: float = 0.25,
) -> torch.Tensor:
    """Load and preprocess a chest X-ray image for model inference.
    
    This function:
    1. Loads the image and converts to grayscale
    2. Resizes to specified size (default 224x224)
    3. Normalizes pixel values to [0, 1]
    4. Applies z-score normalization
    5. Adds batch and channel dimensions
    
    Args:
        image_path: Path to the chest X-ray image file
        size: Target image size (height and width)
        mean: Mean for normalization (default 0.5)
        std: Standard deviation for normalization (default 0.25)
    
    Returns:
        Preprocessed image tensor with shape (1, 1, size, size)
    
    Raises:
        FileNotFoundError: If image file doesn't exist
        ValueError: If image cannot be loaded or processed
    """
    # Validate input path
    img_path = Path(image_path)
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    try:
        # Load image and convert to grayscale (1 channel for torchxrayvision)
        img = Image.open(image_path).convert("L")
        
        # Resize to target size
        img = img.resize((size, size), Image.Resampling.BILINEAR)
        
        # Convert to numpy array and normalize to [0, 1]
        arr = np.asarray(img).astype(np.float32) / 255.0
        
        # Add channel dimension: (H, W) -> (1, H, W)
        arr = np.expand_dims(arr, 0)
        
        # Convert to PyTorch tensor
        tensor = torch.from_numpy(arr)
        
        # Apply z-score normalization
        tensor = (tensor - mean) / std
        
        # Add batch dimension: (1, H, W) -> (1, 1, H, W)
        tensor = tensor.unsqueeze(0)
        
        return tensor
        
    except Exception as e:
        raise ValueError(f"Failed to process image {image_path}: {e}")
