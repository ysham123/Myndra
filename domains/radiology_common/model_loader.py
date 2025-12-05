"""Unified model loader for radiology tasks."""

import os
import torch
import torchxrayvision as xrv
from typing import Tuple, Optional

# Global model cache to avoid reloading
_MODEL_CACHE = {}

def load_radiology_model(
    task: str,
    device: Optional[str] = None,
    weights: str = "densenet121-res224-all",
) -> Tuple[torch.nn.Module, Optional[int], str]:
    """Load a pretrained radiology model for a specific task.
    
    Args:
        task: Target pathology (e.g., "Pneumonia", "Cardiomegaly")
        device: Device to load model on ("cpu", "cuda", or None for auto)
        weights: Model weights identifier
    
    Returns:
        Tuple of (model, task_index, device)
    """
    device = device or os.getenv("MYNDRA_DEVICE", "cpu")
    
    # Check cache
    cache_key = (weights, device)
    if cache_key in _MODEL_CACHE:
        model = _MODEL_CACHE[cache_key]
    else:
        # Load model with specified weights
        model = xrv.models.DenseNet(weights=weights)
        model.eval().to(device)
        _MODEL_CACHE[cache_key] = model
    
    # Find task index in model's pathology list
    task_to_idx = {t: i for i, t in enumerate(model.pathologies)}
    
    # Try exact match first
    idx = task_to_idx.get(task, None)
    
    # Fallback: fuzzy match (case-insensitive substring)
    if idx is None:
        task_lower = task.lower()
        for pathology, i in task_to_idx.items():
            if task_lower in pathology.lower():
                idx = i
                break
    
    if idx is None:
        available = ", ".join(model.pathologies)
        raise ValueError(
            f"Task '{task}' not found in model pathologies. "
            f"Available: {available}"
        )
    
    return model, idx, device
