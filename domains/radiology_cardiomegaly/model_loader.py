"""Cardiomegaly detection model loader."""

from domains.radiology_common.model_loader import load_radiology_model
from typing import Tuple, Optional
import torch

def load_model(device: Optional[str] = None) -> Tuple[torch.nn.Module, Optional[int], str]:
    """Load pretrained model for cardiomegaly detection.
    
    Args:
        device: Device to load model on (defaults to env MYNDRA_DEVICE or "cpu")
    
    Returns:
        Tuple of (model, task_index, device)
    """
    return load_radiology_model(task="Cardiomegaly", device=device)
