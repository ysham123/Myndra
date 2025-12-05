"""Cardiomegaly (heart enlargement) detection pipeline."""

from typing import Dict, Any
import torch
import os
from domains.radiology_common.preprocessing import load_cxr
from domains.radiology_common.heatmap import simple_saliency
from domains.radiology_common.types import RadiologyReport
from .model_loader import load_model

# Classification threshold
CARDIOMEGALY_THRESHOLD = 0.5

def predict(image_path: str, generate_heatmap: bool = True) -> RadiologyReport:
    """Analyze chest X-ray for cardiomegaly (heart enlargement).
    
    Args:
        image_path: Path to chest X-ray image
        generate_heatmap: Whether to generate saliency heatmap
    
    Returns:
        RadiologyReport with diagnosis, probability, steps, and artifacts
    
    Raises:
        FileNotFoundError: If image doesn't exist
        ValueError: If image processing fails
        RuntimeError: If model inference fails
    """
    # Load model
    model, idx, device = load_model()
    
    # Track processing steps
    steps = []
    
    # Preprocess image
    x = load_cxr(image_path)
    x = x.to(device)
    x.requires_grad_(True)  # Enable gradients for saliency
    
    steps.append({
        "name": "preprocess",
        "info": {
            "size": "224x224",
            "normalize": "z-score",
            "channels": "grayscale",
        }
    })
    
    # Run model inference
    with torch.set_grad_enabled(True):  # Keep gradients for saliency
        logits = model(x)
    
    # Apply sigmoid to get probabilities
    probs = torch.sigmoid(logits)
    cardio_prob = float(probs[0, idx]) if idx is not None else 0.0
    
    # Classify based on threshold
    diagnosis = "Cardiomegaly" if cardio_prob >= CARDIOMEGALY_THRESHOLD else "Normal"
    
    steps.append({
        "name": "inference",
        "info": {
            "model": "DenseNet121",
            "source": "torchxrayvision",
            "task": "Cardiomegaly",
            "threshold": CARDIOMEGALY_THRESHOLD,
        }
    })
    
    # Generate saliency heatmap
    artifacts = {}
    if generate_heatmap:
        try:
            score = probs[0, idx]
            out_png = f"results/radiology/heatmaps/cardiomegaly_{os.path.basename(image_path)}.png"
            heatmap_b64 = simple_saliency(x, score, out_png, apply_colormap=True)
            artifacts["heatmap_png"] = heatmap_b64
            steps.append({
                "name": "saliency",
                "info": {"method": "input_gradient", "format": "base64"}
            })
        except Exception as e:
            artifacts["heatmap_error"] = str(e)
    
    return {
        "diagnosis": diagnosis,
        "probability": cardio_prob,
        "steps": steps,
        "artifacts": artifacts,
    }
