from typing import Dict, Any
from domains.radiology_pneumonia.pipeline import predict as predict_pneumonia
from domains.radiology_cardiomegaly.pipeline import predict as predict_cardiomegaly

def run_pneumonia(image_path: str) -> Dict[str, Any]:
    return predict_pneumonia(image_path)

def run_cardiomegaly(image_path: str) -> Dict[str, Any]:
    return predict_cardiomegaly(image_path)

def run_dual(image_path: str) -> Dict[str, Any]:
    """Fan-out to both tasks and return a merged view."""
    lung = predict_pneumonia(image_path)
    heart = predict_cardiomegaly(image_path)
    return {
        "pneumonia": lung,
        "cardiomegaly": heart,
        "orchestrated": {
            "summary": f"{lung['diagnosis']} (p={lung['probability']:.2f}), "
                       f"{heart['diagnosis']} (p={heart['probability']:.2f})"
        }
    }
