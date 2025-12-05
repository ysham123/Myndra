from typing import Literal, TypedDict, Dict, Any, List

Diagnosis = Literal["Pneumonia","Normal","Malignant","Benign","Cardiomegaly","Unknown"]

class Step(TypedDict):
    name: str
    info: Dict[str, Any]

class Artifacts(TypedDict, total=False):
    heatmap_png: str
    log: str

class RadiologyReport(TypedDict):
    diagnosis: Diagnosis
    probability: float
    steps: List[Step]
    artifacts: Artifacts
