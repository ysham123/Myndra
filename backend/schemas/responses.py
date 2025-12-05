from pydantic import BaseModel
from typing import Literal, Dict, Any, List, Optional

Diagnosis = Literal["Pneumonia","Normal","Malignant","Benign","Cardiomegaly","Unknown"]

class Step(BaseModel):
    name: str
    info: Dict[str, Any] = {}

class RadiologyReport(BaseModel):
    diagnosis: Diagnosis
    probability: float
    steps: List[Step]
    artifacts: Dict[str, Any] = {}
    case_id: Optional[str] = None  # Added by API layer
