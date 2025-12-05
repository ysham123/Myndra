import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from domains.radiology_pneumonia.pipeline import predict as pneu
from domains.radiology_cardiomegaly.pipeline import predict as cardio

def test_pipelines_smoke():
    img = "tests/assets/sample_cxr.jpg"
    r1 = pneu(img)
    r2 = cardio(img)
    for r in (r1, r2):
        assert "diagnosis" in r
        assert "probability" in r
        assert "steps" in r
        assert isinstance(r["probability"], float)
