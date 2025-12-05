import argparse
import json
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from domains.radiology_pneumonia.pipeline import predict as pneu
from domains.radiology_cardiomegaly.pipeline import predict as cardio

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--task", choices=["pneumonia","cardiomegaly","dual"], default="dual")
    args = ap.parse_args()

    if args.task == "pneumonia":
        out = pneu(args.image)
    elif args.task == "cardiomegaly":
        out = cardio(args.image)
    else:
        from backend.services.myndra_runner import run_dual
        out = run_dual(args.image)

    print(json.dumps(out, indent=2))
