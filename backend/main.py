from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from backend.schemas.responses import RadiologyReport
from backend.services.myndra_runner import run_pneumonia, run_cardiomegaly, run_dual
import os
import tempfile
import shutil
import uuid
import time
import threading
from datetime import datetime
from typing import Dict, List, Any

app = FastAPI(title="Myndra Radiology API", version="1.0.0")

# CORS configuration for frontend (allow all origins in development)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Thread-safe storage
cases_db: Dict[str, Dict[str, Any]] = {}
db_lock = threading.Lock()

system_metrics = {
    "total_analyses": 0,
    "successful_analyses": 0,
    "failed_analyses": 0,
    "avg_latency_ms": 0.0,
    "uptime_seconds": 0,
}
metrics_lock = threading.Lock()

start_time = time.time()

def _save_temp(upload: UploadFile) -> str:
    """Save uploaded file to temporary location."""
    fd, path = tempfile.mkstemp(suffix=os.path.splitext(upload.filename or '')[-1] or ".jpg")
    with os.fdopen(fd, "wb") as f:
        shutil.copyfileobj(upload.file, f)
    return path

def _store_case(case_id: str, analysis_type: str, result: Dict[str, Any], latency_ms: float):
    """Store case result and update metrics thread-safely."""
    with db_lock:
        cases_db[case_id] = {
            "case_id": case_id,
            "analysis_type": analysis_type,
            "diagnosis": result.get("diagnosis", "Unknown"),
            "probability": result.get("probability", 0.0),
            "date": datetime.now().isoformat(),
            "agent": "MyndraAI",  # Simplified agent name
            "latency_ms": latency_ms
        }
    
    with metrics_lock:
        system_metrics["total_analyses"] += 1
        system_metrics["successful_analyses"] += 1
        # Moving average
        n = system_metrics["successful_analyses"]
        prev_avg = system_metrics["avg_latency_ms"]
        system_metrics["avg_latency_ms"] = prev_avg + (latency_ms - prev_avg) / n

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "uptime_seconds": int(time.time() - start_time),
    }

@app.get("/system/status")
async def system_status():
    """Get system status and metrics."""
    system_metrics["uptime_seconds"] = int(time.time() - start_time)
    return {
        "status": "operational",
        "metrics": system_metrics,
        "timestamp": datetime.utcnow().isoformat(),
    }

@app.get("/cases")
async def get_cases() -> List[Dict[str, Any]]:
    """Get all analysis cases."""
    return list(cases_db.values())

@app.get("/report/{case_id}")
async def get_report(case_id: str):
    """Get detailed report for a specific case."""
    if case_id not in cases_db:
        raise HTTPException(status_code=404, detail="Case not found")
    
    case = cases_db[case_id]
    return {
        **case,
        "orchestrator_trace": [
            {"step": "preprocess", "agent": "DataAgent", "status": "completed"},
            {"step": "inference", "agent": case["agent"], "status": "completed"},
            {"step": "postprocess", "agent": "SummarizerAgent", "status": "completed"},
        ],
        "system_info": {
            "model": "DenseNet121",
            "framework": "torchxrayvision",
            "device": os.getenv("MYNDRA_DEVICE", "cpu"),
        },
    }

@app.post("/analyze_pneumonia", response_model=RadiologyReport)
async def analyze_pneumonia(file: UploadFile = File(...)):
    """Analyze chest X-ray for pneumonia."""
    start = time.time()
    path = _save_temp(file)
    try:
        system_metrics["total_analyses"] += 1
        result = run_pneumonia(path)
        latency_ms = (time.time() - start) * 1000
        
        case_id = str(uuid.uuid4())
        result["case_id"] = case_id
        _store_case(case_id, "pneumonia", result, latency_ms)
        
        system_metrics["successful_analyses"] += 1
        # Update rolling average latency
        total = system_metrics["total_analyses"]
        system_metrics["avg_latency_ms"] = (
            system_metrics["avg_latency_ms"] * (total - 1) + latency_ms
        ) / total
        
        return result
    except Exception as e:
        system_metrics["failed_analyses"] += 1
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
    finally:
        if os.path.exists(path):
            os.remove(path)

@app.post("/analyze_cardiomegaly", response_model=RadiologyReport)
async def analyze_cardiomegaly(file: UploadFile = File(...)):
    """Analyze chest X-ray for cardiomegaly (heart enlargement)."""
    start = time.time()
    path = _save_temp(file)
    try:
        system_metrics["total_analyses"] += 1
        result = run_cardiomegaly(path)
        latency_ms = (time.time() - start) * 1000
        
        case_id = str(uuid.uuid4())
        result["case_id"] = case_id
        _store_case(case_id, "cardiomegaly", result, latency_ms)
        
        system_metrics["successful_analyses"] += 1
        total = system_metrics["total_analyses"]
        system_metrics["avg_latency_ms"] = (
            system_metrics["avg_latency_ms"] * (total - 1) + latency_ms
        ) / total
        
        return result
    except Exception as e:
        system_metrics["failed_analyses"] += 1
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
    finally:
        if os.path.exists(path):
            os.remove(path)

@app.post("/analyze_heart", response_model=RadiologyReport)
async def analyze_heart(file: UploadFile = File(...)):
    """Alias for cardiomegaly analysis (for frontend compatibility)."""
    return await analyze_cardiomegaly(file)

@app.post("/analyze_dual")
async def analyze_dual(file: UploadFile = File(...)):
    """Run both pneumonia and cardiomegaly analysis."""
    start = time.time()
    path = _save_temp(file)
    try:
        system_metrics["total_analyses"] += 1
        result = run_dual(path)
        latency_ms = (time.time() - start) * 1000
        
        case_id = str(uuid.uuid4())
        result["case_id"] = case_id
        # Store as dual analysis
        cases_db[case_id] = {
            "case_id": case_id,
            "patient_id": f"P{len(cases_db) + 1:05d}",
            "analysis_type": "dual",
            "date": datetime.utcnow().isoformat(),
            "result": result,
            "latency_ms": latency_ms,
        }
        
        system_metrics["successful_analyses"] += 1
        total = system_metrics["total_analyses"]
        system_metrics["avg_latency_ms"] = (
            system_metrics["avg_latency_ms"] * (total - 1) + latency_ms
        ) / total
        
        return result
    except Exception as e:
        system_metrics["failed_analyses"] += 1
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
    finally:
        if os.path.exists(path):
            os.remove(path)
