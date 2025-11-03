"""
Profiler module for Myndra v2
--------------------------------
Lightweight timing and metrics logger used across orchestrator,
agents, and domains (MARL, Radiology).
Tracks latency, throughput, GPU utilization, and custom metrics.

Usage:
    from systems.profiler import Profiler
    profiler = Profiler()

    with profiler.track("planner_latency_ms"):
        result = planner.decompose(goal)

    profiler.log_metric("steps_per_second", 512.3)
    profiler.save("results/logs/run_summary.json")
"""

import time
import json
from contextlib import contextmanager
import os
import subprocess

class Profiler:
    def __init__(self):
        """Initialize metric containers and time tracking state"""
        self.timers = {}
        self.metrics = {}

    def start(self, name:str):
        self.timers[name] = {"start":time.time()}

    def stop(self, name:str):
        if name not in self.timers or "start" not in self.timers[name]:
            raise ValueError(f"Timer '{name}' was not started")
        
        end = time.time()
        start = self.timers[name]["start"]
        duration_ms = (end-start) * 1000
        self.timers[name].update({"end":end, "duration_ms": duration_ms})

    @contextmanager
    def track(self, name:str):
        """Context manager wrapper for timing a code block"""
        self.start(name)
        try:
            yield
        finally:
            self.stop(name)



    def log_metric(self, key:str, value:float):
        "record scalar metric like gpu initalization or steps/sec."
        if key not in self.metrics:
            self.metrics[key] = []
        self.metrics[key].append(value)
    
    def sample_gpu_util(self):
        """Sample GPU utilization percentage. Returns None if no GPU or error."""
        try:
            # Try nvidia-smi first
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=1.0
            )
            if result.returncode == 0:
                # Get first GPU utilization
                util = float(result.stdout.strip().split('\n')[0])
                return util
        except (FileNotFoundError, subprocess.TimeoutExpired, ValueError, IndexError):
            pass
        
        # Fallback to torch CUDA utilization if available
        try:
            import torch
            if torch.cuda.is_available():
                util = torch.cuda.utilization(0)  # Device 0
                return float(util) if util is not None else None
        except (ImportError, RuntimeError):
            pass
        
        return None
    def get_summary(self):
        "return all collected metrics as a dictionary"
        summary = {}
        for key, values in self.metrics.items():
            if not values:
                continue
            summary[key] = {
                "last":values[-1],
                "mean":sum(values) / len(values)
            }
        return summary

    def save(self, path:str):
        "write all collected metrics to a json file for later analysis"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = {"timers":self.timers, "metrics":self.metrics}
        with open(path, "w") as f:
            json.dump(data, f, indent=4)


if __name__ == "__main__":
    profiler = Profiler()
    with profiler.track("sleep_test"):
        time.sleep(0.2)
    profiler.log_metric("steps_per_second", 1024.5)
    profiler.save("results.json")
    print(profiler.get_summary())