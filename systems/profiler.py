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
        "record sca;ar metric like gpu initalization or steps/sec."
        pass
    def get_summary(self):
        "return all collected metrics as a dictionary"
    def save(self, path:str):
        "write all collected metrics to a json file for later analysis"
        pass


if __name__ == "__main__":
    profiler = Profiler()
    with profiler.track("sleep_test"):
        time.sleep(0.2)
    print(profiler.timers)