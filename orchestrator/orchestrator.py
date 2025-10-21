from orchestrator.planner import PlannerAdapter
from agents.agent_registry import get_agent
from systems.profiler import Profiler
import os
import json
from systems.async_runtime import AsyncRuntime
import asyncio

class Orchestrator:
    def __init__(self, registry, memory, use_llm=False):
        self.profiler = Profiler()

        self.registry = registry
        self.memory = memory
        self.planner = PlannerAdapter(
            use_llm=os.getenv("MYNDRA_USE_LLM", "0") == "1",
            memory=self.memory
        )
        self.runtime = AsyncRuntime(max_concurrent=4)

    def plan(self, goal):
        subtasks = self.planner.decompose(goal)
        self.memory.write("orchestrator", f"Planned subtasks: {subtasks}")
        return subtasks

        
    def assign(self, subtasks):
        """assign subtasks to appropriate agents. Handles both string and dict subtasks."""
        assignments = []

        for subtask in subtasks:
            # Handle if subtask is a dict (with possible agent/confidence), or str
            if isinstance(subtask, dict):
                task_text = subtask.get("task", "")
                agent_hint = subtask.get("agent", "")
                confidence = subtask.get("confidence", 0.5)
            else:
                task_text = str(subtask)
                agent_hint = ""
                confidence = 0.5

            task_lower = task_text.lower()
            # Agent assignment logic: prefer explicit agent_hint, else auto-detect
            if agent_hint:
                agent = agent_hint
            elif "data" in task_lower or "gather" in task_lower:
                agent = "DataAgent"
            elif "analyze" in task_lower or "pattern" in task_lower:
                agent = "AnalystAgent"
            elif "summarize" in task_lower or "report" in task_lower:
                agent = "SummarizerAgent"
            else:
                agent = "GeneralAgent"
            assignments.append({"task": task_text, "agent": agent, "confidence": confidence})
        self.memory.write("orchestrator", f"Assigned tasks: {assignments}")
        return assignments


    def execute(self, assignments):
        """Execute all agent assignments concurrently using AsyncRuntime."""
        results = []

        try:
            # Run assignments concurrently via AsyncRuntime
            results = asyncio.run(self.runtime.run_batch(assignments, lambda a: get_agent(a, self.memory)))
        except Exception as e:
            print(f"[Execute] Runtime error: {e}")
            results = [{"agent": "system", "task": "runtime_error", "output": str(e)}]

        self.memory.write("agent:orchestrator", f"Execution results: {results}")
        return results


    def adapt(self, results):
        """Adjust agent teams or task flow based on memory feedback."""
        adjustments = []

        for result in results:
            # Handle both old and new formats
            output = result.get("output") or result.get("result") or ""
            output_lower = str(output).lower()

            agent = result.get("agent", "unknown")
            task = result.get("task", "unknown")

            if "error" in output_lower or "failed" in output_lower:
                action = f"Reassigning task '{task}' due to error in {agent}"
                self.memory.write("orchestrator", action)
                adjustments.append({"task": task, "action": "reassign"})
            else:
                action = f"Task '{task}' by {agent} completed successfully"
                self.memory.write("orchestrator", action)
                adjustments.append({"task": task, "action": "retain"})

        summary = {"adaptations": adjustments}
        self.memory.write("orchestrator", f"Adaptation summary: {summary}")
        return summary


    def run(self, goal):
        """Run the full orchestration pipeline."""
        print(f"\nGoal: {goal}")

        # Wrap the entire run for a total time
        with self.profiler.track("total_run_latency"):
            # 1. Plan
            with self.profiler.track("plan_latency"):
                subtasks = self.planner.decompose(goal)
            print("\nPlanned Subtasks:")
            for t in subtasks:
                print(f"  - {t}")

            # 2. Assign
            with self.profiler.track("assign_latency"):
                assignments = self.assign(subtasks)
            print("\nAssignments:")
            for a in assignments:
                print(f"  - {a['task']} → {a['agent']}")

            # 3. Execute
            with self.profiler.track("execute_latency"):
                results = self.execute(assignments)
            print("\nExecution Results:")
            for r in results:
                agent = r.get("agent", "unknown")
                output = r.get("output", r)
                print(f"  - {agent} → {output}")

            # 4. Adapt
            with self.profiler.track("adapt_latency"):
                adaptation = self.adapt(results)
            print("\nAdaptation Summary:")
            for a in adaptation["adaptations"]:
                print(f"  - {a['task']} → {a['action']}")

            # New block for final summary
            print("\nFinal Summary (LLM-driven):")
            with self.profiler.track("summarize_latency"):
                summarizer = get_agent("SummarizerAgent", self.memory)
                summary = summarizer.act(results)
            print(summary)

        # 5. Memory Log (optional)
        print("\nRecent Memory (Orchestrator):")
        for m in self.memory.get_recent("agent:orchestrator"):
            print(f"  • {m['timestamp']} | {m['content']}")
            
        # Save profiling results to file
        self.profiler.save("results/orchestrator_profile.json")

    def print_summary(self):
        summary = self.profiler.get_summary()
        print(json.dumps(summary, indent=2))