from orchestrator.planner import PlannerAdapter
from agents.agent_registry import get_agent

class Orchestrator:
    def __init__(self, registry, memory, use_llm=False):
        import os
        self.registry = registry
        self.memory = memory
        self.planner = PlannerAdapter(
            use_llm=os.getenv("MYNDRA_USE_LLM", "0") == "1",
            memory=self.memory
        )

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
        """Execute each assignment and gather results."""
        results = []

        for assignment in assignments:
            agent_name = assignment["agent"]
            task = assignment["task"]

            agent_instance = get_agent(agent_name, self.memory)
            output = agent_instance.act(task)
            results.append({"agent":agent_name,"task":task,"output":output})
            
            #only mold if agent supports it
            if hasattr(agent_instance, "mold"):
                agent_instance.mold("successfully completed task")
                
        self.memory.write("agent:orchestrator",f"Execution results: {results}")

        return results

    def adapt(self, results):
        """Optional: adjust agent teams or task flow based on memory feedback."""
        adjustments = []

        for result in results:
            output = result["output"].lower()
            agent = result["agent"]
            task = result["task"]

            if "error" in output or "failed" in output:
                action = f"Reassignming task '{task}' due to error in {agent}"
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

    # 1. Plan
        subtasks = self.planner.decompose(goal)
        print("\nPlanned Subtasks:")
        for t in subtasks:
            print(f"  - {t}")

    # 2. Assign
        assignments = self.assign(subtasks)
        print("\nAssignments:")
        for a in assignments:
            print(f"  - {a['task']} → {a['agent']}")

    # 3. Execute
        results = self.execute(assignments)
        print("\nExecution Results:")
        for r in results:
            print(f"  - {r['agent']} → {r['output']}")

    # 4. Adapt
        adaptation = self.adapt(results)
        print("\nAdaptation Summary:")
        for a in adaptation["adaptations"]:
            print(f"  - {a['task']} → {a['action']}")

    # New block for final summary
        print("\nFinal Summary (LLM-driven):")
        summarizer = get_agent("SummarizerAgent", self.memory)
        summary = summarizer.act(results)
        print(summary)

    # 5. Memory Log (optional)
        print("\nRecent Memory (Orchestrator):")
        for m in self.memory.get_recent("agent:orchestrator"):
            print(f"  • {m['timestamp']} | {m['content']}")