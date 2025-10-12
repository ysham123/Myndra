"""
[ User Goal ]
     ↓
[ Planner ] → decomposes goal into subtasks
     ↓
[ Orchestrator ] → assigns tasks, gathers results
     ↓
[ Agents ] ↔ [ SharedMemory ]
"""
class Orchestrator:
    def __init__(self, registry, memory):
        self.registry = registry
        self.memory = memory

    def plan(self, goal):
        """Breaks high-level goal into smaller subtasks."""
        self.memory.write("orchestrator", f"Received goal: {goal}")
        goal_lower = goal.lower() 

        if "analyze" in goal_lower:
            subtasks = [
                "Gather all relevant data",
                "Analyze patterns or anomalies",
                "Summarize the findings"
            ]
        elif "summarize" in goal_lower:
            subtasks = [
                "Identify main points",
                "Write a concise summary"
            ]
        else:
            subtasks = [
                "Interpret the goal",
                "Perform main action",
                "Generate final report"
            ]
        self.memory.write("orchestrator", f"Planned subtasks: {subtasks}")
        return subtasks


    def assign(self, subtasks):
        """assign subtasks to appropriate agents."""
        assignments = []

        for task in subtasks:
            task_lower = task.lower()
            if "data" in task_lower or "gather" in task_lower:
                agent = "DataAgent"
            elif "analyze" in task_lower or "pattern" in task_lower:
                agent = "AnalystAgent"
            elif "summarize" in task_lower or "report" in task_lower:
                agent = "SummarizerAgent"
            else:
                agent = "GeneralAgent"
            assignments.append({"task" : task, "agent": agent})
        self.memory.write("orchestrator", f"Assigned tasks: {assignments}")
        return assignments



    def execute(self, assignments):
        """Execute each assignment and gather results."""
        pass
    def adapt(self, results):
        """Optional: adjust agent teams or task flow based on memory feedback."""
        pass
    def run(self, goal):
        """Main entry point for orchestration."""
        pass