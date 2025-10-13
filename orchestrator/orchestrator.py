from orchestrator.planner import Planner

class Orchestrator:
    def __init__(self, registry, memory):
        self.registry = registry
        self.memory = memory
        self.planner = Planner()

    def plan(self, goal):
        subtasks = self.planner.decompose(goal)
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
        results = []

        for item in assignments:
            agent = item["agent"]
            task = item["task"]

            #log start of task
            self.memory.write("orchestrator", f"Executing task '{task}' assigned to {agent}")

            #simulated result(placeholder until real agent is created)
            simulated_output = f"[{agent} completed task: {task}]"

            #log memory to result
            self.memory.write(agent, simulated_output)

            #store structured result
            results.append({
                "agent": agent,
                "task": task,
                "output": simulated_output
            })

        self.memory.write("orchestrator", f"Execution results: {results}")
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
        """Main entry point for orchestration."""
        pass