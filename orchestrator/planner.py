class Planner:
    """
    Phase 1: Rule-based planner.
    Decomposes high-level goals into ordered subtasks.
    Later, this will be upgraded to an LLM-driven dynamic planner.
    """

    def __init__(self):
        pass

    def decompose(self, goal):
        """Break a high-level goal into smaller, actionable subtasks."""
        goal_lower = goal.lower()

        # Simple keyword-based patterns for now
        if "analyze" in goal_lower:
            subtasks = [
                "Gather all relevant data",
                "Analyze patterns or anomalies",
                "Summarize the findings"
            ]
        elif "design" in goal_lower:
            subtasks = [
                "Define design objectives",
                "Create initial concepts",
                "Review and refine designs"
            ]
        elif "research" in goal_lower:
            subtasks = [
                "Collect background information",
                "Form hypotheses",
                "Run experiments",
                "Interpret results"
            ]
        else:
            subtasks = [
                "Understand the goal context",
                "Propose an action plan",
                "Execute and report results"
            ]

        return subtasks

class LLMPlanner:
    ''