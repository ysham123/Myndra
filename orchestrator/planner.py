import os
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

class Planner:
    """Phase 1: Rule-based planner. Decomposes high-level goals into ordered subtasks. Later, this 
    will be upgraded to an LLM-driven dynamic planner."""

    def __init__(self):
        pass

    def decompose(self, goal):
        """Rule-based task decomposition."""
        goal_lower = goal.lower()

        if "analyze" in goal_lower:
            return [
                "Gather all relevant data",
                "Analyze patterns or anomalies",
                "Summarize the findings"
            ]
        elif "design" in goal_lower:
            return [
                "Define design objectives",
                "Create initial concepts",
                "Review and refine designs"
            ]
        elif "research" in goal_lower:
            return [
                "Collect background information",
                "Form hypotheses",
                "Run experiments",
                "Interpret results"
            ]
        else:
            return [
                "Understand the goal context",
                "Propose an action plan",
                "Execute and report results"
            ]


class LLMPlanner:
    """Phase 3: Memory-aware LLM planner using GPT-5. Incorporates context from SharedMemory."""

    def __init__(self, memory=None, model="gpt-5-mini"):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = model
        self.memory = memory  # shared memory reference

    def decompose(self, goal):
        """Use GPT-5 to break a goal into a list of ordered subtasks with memory context."""
        # Retrieve past context from memory (if available)
        context = ""
        if self.memory:
            try:
                recent = self.memory.get_recent("agent:orchestrator")
                context = "\n".join([f"- {m['content']}" for m in recent[-5:]]) if recent else ""
            except Exception:
                context = ""

        prompt = (
            "You are a smart planning assistant. "
            "Use the following past context (if any) to plan more effectively.\n\n"
            f"Context:\n{context}\n\n"
            f"Goal: {goal}\n\nSubtasks:"
        )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
        )

        text = response.choices[0].message.content
        subtasks = [
            line.strip("123456.-• ").capitalize()
            for line in text.split("\n")
            if line.strip()
        ]
        return subtasks


class PlannerAdapter:
    """Adapter that switches between rule-based and LLM planners."""

    def __init__(self, use_llm=False, memory=None):
        self.rule_based = Planner()
        self.llm_based = LLMPlanner(memory=memory)
        self.use_llm = use_llm

    def decompose(self, goal):
        if self.use_llm:
            return self.llm_based.decompose(goal)
        else:
            return self.rule_based.decompose(goal)