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
        """Use GPT-4o-mini to break a goal into a list of ordered subtasks with memory context, returning JSON with task, agent, confidence."""
        import json
        context = ""
        if self.memory:
            try:
                recent = self.memory.get_recent("agent:orchestrator")
                context = "\n".join([f"- {m['content']}" for m in recent[-5:]]) if recent else ""
            except Exception:
                context = ""

        prompt = (
            "You are an expert project planner. "
            "Given a high-level goal and recent context, decompose the goal into a list of ordered subtasks. "
            "Return ONLY a JSON list of objects. Each object must have fields: 'task' (the subtask as a string), "
            "'agent' (the most suitable agent type, e.g. 'analyst', 'designer', etc.), and "
            "'confidence' (a float between 0 and 1 for your confidence in this step). "
            "If context is empty, proceed as best as possible.\n\n"
            f"Context:\n{context}\n\n"
            f"Goal: {goal}\n\n"
            "Respond with only the JSON list, no explanations."
        )
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.4,
            )
            text = response.choices[0].message.content
            subtasks = json.loads(text)
            # Validate structure
            if not isinstance(subtasks, list):
                raise ValueError("Subtasks not a list")
            for sub in subtasks:
                if not all(k in sub for k in ("task", "agent", "confidence")):
                    raise ValueError("Missing keys in subtask")
            return subtasks
        except Exception:
            # Fallback: return a generic decomposition
            return [
                {"task": "Understand the goal context", "agent": "analyst", "confidence": 0.8},
                {"task": "Propose an action plan", "agent": "planner", "confidence": 0.7},
                {"task": "Execute and report results", "agent": "executor", "confidence": 0.7},
            ]


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