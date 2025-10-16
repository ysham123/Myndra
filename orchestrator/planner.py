import os
import json
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
    """Memory-aware LLM planner using GPT-5-mini. Incorporates context from SharedMemory."""

    def __init__(self, memory=None, model="gpt-5-mini"):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = model  # Using GPT-5-mini as the model
        self.memory = memory  # shared memory reference

    def decompose(self, goal):
        """Use GPT-5-mini to break a goal into a list of ordered subtasks with memory context, returning JSON with task, agent, confidence."""
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
                model=self.model,
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
    def __init__(self, use_llm=False, memory=None):
        self.use_llm = use_llm
        self.memory = memory
        self.llm_planner = LLMPlanner(memory)

    def decompose(self, goal: str):
        """Decompose a goal into subtasks (hierarchical if use_llm=True)."""
        if self.use_llm:
            return self._decompose_with_llm(goal)
        else:
            # simple fallback
            return [
                {"task": "Define objectives and KPIs", "agent": "AnalystAgent", "depends_on": [], "confidence": 0.9},
                {"task": "Gather and preprocess data", "agent": "DataAgent", "depends_on": ["Define objectives and KPIs"], "confidence": 0.8},
                {"task": "Run analysis and extract insights", "agent": "AnalystAgent", "depends_on": ["Gather and preprocess data"], "confidence": 0.7},
                {"task": "Generate visualizations and summary report", "agent": "SummarizerAgent", "depends_on": ["Run analysis and extract insights"], "confidence": 0.9},
            ]

    def _decompose_with_llm(self, goal: str):
        """Use an LLM to create a dependency-aware task hierarchy."""
        return self.llm_planner.decompose(goal)