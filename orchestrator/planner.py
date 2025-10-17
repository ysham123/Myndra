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
    """Memory-aware LLM planner (default: GPT-5-mini; override via MYNDRA_PLANNER_MODEL)."""

    def __init__(self, memory=None, model=None):
        # Resolve model (env override allowed) and initialize client from env OPENAI_API_KEY
        self.model = model or os.getenv("MYNDRA_PLANNER_MODEL") or "gpt-5-mini"
        print(f"🔧 LLMPlanner: using model '{self.model}'.")
        self.memory = memory
        try:
            # OpenAI() reads OPENAI_API_KEY from the environment
            self.client = OpenAI()
        except Exception:
            self.client = None

    def decompose(self, goal):
        """
        Use the LLM to return a JSON list of {task, agent, confidence}.
        Falls back to a generic plan if the LLM is unavailable or returns invalid output.
        """
        # Gather recent orchestrator context (best-effort)
        context = ""
        if self.memory is not None:
            try:
                recent = self.memory.get_recent("agent:orchestrator")
                if recent:
                    def _fmt(m):
                        if isinstance(m, dict) and "content" in m:
                            return f"- {m['content']}"
                        return f"- {str(m)}"
                    context = "\n".join([_fmt(m) for m in recent[-5:]])
            except Exception:
                context = ""

        prompt = (
            "You are an expert project planner. "
            "Given a high-level goal and recent context, decompose the goal into a list of ordered subtasks. "
            "Return ONLY a JSON list of objects. Each object must have fields: 'task' (string), "
            "'agent' (one of: 'analyst', 'planner', 'executor', 'general'), and "
            "'confidence' (a float between 0 and 1). "
            "Do not invent new agent roles. If a design/visualization task is needed, use 'planner'.\n"
            f"Context:\n{context}\n\n"
            f"Goal: {goal}\n\n"
            "Respond with only the JSON list, no explanations."
        )

        if self.client is not None:
            try:
                # Note: Some GPT-5 endpoints only support the default temperature and reject custom values.
                # We omit the temperature parameter for maximum compatibility.
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                )
                text = response.choices[0].message.content.strip()

                # Extract first JSON array if extra text leaked
                start = text.find("[")
                end = text.rfind("]")
                if start != -1 and end != -1:
                    text = text[start:end+1]

                subtasks = json.loads(text)
                if not isinstance(subtasks, list):
                    raise ValueError("Subtasks not a list")

                for sub in subtasks:
                    if not all(k in sub for k in ("task", "agent", "confidence")):
                        raise ValueError("Missing keys in subtask")
                    sub["agent"] = str(sub["agent"]).lower()

                print(f"\nLLM Planner: model={self.model} successfully generated plan.\n")
                return subtasks
            except Exception as e:
                # Make the reason visible in the console so you know why it fell back
                print(f"LLM plan failed: {e}")

        # Fallback: minimal JSON schema the orchestrator expects
        return [
            {"task": "Understand the goal context", "agent": "analyst", "confidence": 0.8},
            {"task": "Propose an action plan", "agent": "planner", "confidence": 0.7},
            {"task": "Execute and report results", "agent": "executor", "confidence": 0.7},
        ]




class PlannerAdapter:
    def __init__(self, use_llm=False, memory=None):
        # Allow env toggle: MYNDRA_USE_LLM=1|true|yes|on
        env_flag = str(os.getenv("MYNDRA_USE_LLM", "")).lower() in ("1", "true", "yes", "on")
        self.use_llm = use_llm or env_flag
        self.memory = memory
        self.llm_planner = LLMPlanner(memory=memory)

    def decompose(self, goal: str):
        """Decompose a goal into subtasks (hierarchical if use_llm=True)."""
        if self.use_llm:
            return self._decompose_with_llm(goal)
        else:
            # simple fallback
            return [
                {"task": "Define objectives and KPIs", "agent": "analyst", "depends_on": [], "confidence": 0.9},
                {"task": "Gather and preprocess data", "agent": "executor", "depends_on": ["Define objectives and KPIs"], "confidence": 0.8},
                {"task": "Run analysis and extract insights", "agent": "analyst", "depends_on": ["Gather and preprocess data"], "confidence": 0.7},
                {"task": "Generate visualizations and summary report", "agent": "planner", "depends_on": ["Run analysis and extract insights"], "confidence": 0.9},
            ]

    def _decompose_with_llm(self, goal: str):
        """Use an LLM to create a dependency-aware task hierarchy."""
        return self.llm_planner.decompose(goal)