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
    """Memory-aware LLM planner that returns structured subtasks. Default model: GPT-5-mini
    (override via MYNDRA_PLANNER_MODEL). Forces JSON object output and normalizes responses
    from diverse model formats to a consistent list of {task, agent, confidence}."""

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

    def parse_llm_json(self, text):
        """Helper to parse JSON from LLM output, trying to fix common formatting issues."""
        import re

        # Remove code fences if present
        text = re.sub(r"^```json\s*|```$", "", text.strip(), flags=re.MULTILINE)

        # Replace smart quotes with normal quotes
        text = text.replace("“", "\"").replace("”", "\"").replace("‘", "'").replace("’", "'")

        # Remove trailing commas before closing brackets/braces
        text = re.sub(r",(\s*[\]}])", r"\1", text)

        # Extract first JSON array if extra text leaked
        start = text.find("[")
        end = text.rfind("]")
        if start != -1 and end != -1 and end > start:
            text = text[start:end+1]

        return json.loads(text)

    def _extract_subtasks(self, parsed, raw_text=""):
        """Normalize various JSON shapes to a list of subtasks.
        Accepts either a list of objects or an object containing one of several list keys.
        Each subtask is normalized to have keys: task(str), agent(str), confidence(float in [0,1]).
        """
        # If the model already returned a list
        if isinstance(parsed, list):
            subtasks = parsed
        elif isinstance(parsed, dict):
            # Common container keys the model might use
            for key in ("subtasks", "steps", "tasks", "plan", "items", "list"):
                if key in parsed and isinstance(parsed[key], list):
                    subtasks = parsed[key]
                    break
            else:
                # Nothing looks like a subtask list
                raise ValueError("No subtask list found in JSON object")
        else:
            raise ValueError("Planner JSON is neither list nor object")

        normalized = []
        for idx, item in enumerate(subtasks):
            if isinstance(item, str):
                # Very lenient fallback: try to parse a line like
                # "Task: X | agent: planner | confidence: 0.9"
                t = item
                agent = "general"
                conf = 0.8
                # crude parsing
                import re
                m_task = re.search(r"task\s*[:=]\s*(.+?)(?:\||$)", t, flags=re.I)
                m_agent = re.search(r"agent\s*[:=]\s*([a-zA-Z]+)", t, flags=re.I)
                m_conf = re.search(r"conf(?:idence)?\s*[:=]\s*([0-9.]+)", t, flags=re.I)
                task_val = (m_task.group(1).strip() if m_task else t).strip().strip("|,")
                if m_agent:
                    agent = m_agent.group(1).strip().lower()
                if m_conf:
                    try:
                        conf = float(m_conf.group(1))
                    except Exception:
                        conf = 0.8
                item = {"task": task_val, "agent": agent, "confidence": conf}
            elif isinstance(item, dict):
                item = dict(item)
            else:
                raise ValueError(f"Unsupported subtask element type at index {idx}: {type(item)}")

            # Normalize keys
            task_val = item.get("task") or item.get("title") or item.get("name")
            agent_val = (item.get("agent") or item.get("role") or "general").lower()
            conf_val = item.get("confidence", 0.8)
            try:
                conf_val = float(conf_val)
            except Exception:
                conf_val = 0.8
            # Clamp confidence
            conf_val = max(0.0, min(1.0, conf_val))
            # Restrict agent vocab
            if agent_val not in ("analyst", "planner", "executor", "general"):
                agent_val = "general"
            if not task_val or not isinstance(task_val, str):
                raise ValueError(f"Missing/invalid 'task' at index {idx}")
            normalized.append({"task": task_val.strip(), "agent": agent_val, "confidence": conf_val})
        return normalized

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

        text = ""
        prompt = (
            "You are an expert project planner. "
            "Given a high-level goal and recent context, decompose the goal into ordered subtasks. "
            "Return a single JSON object with this exact schema: \n"
            "{\n  \"subtasks\": [\n    {\n      \"task\": string,\n      \"agent\": one of ['analyst','planner','executor','general'],\n      \"confidence\": number between 0 and 1\n    }\n  ]\n}\n"
            "Do not include any extra fields or prose. If a design/visualization task is needed, use 'planner'.\n"
            f"Context:\n{context}\n\n"
            f"Goal: {goal}\n\n"
            "Respond with only the JSON object."
        )

        if self.client is not None:
            try:
                # Note: Some GPT-5 endpoints only support the default temperature and reject custom values.
                # We omit the temperature parameter for maximum compatibility.
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"},
                )
                text = response.choices[0].message.content.strip()
                try:
                    parsed = json.loads(text)
                except Exception:
                    # fall back to the permissive fixer
                    parsed = self.parse_llm_json(text)
                # Normalize whatever we got into a proper list
                subtasks = self._extract_subtasks(parsed, raw_text=text)

                print(f"\nLLM Planner: model={self.model} successfully generated plan.\n")
                return subtasks
            except Exception as e:
                # Log raw output to file for debugging
                try:
                    os.makedirs("logs", exist_ok=True)
                    with open("logs/llm_planner_raw_output.log", "a", encoding="utf-8") as f:
                        f.write(f"---\nGoal: {goal}\nError: {e}\nRaw output:\n{text}\n\n")
                except Exception:
                    pass
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
