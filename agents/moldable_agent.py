from agents.base_agent import BaseAgent
from datetime import datetime, timezone

class MoldableAgent(BaseAgent):
    def __init__(self, name, role, memory):
        super().__init__(name, role, memory)

        #agents adaptive state
        self.confidence = 1.0
        self.history = []

    def act(self, task):
        """
        Perform a task and produce an adaptive result.
        This is where the agent 'acts' — later, this can call an LLM, API, or local tool.
        """

        status = "completed" if self.confidence > 0.5 else "attempted with uncertainty"
        result = f"{self.name} ({self.role}) {status}: {task} [confidence={self.confidence:.2f}]"

        self.history.append({
            "task":task,
            "result":result,
            "confidence": self.confidence
        })
        self.reflect(result)

        return result

