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

    def mold(self, feedback):
        """Adap the agent's internal parameters based on feedback.
        This simulates reinforcement-improving or degrading confidence
        """

        if "error" in feedback.lower() or "fail" in feedback.lower():
            self.confidence = max(0.1, self.confidence * 0.9)
            update = f"Decreased confidence to {self.confidence:.2f} due to feedback: {feedback}"
        elif "success" in feedback.lower() or "good" in feedback.lower():
            self.confidence = min(1.0, self.confidence * 1.05)
            update = f"Increased confidence to {self.confidence:.2f} due to feedback {feedback}"
        else:
            update = f"No change to confidence ({self.confidence:.2f}). Feedback: {feedback}"
        
        self.memory.write(self.name, update)
        print(update)