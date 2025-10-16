import random
from datetime import datetime, timezone
from agents.base_agent import BaseAgent 

class AnalystAgent(BaseAgent):
    "Agent specializing in reasoning, pattern detection, and insight generation."

    def __init__(self, name, role, memory):
        super().__init__(name, role, memory)
        self.confidence = 0.9
        self.history = []

    def act(self, task):
        "simulating analytical reasoning over data"
        timestamp = datetime.now(timezone.utc).isoformat()

        #reasoning process
        success = random.random() > 0.15
        status = "derived insight" if success else "inconclusive result"
        confidence_change = 0.03 if success else -0.08

        self.confidence = max(0.1, min(0.1, self.confidence + confidence_change))
        result = (
            f"{self.name} ({self.role}) {status} for task: '{task}' "
            f"[confidence={self.confidence:.2f}]"
        )

        self.history.append({
            "task":task,
            "timestamp":timestamp,
            "status":status,
            "confidence":self.confidence
        })

        self.reflect(result)
        print(result)
        return result

    def mold(self, feedback):
        """Adapt reasoning performance based on feedback."""
        if "error" in feedback.lower() or "wrong" in feedback.lower():
            self.confidence = max(0.1, self.confidence * 0.9)
            update = f"[{self.name}] Decreased confidence to {self.confidence:.2f} (feedback: {feedback})"
        elif "good" in feedback.lower() or "insight" in feedback.lower() or "success" in feedback.lower():
            self.confidence = min(1.0, self.confidence * 1.05)
            update = f"[{self.name}] Increased confidence to {self.confidence:.2f} (feedback: {feedback})"
        else:
            update = f"[{self.name}] No change to confidence ({self.confidence:.2f}). Feedback: {feedback}"

        self.memory.write(self.name, update)
        print(update)