from sqlite3.dbapi2 import Timestamp
import time
from agents.base_agent import BaseAgent
from datetime import datetime, timezone
import random

class DataAgent(BaseAgent):
    """DataAgent specializes in data access, validation, and summarization tasks. Simulates ETL logic-later this can connect to an actual
    data sources"""

    def __init__(self, name, role, memory):
        super().__init__(name, role, memory)
        self.confidence = 0.8
        self.history = []

    def act(self, task):
        """simulate performing data-related tasks (collection, cleaning, validation)."""
        timestamp = datetime.now(timezone.utc).isoformat()

        success = random.random() > 0.1
        status = "successfully processed" if success else "encountered data issue"
        confidence_change = 0.05 if success else -0.1

        self.confidence = max(0.1, min(1.0, self.confidence + confidence_change))
        result = (
            f"{self.name} ({self.role}) {status} task: '{task}' "
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
        """ Adapt based on feedback: adjust confidence and record to memory."""
        if "error" in feedback.lower() or "issue" in feedback.lower():
            self.confidence = max(0.1, self.confidence * 0.9)
            update = f"[{self.name}] Decreased confidence to {self.confidence:.2f} (feedback: {feedback})"
        elif "good" in feedback.lower() or "success" in feedback.lower():
            self.confidence = min(1.0, self.confidence * 1.05)
            update = f"[{self.name}] Increased confidence to {self.confidence:.2f} (feedback: {feedback})"
        else:
            update = f"[{self.name}] No confidence change ({self.confidence:.2f}). Feedback: {feedback}"

        self.memory.write(self.name, update)
        print(update)