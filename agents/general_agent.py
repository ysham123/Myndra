from agents.base_agent import BaseAgent

class GeneralAgent(BaseAgent):
    """A general-purpose agent that can handle broad tasks like planning,
    summarization, or acting as a default fallback for orchestration."""
    def __init__(self, name, role, memory):
        super().__init__(name, role, memory)
        self.confidence = 1.0

    def act(self, task):
        """Perform a general reasoning or coordination task."""
        result = f"{self.name} ({self.role}) completed: {task} [confidence={self.confidence:.2f}]"
        self.reflect(result)
        return result

    def mold(self, feedback):
        """Adjust internal confidence based on feedback."""
        if "error" in feedback.lower():
            self.confidence = max(0.1, self.confidence * 0.9)
        elif "success" in feedback.lower() or "good" in feedback.lower():
            self.confidence = min(1.0, self.confidence * 1.05)