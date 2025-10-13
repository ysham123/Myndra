from datetime import datetime, timezone

class BaseAgent:
    def __init__(self, name, role, memory):
        self.name = name
        self.role = role
        self.memory = memory

    def act(self, task):
        """Perform the assigned task. Override this in subclasses."""
        raise NotImplementedError("Each agent must implement its own act() method.")

    def reflect(self, result):
        """Store the result and metadata into shared memory."""
        entry = {
            "agent_id": self.name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "content": f"[{self.role}] {result}"
        }
        self.memory.write(self.name, entry["content"])

    def __repr__(self):
        return f"<Agent {self.name} ({self.role})>"