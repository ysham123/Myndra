from orchestrator.orchestrator import Orchestrator
from memory.memory_module import SharedMemory

# For now, skip AgentRegistry until agent layer is implemented
memory = SharedMemory()

# Placeholder since agents aren't ready yet
registry = None  

orch = Orchestrator(registry, memory)

subtasks = orch.plan("Analyze system performance metrics")
print("Subtasks:", subtasks)
print("Recent memory:", memory.get_recent("orchestrator"))