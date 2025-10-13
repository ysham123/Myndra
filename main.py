from orchestrator.orchestrator import Orchestrator
from memory.memory_module import SharedMemory

memory = SharedMemory()
orch = Orchestrator(None, memory)

# 1. Plan → 2. Assign → 3. Execute
subtasks = orch.plan("Analyze performance metrics")
assignments = orch.assign(subtasks)
results = orch.execute(assignments)

print("Results:")
for r in results:
    print(r)

print("\nRecent memory:")
print(memory.get_recent("orchestrator"))