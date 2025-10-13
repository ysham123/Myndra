from orchestrator.orchestrator import Orchestrator
from memory.memory_module import SharedMemory

def main():
    # Initialize shared memory
    memory = SharedMemory()

    # Initialize orchestrator (no agent registry yet)
    orch = Orchestrator(None, memory)

    # Define high-level goal
    goal = "Analyze performance metrics"

    # === Full orchestration pipeline ===
    subtasks = orch.plan(goal)
    assignments = orch.assign(subtasks)
    results = orch.execute(assignments)
    adaptation = orch.adapt(results)

    # === Display results ===
    print("\n========== MYNDRA ORCHESTRATION RUN ==========")
    print(f"Goal: {goal}\n")

    print("Planned Subtasks:")
    for s in subtasks:
        print(f"  - {s}")

    print("\nAssignments:")
    for a in assignments:
        print(f"  - {a['task']} → {a['agent']}")

    print("\nExecution Results:")
    for r in results:
        print(f"  - {r['agent']} → {r['output']}")

    print("\nAdaptation Summary:")
    for ad in adaptation['adaptations']:
        print(f"  - {ad['task']} → {ad['action']}")

    print("\nRecent Memory (Orchestrator):")
    for m in memory.get_recent("orchestrator"):
        print(f"  • {m['timestamp']} | {m['content']}")

    print("\n==============================================")

if __name__ == "__main__":
    main()