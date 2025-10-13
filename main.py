from orchestrator.orchestrator import Orchestrator
from memory.memory_module import SharedMemory

if __name__ == "__main__":
    print("\n========== MYNDRA ORCHESTRATION RUN ==========")

    # Initialize shared memory
    memory = SharedMemory()

    # Initialize orchestrator with LLM planner enabled
    orch = Orchestrator(None, memory, use_llm=True)

    # Set your goal
    goal = "Analyze performance metrics"

    # Run the full orchestration pipeline
    adaptation_summary = orch.run(goal)

    print("\n==============================================")