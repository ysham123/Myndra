from orchestrator.orchestrator import Orchestrator
from memory.memory_module import SharedMemory

if __name__ == "__main__":
    print("\n========== MYNDRA ORCHESTRATION RUN ==========\n")

    # 1️⃣ Initialize shared memory
    memory = SharedMemory()

    # 2️⃣ Create the orchestrator (LLM planner on)
    orch = Orchestrator(None, memory, use_llm=True)

    # 3️⃣ Define a high-level goal
    goal = "Analyze performance metrics"

    # 4️⃣ Run full orchestration pipeline
    adaptation_summary = orch.run(goal)

    # 5️⃣ Print memory trace
    print("\n==============================================\n")

    # Optional: view recent orchestrator memory
    recent = memory.get_recent("agent:orchestrator")
    print("Recent Memory Trace (Orchestrator):")
    for item in recent[-5:]:
        print(f"• {item['timestamp']} | {item['content']}")