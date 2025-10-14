from agents.moldable_agent import MoldableAgent
from memory.memory_module import SharedMemory

memory = SharedMemory()
agent = MoldableAgent("TestAgent", "Analyst", memory)

print(agent.act("analyze new dataset"))