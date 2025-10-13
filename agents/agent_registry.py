from agents.base_agent import BaseAgent
from agents.moldable_agent import MoldableAgent

#registry mapping string names to agent classes

AGENT_REGISTRY = {
    "DataAgent":BaseAgent,
    "AnalystAgent":BaseAgent,
    "SummarizerAgent":BaseAgent,
    "GeneralAgent":MoldableAgent
}

def get_agent(agent_name, memory):
    """Retrieve an agent class by name and instantiate it. """
    if agent_name not in AGENT_REGISTRY:
        raise ValueError(f"Unknown agent name: {agent_name}")

    AgentClass = AGENT_REGISTRY[agent_name]
    return AgentClass(name=agent_name, role=agent_name, memory=memory)