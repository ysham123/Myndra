from agents.base_agent import BaseAgent
from agents.data_agent import DataAgent
from agents.moldable_agent import MoldableAgent
from agents.analyst_agent import AnalystAgent
from agents.summarizer_agent import SummarizerAgent

#registry mapping string names to agent classes

AGENT_REGISTRY = {
    "DataAgent":DataAgent,
    "AnalystAgent":AnalystAgent,
    "SummarizerAgent":SummarizerAgent,
    "GeneralAgent":MoldableAgent
}

def get_agent(agent_name, memory):
    """Return an initialized agent instance by name."""
    from agents.general_agent import GeneralAgent
    from agents.data_agent import DataAgent
    from agents.analyst_agent import AnalystAgent
    from agents.summarizer_agent import SummarizerAgent
    from agents.moldable_agent import MoldableAgent  # if used

    name = agent_name.lower()

    # Allow flexible alias mapping from planner roles → real agents
    alias_map = {
        "analyst": "AnalystAgent",
        "planner": "GeneralAgent",
        "executor": "GeneralAgent",
        "summarizer": "SummarizerAgent",
        "data": "DataAgent",
        "general": "GeneralAgent"
    }
    agent_name = alias_map.get(name, agent_name)

    # Instantiate based on normalized name
    if agent_name == "AnalystAgent":
        return AnalystAgent("AnalystAgent", "Analyst", memory)
    elif agent_name == "DataAgent":
        return DataAgent("DataAgent", "Data Engineer", memory)
    elif agent_name == "SummarizerAgent":
        return SummarizerAgent("SummarizerAgent", "Summarizer", memory)
    elif agent_name == "GeneralAgent":
        return GeneralAgent("GeneralAgent", "Generalist", memory)
    elif agent_name == "MoldableAgent":
        return MoldableAgent("MoldableAgent", "Adaptive", memory)
    else:
        raise ValueError(f"Unknown agent name: {agent_name}")