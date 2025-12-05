"""Agent registry for creating and managing agent instances."""

from agents.base_agent import BaseAgent
from agents.data_agent import DataAgent
from agents.analyst_agent import AnalystAgent
from agents.summarizer_agent import SummarizerAgent
from agents.general_agent import GeneralAgent
from agents.moldable_agent import MoldableAgent

# Agent class registry mapping names to classes
AGENT_REGISTRY = {
    "DataAgent": DataAgent,
    "AnalystAgent": AnalystAgent,
    "SummarizerAgent": SummarizerAgent,
    "GeneralAgent": GeneralAgent,
    "MoldableAgent": MoldableAgent,
}

# Alias mapping for flexible agent name resolution
AGENT_ALIASES = {
    "analyst": "AnalystAgent",
    "planner": "GeneralAgent",
    "executor": "GeneralAgent",
    "summarizer": "SummarizerAgent",
    "data": "DataAgent",
    "general": "GeneralAgent",
    "moldable": "MoldableAgent",
    "adaptive": "MoldableAgent",
}

# Agent configuration: name -> (display_name, role_description)
AGENT_CONFIG = {
    "AnalystAgent": ("AnalystAgent", "Analyst"),
    "DataAgent": ("DataAgent", "Data Engineer"),
    "SummarizerAgent": ("SummarizerAgent", "Summarizer"),
    "GeneralAgent": ("GeneralAgent", "Generalist"),
    "MoldableAgent": ("MoldableAgent", "Adaptive"),
}

def get_agent(agent_name: str, memory):
    """Return an initialized agent instance by name.
    
    Args:
        agent_name: Name or alias of the agent to create
        memory: Shared memory instance for the agent
    
    Returns:
        Initialized agent instance
    
    Raises:
        ValueError: If agent_name is not recognized
    """
    # Normalize agent name via alias mapping
    normalized_name = AGENT_ALIASES.get(agent_name.lower(), agent_name)
    
    # Get agent class and configuration
    if normalized_name not in AGENT_REGISTRY:
        raise ValueError(
            f"Unknown agent name: '{agent_name}'. "
            f"Available agents: {list(AGENT_REGISTRY.keys())}"
        )
    
    agent_class = AGENT_REGISTRY[normalized_name]
    display_name, role = AGENT_CONFIG[normalized_name]
    
    # Instantiate and return agent
    return agent_class(display_name, role, memory)