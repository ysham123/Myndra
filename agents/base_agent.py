"""
┌───────────┐     ┌───────┐     ┌─────┐     ┌──────────┐     ┌─────────┐
│ observe   │──→──│ plan  │──→──│ act │──→──│ reflect  │──→──│ commit  │
└───────────┘     └───────┘     └─────┘     └──────────┘     └─────────┘
     ▲                                                           │
     └───────────────────── SharedMemory ────────────────────────┘
"""

#interface

class BaseAgent:
    def __init__(self, agent_id, SharedMemory, policy, config=None):
        self.agent_id = agent_id
        self.sm = SharedMemory
        self.policy = policy
        self.config = config or {}