from collections import deque
import networkx as nx
from typing import List, Tuple
#short term memory
from collections import deque

class EpisodicMemory:
    def __init__(self, max_length=50, strict_mode=False):
        if max_length <= 0:
            raise ValueError("Max Length must be > 0")
        self.memory = {}  # agent_id: deque of recent strings
        self.max_length = max_length
        self.strict_mode = strict_mode

    def store(self, agent_id: str, content: str):
        if not isinstance(agent_id, str) or not agent_id.strip():
            raise ValueError("Agent ID is missing or invalid")
            
        if not isinstance(content, str) or not content.strip():
            return  # ignore empty

        if agent_id not in self.memory:
            if self.strict_mode:
                raise ValueError(f"Agent '{agent_id}' not in memory")
            self.memory[agent_id] = deque(maxlen=self.max_length)

        # store content (auto-evicts oldest if full)
        self.memory[agent_id].append(content)
        


    def get_recent(self, agent_id:str, n:int) -> List[str]:
        #return last n memories for the agent
        pass
    def register_agent(self, query:str) -> List[Tuple[str,str]]:
        #return all memories matching a keyword
        pass

#knowledge graph, long term

class knowledge_Graph:
    def __init__(self):
        self.graph = nx.DiGraph()

    def add_node(self, agent_id:str,content:str,context:Optional[str]):
        #add a node(observation) and optionally link to context
        pass
    def get_node(self, query:str) -> List[str]:
        #search graph nodes by keyword
        pass
    def get_related(self, query:str, depth: int = 1) -> List[str]:
        #find related nodes from a starting query node
        pass

#shared memory

class Shared_Memory:
    def __init__(self):
        self.short_term = EpisodicMemory()
        self.long_term = knowledge_Graph()
    
    def write(self, agent_id:str, content:str, context:Optional[str] = None):
        #store content into both short and long term memory
        pass
    def retrieve(self, agent_id:str, query:str) -> List[str]:
        #query both memory types and combine results
        pass
    def get_recent(self, agent_id:str, n:int = 5) -> List[str]:
        #get last n short-term entries
        pass