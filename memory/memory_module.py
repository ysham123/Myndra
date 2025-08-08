from collections import deque
import networkx as nx
#short term memory
class EpisodicMemory:
    def __init__(self, max_length=50):
        self.memory = {} #agent id:deque of recent strings
    
    def store(self, agent_id:str, content:str):
        #store new memory string for agent
        pass
    def get_recent(self, agent_id:str, n:int) -> List[str]:
        #return last n memories for the agent
        pass
    def recieve(self, query:str) -> List[Tuple[str,str]]:
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