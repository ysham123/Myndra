from collections import deque
import datetime
from datetime import timezone
import networkx as nx
from typing import List, Tuple, Optional
#short term memory

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
        

    def get_recent(self, agent_id: str, n: int) -> List[str]:
    # Validate inputs
        if not isinstance(agent_id, str) or not agent_id.strip():
            raise ValueError(f"Invalid agent ID: {agent_id!r}")
        if n < 0:
            raise ValueError("n must be >= 0")
        if n == 0:
            return []

    # Check memory presence
        if agent_id not in self.memory:
            if self.strict_mode:
                raise KeyError(f"Agent {agent_id} not in memory")
            return []

    # Retrieve last n entries
        buf = self.memory[agent_id]
        k = min(n, len(buf))
        return list(buf)[-k:]


        
    def retrieve(self, query: str) -> List[Tuple[str, str]]:
        q = (query or "").strip()
        if not q:
            return []

        q_lower = q.lower()
        results: List[Tuple[str, str]] = []

        for agent_id, mem_queue in self.memory.items():  # relies on dict insertion order
            for mem in mem_queue:                        # deque iterates oldest → newest
                if q_lower in mem.lower():
                    results.append((agent_id, mem))

        return results

#knowledge graph, long term
class KnowledgeGraph:
    def __init__(self, name="MyndraKG", created_at_iso=None):
        self.graph = nx.DiGraph()
        self.graph.graph["name"] = name
        self.graph.graph["schema_version"] = "0.1"
        self.graph.graph["created_at"] = created_at_iso or datetime.datetime.now(timezone.utc).isoformat()
        self.graph.graph["nodes_added"] = 0
        self.graph.graph["edges_added"] = 0

    def _now_iso(self):
        return datetime.datetime.now(timezone.utc).isoformat()

    def _norm(self, s: Optional[str]) -> str:
        return (s or "").strip()

    def _upsert_content_node(self, content_key: str, agent_id: Optional[str] = None):
        if not content_key:
            return False
        if content_key not in self.graph:
            self.graph.add_node(
                content_key,
                agent_ids=set([agent_id]) if agent_id else set(),
                tags=set(),
                created_at=self._now_iso(),
                updated_at=self._now_iso(),
            )
            self.graph.graph["nodes_added"] += 1
        else:
            node_data = self.graph.nodes[content_key]
            if agent_id:
                node_data.setdefault("agent_ids", set()).add(agent_id)
            node_data["updated_at"] = self._now_iso()
        return True

    def add_node(self, agent_id: str, content: str, context: Optional[str] = None):
        content_key = self._norm(content)
        if not content_key:
            return
        self._upsert_content_node(content_key, agent_id=agent_id)
        context_key = self._norm(context)
        if context_key and context_key != content_key:
            self._upsert_content_node(context_key, agent_id=agent_id)
            self.add_edge(context_key, content_key, edge_type="context")

    def add_edge(self, src: str, dst: str, *, edge_type: str, **attrs):
        src_key = self._norm(src)
        dst_key = self._norm(dst)
        if not src_key or not dst_key:
            return
        self._upsert_content_node(src_key)
        self._upsert_content_node(dst_key)
        edge_attrs = {"type": edge_type, "created_at": self._now_iso(), **attrs}
        self.graph.add_edge(src_key, dst_key, **edge_attrs)
        self.graph.graph["edges_added"] += 1

    def get_node_by_id(self, node_id: str) -> Optional[dict]:
        if node_id not in self.graph:
            return None
        attrs = dict(self.graph.nodes[node_id])
        attrs["id"] = node_id
        return attrs

    def get_node(self, query: str) -> List[str]:
        q = self._norm(query).lower()
        if not q:
            return []
        return [node_id for node_id in self.graph.nodes if q in node_id.lower()]

    def get_related(self, query: str, *, depth: int = 1, direction: str = "out") -> List[str]:
        seeds = self.get_node(query)
        if not seeds or depth <= 0:
            return []
        visited = set(seeds)
        queue = [(s, 0) for s in seeds]
        results: List[str] = []

        def neighbors(n):
            if direction == "out":
                return self.graph.successors(n)
            elif direction == "in":
                return self.graph.predecessors(n)
            else:
                return list(self.graph.successors(n)) + list(self.graph.predecessors(n))

        head = 0
        while head < len(queue):
            node, d = queue[head]
            head += 1
            if d >= depth:
                continue
            for nbr in neighbors(node):
                if nbr in visited:
                    continue
                visited.add(nbr)
                results.append(nbr)
                queue.append((nbr, d + 1))
        return results

#shared memory

class SharedMemory:
    def __init__(self):
        self.short_term = EpisodicMemory()
        self.long_term = KnowledgeGraph()

    def write(self, agent_id:str, content:str, context:Optional[str] = None):
        #store content into both short and long term memory
        pass
    def retrieve(self, agent_id:str, query:str) -> List[str]:
        #query both memory types and combine results
        pass
    def get_recent(self, agent_id:str, n:int = 5) -> List[str]:
        #get last n short-term entries
        pass