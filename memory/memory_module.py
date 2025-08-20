from collections import deque
import datetime
from datetime import timezone
import json
import re
import networkx as nx
from typing import List, Tuple, Optional

from .memory_types import (
    Defaults, EdgeType,
    normalize_id, ensure_agent_prefix, ensure_task_id, decay_linear
)

class EpisodicMemory:
    """In-memory store of recent events per agent.

    - strict_mode=True: accessing unknown agents raises KeyError.
    - Events are dicts with keys: timestamp, agent_id, content.
    """
    def __init__(self, max_length=Defaults.EP_MAX_LENGTH, strict_mode=Defaults.STRICT_MODE):
        if max_length <= 0:
            raise ValueError("Max Length must be > 0")
        self.memory = {}
        self.max_length = max_length
        self.strict_mode = strict_mode

    def store(self, agent_id: str, content: str):
        """Store an event for an agent."""
        if not isinstance(agent_id, str) or not agent_id.strip():
            raise ValueError("Agent ID is missing or invalid")
        if not isinstance(content, str) or not content.strip():
            return
        if agent_id not in self.memory:
            if self.strict_mode:
                raise KeyError(f"Agent '{agent_id}' not in memory")
            self.memory[agent_id] = deque(maxlen=self.max_length)
        event = {
            "timestamp": datetime.datetime.now(timezone.utc).isoformat(),
            "agent_id": agent_id,
            "content": content,
        }
        self.memory[agent_id].append(event)

    def get_recent(self, agent_id: str, n: int = 5) -> List[dict]:
        """Return up to n most recent events for agent.
        In strict mode, raises KeyError if agent has no events.
        """
        if agent_id not in self.memory:
            if self.strict_mode:
                raise KeyError(f"Agent '{agent_id}' not in memory")
            return []
        return list(self.memory[agent_id])[-n:]

    def retrieve(self, agent_id: str, query: str) -> List[dict]:
        """Retrieve events for an agent containing the query string."""
        if agent_id not in self.memory:
            return []
        q = query.lower()
        events = [e for e in self.memory[agent_id] if q in e["content"].lower()]
        return events


class KnowledgeGraph:
    """Directed graph of normalized text nodes with simple provenance.

    - add_node(agent_id, content, context): adds nodes and edge context -> content
    - get_related(node, depth): returns list of {src, dst} edges by traversing predecessors
    """
    def __init__(self):
        self.graph = nx.DiGraph()

    def add_node(self, agent_id: str, content: str, context: Optional[str] = None, timestamp: Optional[str] = None, tags=None):
        """Add a content node (and optional context) with provenance.
        - agent_id: the agent responsible (stored as provenance on nodes)
        - content: main node text
        - context: optional context node; creates edge context -> content
        """
        content_id = normalize_id(content)
        if not content_id:
            return
        if content_id not in self.graph:
            self.graph.add_node(content_id, agent_ids=set(), timestamps=[], tags=set())
        if agent_id:
            self.graph.nodes[content_id]["agent_ids"].add(agent_id)
        if timestamp:
            self.graph.nodes[content_id]["timestamps"].append(timestamp)
        if tags:
            self.graph.nodes[content_id]["tags"].update(tags)
        if context:
            ctx_id = normalize_id(context)
            if ctx_id not in self.graph:
                self.graph.add_node(ctx_id, agent_ids=set(), timestamps=[], tags=set())
            if agent_id:
                self.graph.nodes[ctx_id]["agent_ids"].add(agent_id)
            if timestamp:
                self.graph.nodes[ctx_id]["timestamps"].append(timestamp)
            if tags:
                self.graph.nodes[ctx_id]["tags"].update(tags)
            self.add_edge(ctx_id, content_id, edge_type=EdgeType.CONTEXT)

    def add_edge(self, src: str, dst: str, edge_type: EdgeType = EdgeType.CONTEXT):
        """Add an edge between two nodes."""
        src, dst = normalize_id(src), normalize_id(dst)
        if not src or not dst:
            return
        if src not in self.graph:
            self.graph.add_node(src, agent_ids=set(), timestamps=[], tags=set())
        if dst not in self.graph:
            self.graph.add_node(dst, agent_ids=set(), timestamps=[], tags=set())
        self.graph.add_edge(src, dst, type=edge_type)

    def get_node(self, node_id: str) -> dict:
        """Get a node by its ID."""
        node_id = normalize_id(node_id)
        return self.graph.nodes.get(node_id, {})

    def get_related(self, node_id: str, depth: int = Defaults.KG_DEFAULT_DEPTH) -> List[dict]:
        """BFS over predecessors up to depth, returning edge dicts {src, dst}.
        src is the queried/current node; dst is a predecessor (e.g., context of content).
        """
        node_id = normalize_id(node_id)
        if node_id not in self.graph:
            return []
        visited = set([node_id])
        queue = [(node_id, 0)]
        related = []
        while queue:
            current, d = queue.pop(0)
            if d >= depth:
                continue
            # Use predecessors so that if context -> content, querying content returns its contexts
            for neighbor in self.graph.predecessors(current):
                related.append({"src": current, "dst": neighbor})
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, d + 1))
        return related


class SharedMemory:
    """Facade combining episodic memory and knowledge graph.

    - write(): stores event and updates KG (context -> content; task -> content)
    - get_recent(): delegates to episodic memory (agent prefix enforced)
    - retrieve(): returns list of {"content": str} from hybrid search
    """
    def __init__(self, policy=Defaults):
        self.policy = policy
        self.short_term = EpisodicMemory(max_length=policy.EP_MAX_LENGTH, strict_mode=policy.STRICT_MODE)
        self.long_term = KnowledgeGraph()

    def write(self, agent_id: str, content: str, context: Optional[str] = None):
        """Persist content for agent, optionally linking a context.
        Also extracts 'task\\d+' and links task -> content.
        """
        agent_id = ensure_agent_prefix(agent_id, self.policy.AGENT_PREFIX)
        self.short_term.store(agent_id, content)
        timestamp = datetime.datetime.now(timezone.utc).isoformat()
        # add to KG with provenance and optional context edge
        self.long_term.add_node(agent_id, content, context=context, timestamp=timestamp)
        task_match = re.search(r"task(\d+)", content, re.IGNORECASE)
        if task_match:
            task_id = ensure_task_id(task_match.group(1), self.policy.TASK_PREFIX)
            # Add the task node and connect it to the content
            self.long_term.add_node(agent_id, task_id, timestamp=timestamp)
            self.long_term.add_edge(task_id, content, edge_type=EdgeType.ASSIGNED_TO)

    def get_recent(self, agent_id: str, n: int = 5) -> List[dict]:
        """Return recent episodic events for the agent (with agent prefix normalization)."""
        agent_id = ensure_agent_prefix(agent_id, self.policy.AGENT_PREFIX)
        return self.short_term.get_recent(agent_id, n)

    def retrieve(self, agent_id: str, query: str) -> List[dict]:
        """Hybrid retrieval, returning list of dicts {"content": str}."""
        ep_hits = self.search_episodes(agent_id, query)
        kg_hits = self.search_kg(query)
        combined = ep_hits[: self.policy.EP_BUCKET_CAP] + kg_hits[: self.policy.KG_BUCKET_CAP]
        seen = set()
        results = []
        for content, score in sorted(combined, key=lambda x: x[1], reverse=True):
            if content not in seen:
                results.append({"content": content})
                seen.add(content)
            if len(results) >= self.policy.FINAL_TOPK:
                break
        return results

    def search_episodes(self, agent_id: str, query: str) -> List[Tuple[str, float]]:
        events = self.short_term.retrieve(agent_id, query)
        scored = []
        now = datetime.datetime.now(timezone.utc)
        for e in events:
            ts = datetime.datetime.fromisoformat(e["timestamp"])
            age_days = (now - ts).days
            score_time = decay_linear(age_days, self.policy.DECAY_WINDOW_DAYS)
            score = (self.policy.EP_SCORE_KEYWORD_W * (query.lower() in e["content"].lower())) + (
                self.policy.EP_SCORE_TIME_W * score_time
            )
            scored.append((e["content"], score))
        return sorted(scored, key=lambda x: x[1], reverse=True)

    def search_kg(self, query: str) -> List[Tuple[str, float]]:
        results = []
        q = normalize_id(query)
        for node in self.long_term.graph.nodes:
            if q in node:
                score = self.policy.KG_SCORE_KEYWORD_W
                results.append((node, score))
        return sorted(results, key=lambda x: x[1], reverse=True)
