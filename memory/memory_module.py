from collections import deque
import datetime
from datetime import timezone
import json
import re
from hashlib import sha1
import networkx as nx
from typing import List, Tuple, Optional


class EpisodicMemory:
    """
    Short-term per-agent memory.

    Invariants:
      - self.memory preserves insertion order by agent_id.
      - Each agent_id maps to a deque of strings (oldest -> newest).
      - Schema for stored JSON rows is stable: {episode_id, agent_id, content, context, timestamp}.
        Unknown keys are ignored by readers.
    """
    def __init__(self, max_length=50, strict_mode=False):
        if max_length <= 0:
            raise ValueError("Max Length must be > 0")
        self.memory = {}  # agent_id -> deque[str]
        self.max_length = max_length
        self.strict_mode = strict_mode

    def store(self, agent_id: str, content: str):
        """
        Append an item to an agent's deque (auto-evicts oldest when full).

        Errors:
          - ValueError for invalid inputs (bad types/empty strings).
          - KeyError if strict_mode=True and agent_id not present (unknown key).
        """
        if not isinstance(agent_id, str) or not agent_id.strip():
            raise ValueError("Agent ID is missing or invalid")
        if not isinstance(content, str) or not content.strip():
            return
        if agent_id not in self.memory:
            if self.strict_mode:
                # Unknown key in strict mode → KeyError (not ValueError).
                raise KeyError(f"Agent '{agent_id}' not in memory")
            self.memory[agent_id] = deque(maxlen=self.max_length)
        self.memory[agent_id].append(content)

    def get_recent(self, agent_id: str, n: int) -> List[str]:
        """Return last n items (oldest->newest slice of size n)."""
        if not isinstance(agent_id, str) or not agent_id.strip():
            raise ValueError(f"Invalid agent ID: {agent_id!r}")
        if n < 0:
            raise ValueError("n must be >= 0")
        if n == 0:
            return []
        if agent_id not in self.memory:
            if self.strict_mode:
                raise KeyError(f"Agent {agent_id} not in memory")
            return []
        buf = self.memory[agent_id]
        k = min(n, len(buf))
        return list(buf)[-k:]

    def retrieve(self, query: str) -> List[Tuple[str, str]]:
        """
        Substring search (case-insensitive) over all agents' deques.

        Returns: list of (agent_id, matching_text) in insertion order.
        """
        q = (query or "").strip()
        if not q:
            return []
        q_lower = q.lower()
        results: List[Tuple[str, str]] = []
        # Non-JSON rows are treated as raw text (see search_episodes).
        for agent_id, mem_queue in self.memory.items():
            for mem in mem_queue:
                if q_lower in mem.lower():
                    results.append((agent_id, mem))
        return results


class KnowledgeGraph:
    """
    Content-centric directed knowledge graph.

    Invariants:
      - Node IDs are **normalized content strings** (strip applied).
      - Node attrs keep provenance and tags as **Python sets** for speed.
        NOTE: sets are not JSON/GraphML serializable — convert to (sorted) lists on export.
      - get_node(query) performs case-insensitive substring search over node IDs.
      - get_node_by_id(id) is exact ID lookup.
      - get_related(seed_id, ...) does BFS from a **seed node ID** (two-step: search → pick seed → traverse).
      - Neighbor expansion order is deterministic via NetworkX insertion order.

    Edge timestamp:
      - Canonical key is 'created_at'. We may carry an additional 'timestamp' from the writer,
        but readers (edge_recency) prefer 'created_at' and only fall back to 'timestamp'.
    """
    def __init__(self, name="MyndraKG", created_at_iso=None):
        self.graph = nx.DiGraph()
        self.graph.graph["name"] = name
        self.graph.graph["schema_version"] = "0.1"
        self.graph.graph["created_at"] = created_at_iso or datetime.datetime.now(timezone.utc).isoformat()
        self.graph.graph["nodes_added"] = 0
        self.graph.graph["edges_added"] = 0

    # --- helpers ---
    def _now_iso(self):
        return datetime.datetime.now(timezone.utc).isoformat()

    def _norm(self, s: Optional[str]) -> str:
        return (s or "").strip()

    def _upsert_content_node(self, content_key: str, agent_id: Optional[str] = None, **attrs):
        """
        Create or update a content node (ID = normalized content_key).
        Merges provenance (agent_ids) and tags. Updates updated_at.
        """
        if not content_key:
            return False
        if content_key not in self.graph:
            self.graph.add_node(
                content_key,
                agent_ids=set([agent_id]) if agent_id else set(),
                tags=set(),
                created_at=self._now_iso(),
                updated_at=self._now_iso(),
                **attrs,
            )
            # If a 'type' attribute is provided, also mirror it into tags and set 'type'.
            if "type" in attrs:
                self.graph.nodes[content_key]["tags"].add(attrs["type"])
                self.graph.nodes[content_key]["type"] = attrs["type"]
            self.graph.graph["nodes_added"] += 1
        else:
            node_data = self.graph.nodes[content_key]
            if agent_id:
                node_data.setdefault("agent_ids", set()).add(agent_id)
            if "type" in attrs:
                node_data.setdefault("tags", set()).add(attrs["type"])
                node_data["type"] = attrs["type"]
            for k, v in attrs.items():
                if k != "type":
                    node_data[k] = v
            node_data["updated_at"] = self._now_iso()
        return True

    # --- public API ---
    def add_node(self, agent_id: str, content: str, context: Optional[str] = None, **attrs):
        """
        Upsert a content node (ID = norm(content)).
        If non-empty context is provided and differs from content, also upsert the context node
        and add a 'context' edge: context → content.
        """
        content_key = self._norm(content)
        if not content_key:
            return
        self._upsert_content_node(content_key, agent_id=agent_id, **attrs)
        context_key = self._norm(context)
        if context_key and context_key != content_key:
            self._upsert_content_node(context_key, agent_id=agent_id)
            self.add_edge(context_key, content_key, edge_type="context")

    def add_edge(self, src: str, dst: str, *, edge_type: str, **attrs):
        """
        Add an edge between normalized node IDs. No-ops if endpoints normalize to empty.
        Uses normalized keys for both node upsert and edge insertion (prevents duplicate IDs).
        """
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
        """Exact node lookup by ID. Returns attrs + 'id'."""
        if node_id not in self.graph:
            return None
        attrs = dict(self.graph.nodes[node_id])
        attrs["id"] = node_id
        return attrs

    def get_node(self, query: str) -> List[str]:
        """Case-insensitive substring search over node IDs (returns IDs in insertion order)."""
        q = self._norm(query).lower()
        if not q:
            return []
        return [node_id for node_id in self.graph.nodes if q in node_id.lower()]

    def get_related(self, node_id: str, *, depth: int = 1, direction: str = "out", limit: Optional[int] = None) -> List[dict]:
        """
        BFS traversal starting from an **exact seed node_id**.
        Two-step usage: ids = get_node(query); then pick one id and call get_related(id, depth=...).

        direction: 'out' | 'in' | 'both'
        Returns a list of dicts with {src, dst, type, edge, node}, excluding the seed itself.
        """
        node_id = self._norm(node_id)
        if node_id not in self.graph or depth <= 0:
            return []

        def neighbors(n):
            if direction == "out":
                return [(n, nbr, self.graph[n][nbr]) for nbr in self.graph.successors(n)]
            elif direction == "in":
                return [(nbr, n, self.graph[nbr][n]) for nbr in self.graph.predecessors(n)]
            else:
                outs = [(n, nbr, self.graph[n][nbr]) for nbr in self.graph.successors(n)]
                ins = [(nbr, n, self.graph[nbr][n]) for nbr in self.graph.predecessors(n)]
                return outs + ins

        visited = {node_id}
        queue = [(node_id, 0)]
        out: List[dict] = []
        head = 0

        while head < len(queue):
            cur, d = queue[head]
            head += 1
            if d >= depth:
                continue
            for src, dst, eattrs in neighbors(cur):
                # pick the true neighbor node we should visit next
                neighbor = dst if src == cur else src
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                out.append({
                    "src": src,
                    "dst": dst,
                    "type": eattrs.get("type", "related"),
                    "edge": dict(eattrs),
                    "node": self.get_node_by_id(neighbor) or {"id": neighbor},
                })
                queue.append((neighbor, d + 1))
                if limit is not None and len(out) >= limit:
                    return out
        return out


def now_iso():
    return datetime.datetime.now(timezone.utc).isoformat()


def mk_episode_id(agent_id, content, timestamp):
    # hash-based ID to reduce accidental duplicates
    base = f"{agent_id}|{timestamp}|{(content or '')[:64]}"
    return sha1(base.encode()).hexdigest()


def jaccard(query, text):
    """
    Very simple token-overlap score (v0.1). Brittle for morphology/semantics.
    TODO: replace with embeddings-based similarity.
    """
    q = set((query or "").lower().split())
    t = set((text or "").lower().split())
    if not q or not t:
        return 0.0
    inter = q & t
    union = q | t
    return len(inter) / len(union)


def age_in_days(row):
    """
    Parse JSON row and compute age in days from 'timestamp' field.
    Non-JSON rows return 0.0 (treated as fresh by decay()).
    """
    if isinstance(row, str):
        try:
            data = json.loads(row)
            timestamp = data.get("timestamp")
        except json.JSONDecodeError:
            return 0.0
    else:
        return 0.0
    if not timestamp:
        return 0.0
    try:
        dt = datetime.datetime.fromisoformat(timestamp)
        now = datetime.datetime.now(timezone.utc)
        delta = now - dt
        return delta.total_seconds() / (24 * 3600)
    except Exception:
        return 0.0


def decay(age_days):
    """
    Linear freshness decay with ~30-day shelf life.
    At 15 days, score ~0.5; at 30 days, score ~0.
    """
    if age_days <= 0:
        return 1.0
    return max(0.0, 1.0 - (age_days / 30.0))


def topk(items, k):
    return sorted(items, reverse=True)[:k]


def dedupe(items):
    seen = set()
    result = []
    for score, text in items:
        if text not in seen:
            seen.add(text)
            result.append((score, text))
    return result


def normalize(text):
    return (text or "").lower().strip()


def match_nodes(kg, query):
    return kg.get_node(query)


def fmt_node(kg, node_id):
    node = kg.get_node_by_id(node_id)
    if not node:
        return f"Node: {node_id}"
    return f"Node: {node['id']} (Type: {node.get('type', 'unknown')})"


def fmt_edge(rel):
    return f"Relation: {rel.get('src', '?')} --[{rel.get('type', '?')}]--> {rel.get('dst', '?')}"


def pretty(row):
    if isinstance(row, str):
        try:
            data = json.loads(row)
            return f"{data.get('agent_id', '?')}: {data.get('content', '?')}"
        except json.JSONDecodeError:
            return row
    return str(row)


def edge_recency(rel: dict) -> float():
    """
    Recency score for edges.
    Canonical timestamp: 'created_at'. Falls back to 'timestamp' if needed.
    """
    edge = rel.get("edge", {})
    ts = edge.get("created_at") or edge.get("timestamp")
    if not ts:
        return 0.5
    try:
        dt = datetime.datetime.fromisoformat(ts)
        now = datetime.datetime.now(timezone.utc)
        age_days = (now - dt).total_seconds() / (24 * 3600)
    except Exception:
        return 0.5
    return max(0.0, 1.0 - (age_days / 30.0))


class SharedMemory:
    """
    Facade over short-term (episodic) and long-term (knowledge) memory.

    Workflow (write):
      1) Store an episodic JSON row: {episode_id, agent_id, content, context, timestamp}.
      2) Extract lightweight entities/relations (stubbed NER).
      3) Upsert entities as content nodes; insert relations as edges.

    Retrieval:
      - Episodes: keyword+recency scoring (jaccard + linear decay).
      - KG: substring-match nodes, then traverse small neighborhoods.
    """
    def __init__(self):
        self.short_term = EpisodicMemory()
        self.long_term = KnowledgeGraph()
        # Stub extractor: task(\d+)
        self.PAT_TASK = re.compile(r"\btask\s*(\d+)\b", re.I)

    def write(self, agent_id: str, content: str, context: Optional[str] = None):
        # Normalize agent id once for consistency (e.g., "agent:alice")
        agent_key = agent_id if agent_id.startswith("agent:") else f"agent:{agent_id}"
        ts = now_iso()
        ep_id = mk_episode_id(agent_key, content, ts)
        # Stable episodic schema; unknown keys ignored by readers.
        self.short_term.store(agent_key, json.dumps({
            "episode_id": ep_id,
            "agent_id": agent_key,
            "content": content,
            "context": context,
            "timestamp": ts
        }))
        # Stub NER; future: agents, tools, docs, etc.
        ents, rels = self.extract(agent_key, content, context)
        for e in ents:
            attrs = {"type": e.get("type")} if e.get("type") else {}
            # Use actual context so a context → entity edge is created
            self.long_term.add_node(agent_key, content=e["id"], context=context, **attrs)
        for r in rels:
            self.long_term.add_edge(
                r["src"], r["dst"],
                edge_type=r["type"],
                from_episode_id=ep_id,
                timestamp=ts,    # extra; readers prefer created_at and fallback here
                evidence=content[:160]
            )

    def search_episodes(self, agent_id: str, q: str, M: int = 200, k: int = 5):
        """
        Score recent M rows for agent by keyword overlap + recency decay.
        Non-JSON rows are treated as raw text (see try/except).
        """
        items = self.short_term.get_recent(agent_id, M)
        scored = []
        for row in items:
            try:
                data = json.loads(row) if isinstance(row, str) else row
                text = data.get("content", "")
            except Exception:
                # Non-JSON row -> treat as raw text.
                text = row if isinstance(row, str) else str(row)
            score_kw = jaccard(q, text)
            age_days_val = age_in_days(row)
            # 30-day shelf life; jaccard is brittle but OK for v0.1.
            score = 0.7 * score_kw + 0.3 * decay(age_days_val)
            if score > 0:
                scored.append((score, f"[EP]{text}"))
        return topk(scored, k)

    def search_kg(self, q: str, k: int = 5):
        """
        Substring-match nodes, then score small outgoing neighborhoods by
        (keyword match on dst node title/id) and edge recency.
        """
        hits = match_nodes(self.long_term, q)
        cand = []
        for node_id in hits:
            cand.append((0.9, f"[KG] {fmt_node(self.long_term, node_id)}"))
            for rel in self.long_term.get_related(node_id, direction="out", limit=3):
                node_title = (rel.get("node") or {}).get("title") or (rel.get("node") or {}).get("id") or rel.get("dst")
                s = 0.6 * jaccard(q, node_title or "") + 0.4 * edge_recency(rel)
                cand.append((s, f"[KG] {fmt_edge(rel)}"))
        return topk(cand, k)

    def retrieve(self, agent_id: str, query: str):
        """Merge episodic and KG hits, dedupe by text, return top 10."""
        agent_key = agent_id if agent_id.startswith("agent:") else f"agent:{agent_id}"
        q = normalize(query)
        ep = self.search_episodes(agent_key, q)
        kg = self.search_kg(q)
        return [t for _, t in topk(dedupe(ep + kg), 10)]

    def get_recent(self, agent_id, n=5):
        agent_key = agent_id if agent_id.startswith("agent:") else f"agent:{agent_id}"
        rows = self.short_term.get_recent(agent_key, n)
        return [pretty(r) for r in rows]

    def extract(self, agent_id: str, content: str, context: Optional[str]):
        """
        Very light regex-based NER stub.
        Current patterns:
          - task(\d+) → entity id 'task:{n}', relation agent:{agent_id} --[MENTIONS]--> task:{n}
        """
        # agent_id is already normalized (e.g., "agent:alice")
        ents, rels = [], []
        for m in self.PAT_TASK.finditer(content):
            task_id = f"task:{m.group(1)}"
            ents.append({"id": task_id, "type": "task"})
            rels.append({"src": agent_id, "dst": task_id, "type": "MENTIONS"})
        return ents, rels
