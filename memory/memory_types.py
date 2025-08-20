from enum import Enum
import re
from typing import Optional

class EdgeType(str, Enum):
    """Types of relationships between nodes in the Knowledge Graph."""
    CONTEXT = "context"        # context -> content
    MENTIONS = "MENTIONS"      # agent -> entity
    ASSIGNED_TO = "ASSIGNED_TO" # task -> agent
    CREATED = "CREATED"        # agent -> artifact
    SUMMARIZED = "SUMMARIZED"  # agent -> doc
    DERIVED_FROM = "DERIVED_FROM" # artifact -> artifact
    NEXT = "NEXT"              # step ordering


class Defaults:
    """Centralized constants and policy knobs for memory behavior."""
    # Episodic memory
    EP_MAX_LENGTH = 50
    STRICT_MODE = False
    DECAY_WINDOW_DAYS = 30
    EP_SCORE_KEYWORD_W = 0.7
    EP_SCORE_TIME_W = 0.3

    # Knowledge Graph
    KG_DEFAULT_DEPTH = 1
    KG_NEIGHBOR_LIMIT = 3
    KG_SCORE_KEYWORD_W = 0.6
    KG_SCORE_TIME_W = 0.4

    # Hybrid retrieval
    FINAL_TOPK = 10
    EP_BUCKET_CAP = 5
    KG_BUCKET_CAP = 5

    # Normalization
    AGENT_PREFIX = "agent:"
    TASK_PREFIX = "task:"


#  Normalization helpers 

_WS = re.compile(r'\s+')


def normalize_id(text: Optional[str]) -> str:
    """Normalize text to a canonical node ID: lowercase, trimmed, single-spaced."""
    if not text:
        return ""
    return _WS.sub(" ", text.strip()).lower()


def ensure_agent_prefix(agent_id: str, prefix: str = Defaults.AGENT_PREFIX) -> str:
    """Ensure an agent ID has the correct prefix."""
    return agent_id if agent_id.startswith(prefix) else f"{prefix}{agent_id}"


def ensure_task_id(numeric: str, prefix: str = Defaults.TASK_PREFIX) -> str:
    """Ensure a task ID has the correct prefix."""
    return numeric if numeric.startswith(prefix) else f"{prefix}{numeric}"


def decay_linear(age_days: float, window: int = Defaults.DECAY_WINDOW_DAYS) -> float:
    """
    Linear decay scoring function:
    - age_days <= 0 → score 1.0
    - age_days >= window → score 0.0
    """
    if age_days <= 0:
        return 1.0
    return max(0.0, 1.0 - (age_days / window))


#Optional helpers for serialization 

def sets_to_lists(d: dict) -> dict:
    """Convert set values in a dict to sorted lists for JSON serialization."""
    return {k: (sorted(v) if isinstance(v, set) else v) for k, v in d.items()}


def lists_to_sets(d: dict) -> dict:
    """Convert list values in a dict back to sets."""
    return {k: (set(v) if isinstance(v, list) else v) for k, v in d.items()}


__all__ = [
    "EdgeType", "Defaults",
    "normalize_id", "ensure_agent_prefix", "ensure_task_id",
    "decay_linear", "sets_to_lists", "lists_to_sets"
]
