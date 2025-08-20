import pytest
from memory.memory_module import EpisodicMemory
from memory.memory_types import Defaults

def test_store_and_recent_order():
    ep = EpisodicMemory(max_length=3)
    ep.store("agent:alice", "a")
    ep.store("agent:alice", "b")
    ep.store("agent:alice", "c")
    ep.store("agent:alice", "d")
    recents = ep.get_recent("agent:alice", 3)
    assert [r["content"] for r in recents] == ["b", "c", "d"]

def test_strict_mode_keyerror():
    ep = EpisodicMemory(strict_mode=True)
    with pytest.raises(KeyError):
        ep.get_recent("agent:bob")

def test_retrieve_case_insensitive():
    ep = EpisodicMemory()
    ep.store("agent:alice", "Hammer broke")
    hits = ep.retrieve("agent:alice", "hammer")
    assert any("Hammer broke" in h["content"] for h in hits)
