from memory.memory_module import KnowledgeGraph

def test_add_node_and_provenance():
    kg = KnowledgeGraph()
    kg.add_node("agent:alice", "Hammer failed", context="Tried hammer")
    assert "hammer failed" in kg.graph
    assert "tried hammer" in kg.graph
    edges = list(kg.graph.edges())
    assert ("tried hammer", "hammer failed") in edges

def test_bfs_depth_limit():
    kg = KnowledgeGraph()
    kg.add_node("a", "root", context="child")
    kg.add_node("a", "child", context="leaf")
    related = kg.get_related("root", depth=1)
    nodes = {r["dst"] for r in related}
    assert "child" in nodes
    assert "leaf" not in nodes
