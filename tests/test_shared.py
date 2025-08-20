from memory.memory_module import SharedMemory

def test_write_integration_and_retrieve():
    sm = SharedMemory()
    sm.write("alice", "Started task 12", "Kickoff")
    results = sm.retrieve("alice", "task 12")
    # should pull from both episodic + KG
    texts = " ".join(r["content"] for r in results)
    assert "started task 12" in texts
    assert "kickoff" in texts or "task 12" in texts

def test_get_recent_passthrough():
    sm = SharedMemory()
    sm.write("bob", "first", "ctx")
    sm.write("bob", "second", "ctx")
    recents = sm.get_recent("bob", n=1)
    assert recents[0]["content"] == "second"
