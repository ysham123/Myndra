# Memory Module Documentation

> **Current Status:** In Development (August 2025)  
> **Author:** Yosef Shammout  
> **Focus:** Testing & Implementation

## Overview

The memory module is the cornerstone of Myndra's adaptive orchestration system, enabling agents to share knowledge, maintain context, and evolve their behavior based on past interactions.

## Architecture

![Memory Architecture](https://via.placeholder.com/800x400?text=Memory+Architecture+Diagram)

### Components

1. **EpisodicMemory (Short-term)**
   - Per-agent rolling buffer of recent experiences
   - Maintains insertion order with configurable max length
   - Supports agent-specific retrieval and cross-agent search

2. **KnowledgeGraph (Long-term)**
   - Content-centric directed knowledge graph
   - Node IDs are normalized content strings
   - Supports case-insensitive substring search and BFS traversal

3. **SharedMemory (Unified Interface)**
   - Combines both memory types into a single API
   - Handles writing to both memory systems
   - Provides unified retrieval with relevance scoring

## Current Development Focus

### Testing Suite (In Progress)

We are currently developing comprehensive tests for:

- **EpisodicMemory**
  - [x] Basic storage and retrieval
  - [x] Agent-specific memory isolation
  - [ ] Memory eviction policies
  - [ ] Concurrency safety

- **KnowledgeGraph**
  - [x] Node creation and edge insertion
  - [x] Basic traversal operations
  - [ ] Complex graph queries
  - [ ] Serialization/deserialization

- **SharedMemory**
  - [x] Unified write operations
  - [ ] Relevance scoring algorithms
  - [ ] Memory decay functions
  - [ ] Performance benchmarking

## Implementation Timeline

| Week | Focus Area | Status |
|------|-----------|--------|
| Week 1 | Core API design | ✅ Completed |
| Week 2 | EpisodicMemory implementation | ✅ Completed |
| Week 3 | KnowledgeGraph implementation | ✅ Completed |
| Week 4 | SharedMemory integration | ✅ Completed |
| **Week 5** | **Basic test coverage** | **🔄 In Progress** |
| Week 6 | Advanced testing | 📅 Planned |
| Week 7 | Performance optimization | 📅 Planned |
| Week 8 | Documentation & examples | 📅 Planned |

## Usage Examples

```python
# Initialize the memory system
memory = SharedMemory()

# Store agent observations
memory.write("agent_1", "Found a solution to problem X", context="Solving problem X")
memory.write("agent_2", "Verified solution to problem X", context="Found a solution to problem X")

# Retrieve recent memories
recent = memory.get_recent("agent_1", n=5)

# Search across all memories
results = memory.retrieve("agent_1", "solution")

# Find related knowledge
related = memory.long_term.get_related("Found a solution to problem X")
```

## Next Steps

1. Complete the testing suite in `memory_tests.py`
2. Implement type definitions in `memory_types.py`
3. Add performance benchmarks
4. Create integration examples with the orchestrator

## Contributing

When contributing to the memory module, please ensure:

1. All tests pass in `memory_tests.py`
2. Documentation is updated to reflect changes
3. Performance considerations are documented
4. Type hints are properly maintained
