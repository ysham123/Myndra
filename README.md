# Myndra: Multi-Agent Orchestration Framework

**Current Version:** v1.1 (October 2025)  
**Focus:** Orchestrator & LLM Planner Integration

---

## Overview

Myndra is a research-oriented multi-agent orchestration framework designed for adaptive, human-in-the-loop collaboration. It features a dynamic orchestrator that plans, assigns, executes, and summarizes tasks among moldable agents that adapt based on shared memory and feedback. Currently, Myndra is on a path toward becoming a full **Multi-Agent Reinforcement Learning (MARL)** framework within approximately 1.5 months, aiming to integrate learning-based agent policies and environment interaction for autonomous multi-agent coordination.

---

## Key Features (v1.1)

- **Adaptive Orchestration:** Real-time task planning and agent role assignment  
- **Moldable Agents:** Behavior shaped by shared episodic and long-term memory  
- **Hybrid Memory System:** Combines short-term episodic buffers with a semantic knowledge graph  
- **LLM Planner Integration:** Supports `gpt-5-mini` by default and optional `gpt-5` for complex reasoning  
- **Fallback Planner:** Deterministic planner available when LLM is disabled  
- **Agent Registry:** Stable agent naming and modular implementation  
- **Testing Suite:** Ongoing development for memory and orchestrator components  

---

## Architecture

```
├── agents/             # Agent implementations and registry
├── memory/             # Episodic and knowledge graph memory modules
├── orchestrator/       # Orchestrator and planner logic
├── interface/          # User interaction components
└── main.py             # Entry point
```

---

## Roadmap

### v1.2 (Upcoming)

- EvaluatorAgent for automated grading and run-time critique  
- Enhanced memory retrieval with decay and relevance scoring  
- Domain-specific adapters (e.g., radiology image QA, tabular analytics)  

### MARL Development Plan (v1.2 → v2.0)

This ongoing open-source research project aims to transition Myndra into a full Multi-Agent Reinforcement Learning (MARL) framework over the next approximately 1.5 months. The development plan focuses on key milestones including the implementation of an EvaluatorAgent for reward modeling, development of an environment interface for agent interaction, integration of agent policy learning mechanisms, and the establishment of a MARL training loop. These efforts will enable autonomous, learning-driven multi-agent coordination within the Myndra ecosystem.

### v2.0 (Research-Ready)

- Comprehensive evaluation harness with reproducible datasets and metrics  
- Dataset adapters (DICOM, CSV, JSON) and de-identification tools  
- Reliability guardrails including self-checks, re-planning, and timeouts  
- Technical report and academic publications  

---

## Getting Started

```bash
git clone https://github.com/ysham123/Myndra.git
cd Myndra
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

Set environment variables:

```bash
export OPENAI_API_KEY=your_api_key
export MYNDRA_USE_LLM=1         # Enable LLM planner (optional)
export MYNDRA_PLANNER_MODEL=gpt-5-mini  # Default planner model
```

Run the application:

```bash
python3 main.py
```

Run tests:

```bash
pytest -v
```

---

## Authorship & Research Ownership

Myndra is an open-source research framework developed and architected by **Yosef Shammout** (Computer Science, Wayne State University). Yosef leads the design of the orchestration logic and agent-memory architecture. While community contributions are welcomed under the MIT license, all core conceptual and architectural work should be credited to Yosef in academic and technical references.

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

For detailed documentation, examples, and troubleshooting, please refer to the project repository.