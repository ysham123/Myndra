# Myndra v2: Planner-Aware Multi-Agent Reinforcement Learning

**Current Version:** v2.0 (November 2025)  
**Focus:** Scalable MARL with Lightweight Planner Context Integration

---

## Overview

Myndra v2 is a research framework for **planner-aware multi-agent reinforcement learning (MARL)**. It demonstrates how lightweight goal decomposition from a deterministic planner can be injected into agent observations to improve coordination—without requiring heavy LLM inference in the rollout loop. The system supports multi-actor rollouts, automatic mixed precision (AMP), torch.compile(), and comprehensive profiling for reproducible MARL research.

---

## Key Features (v2.0)

- **Planner Context Injection:** Lightweight goal vectors appended to agent observations
- **Multi-Actor Rollouts:** Parallel environment collection with `--actors {1,2,4,8}`
- **PPO Implementation:** Stable policy gradient training with clipped surrogate objective
- **AMP Support:** Automatic mixed precision with GradScaler for throughput
- **torch.compile():** Model compilation toggle for optimized inference
- **System Metrics:** GPU utilization tracking and time-to-target-return logging
- **Reproducible Experiments:** Deterministic seeding across `torch`, `numpy`, `random`
- **Visualization Suite:** Learning curves, AUC analysis, scaling plots, ablation comparisons

---

## Repository Structure

```
Myndra/
├── marl/
│   ├── env_wrapper.py       # PettingZoo environment wrapper with planner context
│   └── train_ppo.py         # PPO agent + training loop
├── scripts/
│   ├── run_marl.py          # Multi-seed orchestration script
│   ├── plot_curves.py       # Learning curve + AUC plotting
│   ├── plot_scaling.py      # Actor scaling analysis
│   └── plot_ablations.py    # Method comparison plots
├── systems/
│   └── profiler.py          # Lightweight profiling (timers, GPU util)
├── results/marl/            # Experiment outputs (CSV, JSON, PNG)
└── requirements.txt         # Python dependencies
```

---

## Quickstart

### Installation

```bash
git clone https://github.com/ysham123/Myndra.git
cd Myndra
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

### Run Baseline IPPO (5 seeds)

```bash
./venv/bin/python3 scripts/run_marl.py \
  --env simple_spread_v3 \
  --method ippo \
  --seeds 5 \
  --steps 5000 \
  --actors 4
```

### Run Planner-Aware Myndra-MAPPO

```bash
./venv/bin/python3 scripts/run_marl.py \
  --env simple_spread_v3 \
  --method myndra_mappo \
  --seeds 5 \
  --steps 5000 \
  --actors 4 \
  --interval 32 \
  --context-dim 4
```

### Run with AMP or Compile

```bash
# AMP (automatic mixed precision)
./venv/bin/python3 scripts/run_marl.py \
  --env simple_spread_v3 \
  --method ippo \
  --seeds 3 \
  --steps 3000 \
  --actors 4 \
  --amp on

# torch.compile()
./venv/bin/python3 scripts/run_marl.py \
  --env simple_spread_v3 \
  --method ippo \
  --seeds 3 \
  --steps 3000 \
  --actors 4 \
  --compile on
```

### Generate Plots

```bash
# Learning curves + AUC
./venv/bin/python3 scripts/plot_curves.py \
  --env simple_spread_v3 \
  --method ippo

# Scaling analysis
./venv/bin/python3 scripts/plot_scaling.py \
  --env simple_spread_v3 \
  --method ippo \
  --actors 1 2 4 8

# Ablation comparison
./venv/bin/python3 scripts/plot_ablations.py \
  --env simple_spread_v3 \
  --methods ippo myndra_mappo
```

---

## Results

**Environment:** `simple_spread_v3` (PettingZoo MPE)  
**Training:** 5000 steps, 5 seeds, log_interval=1000

| Method           | Actors | Steps/s  | AUC (mean ± 95% CI)      | Planner Latency P95 (ms) |
|------------------|--------|----------|--------------------------|---------------------------|
| IPPO             | 1      | ~800     | 4325 [4198, 4453]        | –                         |
| IPPO             | 2      | ~1033    | –                        | –                         |
| IPPO             | 4      | ~1071    | –                        | –                         |
| IPPO             | 8      | ~1595    | –                        | –                         |
| Myndra-MAPPO     | 4      | ~1510    | –                        | 0.004                     |
| IPPO + AMP       | 2      | ~772     | –                        | –                         |

**Key Observations:**
- Sub-linear scaling with actors (expected on CPU)
- Planner overhead negligible (~0.004ms P95)
- AMP stable, no NaNs observed
- Reward curves consistent across methods (-1.0 to -1.2)

---

## Toggles & Ablations

| Flag               | Values           | Description                                      |
|--------------------|------------------|--------------------------------------------------|
| `--method`         | `ippo`, `myndra_mappo` | Baseline vs planner-aware MARL             |
| `--actors`         | `1,2,4,8`        | Number of parallel actor environments            |
| `--interval`       | int (default: 32)| Planner context update frequency (steps)         |
| `--context-dim`    | int (default: 4) | Dimensionality of planner context vector         |
| `--planner-cache`  | `on`, `off`      | Cache planner context between intervals          |
| `--amp`            | `on`, `off`      | Automatic mixed precision (AMP)                  |
| `--compile`        | `on`, `off`      | torch.compile() for actor/critic networks        |
| `--target-return`  | float or None    | Track time to reach target mean reward           |

---

## Authorship & Research Ownership

Myndra is an open-source research framework developed and architected by **Yosef Shammout** (Computer Science, Wayne State University). Yosef leads the design of the orchestration logic and agent-memory architecture. While community contributions are welcomed under the MIT license, all core conceptual and architectural work should be credited to Yosef in academic and technical references.

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

For detailed documentation, examples, and troubleshooting, please refer to the project repository.