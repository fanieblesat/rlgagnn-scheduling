# Hybrid RL–GA–GNN Framework for Parallel Machine Scheduling under Machine Failures

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the source code, benchmark instances, and experimental results accompanying the paper:

> **Niebles-Atencio, F.** (2026). A Hybrid Reinforcement Learning–Genetic Algorithm–Graph Neural Network Framework for Stochastic Parallel Machine Scheduling under Machine Failures. *Computers & Operations Research*, Special Issue: Stochastic Simulation and AI.

## Overview

We propose a hybrid optimization framework integrating three complementary AI techniques for scheduling parallel machines under stochastic failures:

- **Graph Neural Networks (GNN):** Encode the scheduling state as a bipartite job–machine graph, producing compact embeddings that capture structural relationships and dynamic failure effects.
- **Reinforcement Learning (RL):** A DQN agent trained within a stochastic simulation environment learns allocation policies robust to machine failures.
- **Genetic Algorithm (GA):** Evolves schedules seeded by RL solutions, with fitness evaluated through Monte Carlo simulation using Common Random Numbers (CRN).

The stochastic simulation environment serves three roles: (1) RL training environment, (2) GA fitness evaluator via Monte Carlo, and (3) final robustness assessment tool.

## Key Results

- GA-based optimization reduces expected makespan by **9–15%** over classical dispatching rules (LPT, SPT, EDD, ATC).
- The hybrid RL–GA–GNN provides an additional **0.5–0.7%** improvement over standalone GA on medium instances (50 jobs), with slightly larger gains in CVaR (tail-risk robustness).
- The hybrid's advantage is **scale-dependent**: negligible on small instances (10 jobs) where random GA initialization suffices, but consistent on larger instances.

## Repository Structure

```
├── src/
│   ├── __init__.py              # Package exports
│   ├── simulation.py            # Discrete-event stochastic simulation environment
│   ├── gnn_model.py             # Bipartite GraphSAGE encoder (plain PyTorch)
│   ├── rl_agent.py              # DQN agent with GNN state embeddings
│   ├── ga_optimizer.py          # GA with OX crossover, RL seeding, MC fitness
│   ├── hybrid_framework.py      # Three-phase pipeline + fidelity analysis
│   └── baselines.py             # Dispatching rules + statistical tests
├── instances/
│   ├── generator.py             # Instance generator
│   ├── small/                   # 30 small instances (n=10, m∈{2,3,5})
│   ├── medium/                  # 30 medium instances (n=50, m∈{5,8,10})
│   └── metadata.json            # Instance suite metadata
├── experiments/
│   ├── run_experiments.py       # Full experiment runner (all methods)
│   └── run_fast.py              # Fast runner (skips RL-only baseline)
├── results/                     # Experimental results (JSON)
│   ├── lambda_0.01/             # Low failure rate results
│   ├── lambda_0.05/             # Medium failure rate results
│   └── lambda_0.10/             # High failure rate results
├── requirements.txt
├── LICENSE
└── README.md
```

## Installation

```bash
git clone https://github.com/fnieblesa/rlgagnn-scheduling.git
cd rlgagnn-scheduling
pip install -r requirements.txt
```

### Dependencies

- Python ≥ 3.10
- NumPy ≥ 1.24
- SciPy ≥ 1.10
- PyTorch ≥ 2.0 (CPU sufficient; GPU optional)

## Quick Start

### 1. Generate Benchmark Instances

```bash
python instances/generator.py --output instances
```

Generates 90 instances: 30 small (n=10) and 30 medium (n=50) across three machine configurations and three due-date tightness levels (τ ∈ {0.5, 1.0, 1.5}).

### 2. Run a Single Experiment

```bash
python experiments/run_experiments.py \
    --instance instances/small/small_10j_2m_tau1.0_s0.json \
    --output results \
    --failure_rate 0.05 \
    --rl_episodes 2000
```

### 3. Run Fast Experiments (GA + Baselines, no RL-only)

```bash
python experiments/run_fast.py \
    --instance instances/medium/medium_50j_5m_tau1.0_s0.json \
    --output results/lambda_0.05 \
    --failure_rate 0.05
```

### 4. Batch Experiments

```bash
# All small instances, all failure rates
for f in instances/small/*.json; do
  for rate in 0.01 0.05 0.10; do
    python experiments/run_fast.py \
      --instance "$f" --failure_rate $rate \
      --output results/lambda_${rate}
  done
done
```

## Using the Framework Programmatically

```python
from src.simulation import SchedulingInstance
from src.hybrid_framework import HybridFramework

# Load instance
instance = SchedulingInstance.load('instances/medium/medium_50j_5m_tau1.0_s0.json')

# Run hybrid framework
framework = HybridFramework(
    instance,
    rl_episodes=2000,
    ga_pop_size=100,
    ga_generations=200,
    ga_n_sim=50,           # MC replications per fitness evaluation
    ga_alpha=0.6,          # Makespan vs. resilience trade-off
    failure_model='exponential',
)
results = framework.run(verbose=True)

print(f"Best makespan: {results['final_metrics']['mean_makespan']:.2f}")
print(f"CVaR_95:       {results['final_metrics']['cvar_95']:.2f}")
```

## Configuration

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `rl_episodes` | 2000 | RL training episodes |
| `ga_pop_size` | 100 | GA population size |
| `ga_generations` | 200 | GA evolution generations |
| `ga_n_sim` | 50 | MC replications per fitness evaluation |
| `ga_alpha` | 0.6 | Trade-off: makespan (1.0) vs. resilience (0.0) |
| `failure_model` | exponential | Failure distribution (`exponential` or `weibull`) |
| `n_gnn_layers` | 3 | GNN message-passing layers |
| `d_hidden` | 64 | GNN hidden dimension |
| `d_embed` | 128 | GNN output embedding dimension |

### Failure Rate Regimes

| Regime | λ | MTBF |
|--------|---|------|
| Low | 0.01 | 100 time units |
| Medium | 0.05 | 20 time units |
| High | 0.10 | 10 time units |

## Simulation Environment

The discrete-event stochastic simulation serves three roles:

1. **RL Training Environment:** Generates episodes with stochastic failure/repair events for policy learning via DQN.
2. **GA Fitness Evaluator:** Monte Carlo evaluation of candidate schedules over multiple failure scenarios, using Common Random Numbers (CRN) for fair comparison.
3. **Robustness Assessment:** Large-sample (N=1,000) evaluation of the final solution with CVaR estimation.

## Reproducing Paper Results

To reproduce the main experimental results reported in the paper:

```bash
# Generate instances
python instances/generator.py --output instances

# Run small instances (≈10 hours)
for f in instances/small/*.json; do
  for rate in 0.01 0.05 0.10; do
    python experiments/run_experiments.py \
      --instance "$f" --failure_rate $rate \
      --output results/lambda_${rate} --rl_episodes 2000
  done
done

# Run medium instances with hybrid (≈90 min each)
for f in instances/medium/medium_50j_5m_*.json; do
  for rate in 0.01 0.05 0.10; do
    python experiments/run_fast.py \
      --instance "$f" --failure_rate $rate \
      --output results/lambda_${rate} --rl_episodes 2000
  done
done

# Run medium instances GA-only for 8m and 10m (≈15 min each)
for f in instances/medium/medium_50j_8m_*.json instances/medium/medium_50j_10m_*.json; do
  for rate in 0.01 0.05 0.10; do
    python experiments/run_fast.py \
      --instance "$f" --failure_rate $rate \
      --output results/lambda_${rate} --rl_episodes 2000
  done
done
```

## Citation

If you use this code or benchmark instances in your research, please cite:

```bibtex
@article{niebles2026hybrid,
  title={A Hybrid Reinforcement Learning--Genetic Algorithm--Graph Neural Network 
         Framework for Stochastic Parallel Machine Scheduling under Machine Failures},
  author={Niebles-Atencio, Fabricio},
  journal={Computers \& Operations Research},
  year={2026},
  note={Special Issue: Stochastic Simulation and AI}
}
```

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## Contact

Fabricio Niebles-Atencio  
Institute of Information Systems, University of Hamburg  
fabricio.niebles.atencio@uni-hamburg.de
