"""
Hybrid RL-GA-GNN Framework for Parallel Machine Scheduling under Machine Failures.

Modules:
    simulation       - Discrete-event stochastic simulation environment
    gnn_model        - GNN encoder for bipartite job-machine graphs
    rl_agent         - DQN agent with GNN state embeddings
    ga_optimizer     - Genetic algorithm with RL seeding and MC fitness
    hybrid_framework - Integration of all components
    baselines        - Dispatching rules and statistical tests
"""

# Core (no torch dependency)
from .simulation import SimulationEnvironment, SchedulingInstance
from .baselines import lpt_schedule, spt_schedule, edd_schedule, atc_schedule
from .ga_optimizer import GAOptimizer

# ML components (require torch)
try:
    from .gnn_model import GNNEncoder, state_to_graph
    from .rl_agent import DQNAgent
    from .hybrid_framework import HybridFramework
except ImportError:
    pass  # torch not installed; ML components unavailable
