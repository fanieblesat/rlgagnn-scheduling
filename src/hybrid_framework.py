"""
Hybrid RL-GA-GNN Framework Integration.

Orchestrates the three-phase pipeline:
  Phase 1: Train RL agent with GNN embeddings in simulation
  Phase 2: Generate elite seeds from trained RL policy
  Phase 3: Run GA optimization with RL-seeded population and MC fitness
  Phase 4: Final robustness evaluation
"""

import numpy as np
import time
import json
from typing import List, Dict, Optional

from .simulation import SimulationEnvironment, SchedulingInstance
from .rl_agent import DQNAgent
from .ga_optimizer import GAOptimizer
from .baselines import evaluate_schedule


class HybridFramework:
    """
    Main entry point for the hybrid RL-GA-GNN scheduling framework.
    """

    def __init__(self, instance: SchedulingInstance,
                 # GNN params
                 d_hidden: int = 64, d_embed: int = 128, n_gnn_layers: int = 3,
                 # RL params
                 rl_episodes: int = 2000, rl_lr: float = 1e-3,
                 # GA params
                 ga_pop_size: int = 100, ga_generations: int = 200,
                 ga_n_sim: int = 50, ga_alpha: float = 0.6,
                 # General
                 failure_model: str = 'exponential',
                 n_eval_replications: int = 1000,
                 device: str = 'cpu',
                 seed: int = 42):
        """
        Args:
            instance: Scheduling problem instance
            d_hidden: GNN hidden dimension
            d_embed: GNN output embedding dimension
            n_gnn_layers: Number of GNN message-passing layers
            rl_episodes: Number of RL training episodes
            rl_lr: RL learning rate
            ga_pop_size: GA population size
            ga_generations: Number of GA generations
            ga_n_sim: MC replications per GA fitness evaluation
            ga_alpha: Trade-off parameter (makespan vs resilience)
            failure_model: 'exponential' or 'weibull'
            n_eval_replications: Replications for final evaluation
            device: 'cpu' or 'cuda'
            seed: Random seed
        """
        self.instance = instance
        self.failure_model = failure_model
        self.n_eval_replications = n_eval_replications
        self.seed = seed

        self.rl_episodes = rl_episodes

        # Initialize RL agent
        self.rl_agent = DQNAgent(
            instance, d_hidden=d_hidden, d_embed=d_embed,
            n_gnn_layers=n_gnn_layers, lr=rl_lr, device=device
        )

        # GA config
        self.ga_pop_size = ga_pop_size
        self.ga_generations = ga_generations
        self.ga_n_sim = ga_n_sim
        self.ga_alpha = ga_alpha

        # Results storage
        self.results = {}

    def run(self, verbose: bool = True) -> Dict:
        """
        Run the complete hybrid framework pipeline.

        Returns:
            Dictionary with results from all phases.
        """
        total_start = time.time()

        if verbose:
            print("=" * 60)
            print("HYBRID RL-GA-GNN SCHEDULING FRAMEWORK")
            print(f"Instance: {self.instance.n_jobs} jobs, "
                  f"{self.instance.n_machines} machines")
            print(f"Failure model: {self.failure_model}")
            print("=" * 60)

        # ============================================================
        # Phase 1: Train RL agent with GNN embeddings
        # ============================================================
        if verbose:
            print("\n[Phase 1] Training RL agent with GNN embeddings...")

        phase1_start = time.time()
        rl_rewards = self.rl_agent.train_agent(
            n_episodes=self.rl_episodes,
            failure_model=self.failure_model,
            verbose=verbose
        )
        phase1_time = time.time() - phase1_start

        if verbose:
            print(f"  RL training complete in {phase1_time:.1f}s")

        # ============================================================
        # Phase 2: Generate RL elite seeds + heuristic seeds
        # ============================================================
        if verbose:
            print("\n[Phase 2] Generating elite seeds (RL + heuristics)...")

        phase2_start = time.time()
        
        # Generate RL seeds with varying exploration levels
        rl_seeds_raw = []
        epsilons = [0.0, 0.05, 0.10, 0.15, 0.20]
        for eps in epsilons:
            for trial in range(5):
                schedule = self.rl_agent.generate_schedule(
                    failure_model=self.failure_model,
                    epsilon=eps,
                    rng_seed=trial * 100 + int(eps * 100)
                )
                rl_seeds_raw.append(schedule)

        # Remove duplicates
        unique_seeds = []
        seen = set()
        for s in rl_seeds_raw:
            key = tuple(s)
            if key not in seen:
                seen.add(key)
                unique_seeds.append(s)

        # Also add strong heuristic seeds — these give the GA a known-good
        # starting point and ensure we never do worse than the best heuristic
        from .baselines import lpt_schedule, spt_schedule, edd_schedule, atc_schedule
        for heuristic_fn in [lpt_schedule, spt_schedule, edd_schedule, atc_schedule]:
            h_sched = heuristic_fn(self.instance)
            key = tuple(h_sched)
            if key not in seen:
                seen.add(key)
                unique_seeds.append(h_sched)

        # Evaluate all seeds via quick MC to rank them
        if verbose:
            print(f"  Evaluating {len(unique_seeds)} candidate seeds...")
        
        seed_scores = []
        env = SimulationEnvironment(self.instance, failure_model=self.failure_model)
        for i, sched in enumerate(unique_seeds):
            mc = env.monte_carlo_evaluate(sched, n_replications=20, base_seed=7777)
            seed_scores.append((mc['mean_makespan'], i, sched))
        
        # Sort by makespan (ascending = best first) and keep top seeds
        seed_scores.sort(key=lambda x: x[0])
        max_seeds = max(5, int(self.ga_pop_size * 0.25))
        rl_seeds = [s[2] for s in seed_scores[:max_seeds]]

        phase2_time = time.time() - phase2_start

        if verbose:
            best_seed_ms = seed_scores[0][0]
            worst_kept_ms = seed_scores[min(max_seeds-1, len(seed_scores)-1)][0]
            print(f"  Kept {len(rl_seeds)} best seeds (makespan range: "
                  f"{best_seed_ms:.1f} - {worst_kept_ms:.1f}) in {phase2_time:.1f}s")

        # ============================================================
        # Phase 3: GA optimization with MC fitness evaluation
        # ============================================================
        if verbose:
            print("\n[Phase 3] Running GA optimization with MC fitness...")

        phase3_start = time.time()
        ga = GAOptimizer(
            self.instance,
            pop_size=self.ga_pop_size,
            n_generations=self.ga_generations,
            n_sim=self.ga_n_sim,
            alpha=self.ga_alpha,
            failure_model=self.failure_model,
            rng_seed=self.seed
        )
        best_individual, ga_history = ga.evolve(
            rl_seeds=rl_seeds, verbose=verbose
        )
        phase3_time = time.time() - phase3_start

        # ============================================================
        # Phase 4: Final robustness evaluation
        # ============================================================
        if verbose:
            print(f"\n[Phase 4] Final evaluation ({self.n_eval_replications} replications)...")

        phase4_start = time.time()
        final_metrics = evaluate_schedule(
            self.instance,
            best_individual.chromosome,
            n_replications=self.n_eval_replications,
            failure_model=self.failure_model,
            base_seed=12345
        )
        phase4_time = time.time() - phase4_start

        total_time = time.time() - total_start

        # ============================================================
        # Compile results
        # ============================================================
        self.results = {
            'best_schedule': best_individual.chromosome,
            'final_metrics': final_metrics,
            'ga_history': {
                'best_fitness': ga_history['best_fitness'],
                'avg_fitness': ga_history['avg_fitness'],
            },
            'timing': {
                'phase1_rl_training': phase1_time,
                'phase2_seed_generation': phase2_time,
                'phase3_ga_optimization': phase3_time,
                'phase4_final_evaluation': phase4_time,
                'total': total_time,
            },
            'config': {
                'n_jobs': self.instance.n_jobs,
                'n_machines': self.instance.n_machines,
                'failure_model': self.failure_model,
                'ga_alpha': self.ga_alpha,
                'ga_n_sim': self.ga_n_sim,
                'ga_pop_size': self.ga_pop_size,
                'ga_generations': self.ga_generations,
                'n_rl_seeds': len(rl_seeds),
            }
        }

        if verbose:
            print("\n" + "=" * 60)
            print("RESULTS SUMMARY")
            print("=" * 60)
            print(f"  Best schedule makespan:  {final_metrics['mean_makespan']:.2f} "
                  f"(± {final_metrics['std_makespan']:.2f})")
            print(f"  CVaR_95:                 {final_metrics['cvar_95']:.2f}")
            print(f"  CV(makespan):            {final_metrics['cv_makespan']:.4f}")
            print(f"  On-time delivery:        {final_metrics['mean_on_time_pct']:.1f}%")
            print(f"  Total computation time:  {total_time:.1f}s")
            print(f"    - RL training:         {phase1_time:.1f}s")
            print(f"    - Seed generation:     {phase2_time:.1f}s")
            print(f"    - GA optimization:     {phase3_time:.1f}s")
            print(f"    - Final evaluation:    {phase4_time:.1f}s")

        return self.results

    def save_results(self, filepath: str):
        """Save results to JSON."""
        # Convert numpy types for JSON serialization
        def convert(obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        serializable = json.loads(json.dumps(self.results, default=convert))
        with open(filepath, 'w') as f:
            json.dump(serializable, f, indent=2)


def run_simulation_fidelity_analysis(instance: SchedulingInstance,
                                      job_sequence: List[int],
                                      n_sim_values: List[int] = None,
                                      n_replications: int = 30,
                                      failure_model: str = 'exponential',
                                      verbose: bool = True) -> Dict:
    """
    Analyze the impact of simulation fidelity (N_sim) on solution
    quality and computational cost.

    This implements the analysis described in Section 4.5.3 of the paper:
    the trade-off between Monte Carlo replications per fitness evaluation
    and optimization performance.

    Args:
        instance: Problem instance
        job_sequence: A reference schedule to evaluate
        n_sim_values: List of N_sim values to test
        n_replications: Independent replications per N_sim
        failure_model: Failure distribution
        verbose: Print progress

    Returns:
        Dict with fidelity analysis results
    """
    if n_sim_values is None:
        n_sim_values = [5, 10, 20, 50, 100, 200]

    results = {}

    # Ground truth: high-fidelity evaluation
    if verbose:
        print("Computing ground truth (N_sim=1000)...")
    env = SimulationEnvironment(instance, failure_model=failure_model)
    ground_truth = env.monte_carlo_evaluate(job_sequence, n_replications=1000, base_seed=99999)
    true_mean = ground_truth['mean_makespan']
    true_cvar = ground_truth['cvar_95']

    for n_sim in n_sim_values:
        if verbose:
            print(f"  Testing N_sim = {n_sim}...")

        estimates_mean = []
        estimates_cvar = []
        times = []

        for rep in range(n_replications):
            start = time.time()
            env = SimulationEnvironment(instance, failure_model=failure_model)
            mc = env.monte_carlo_evaluate(
                job_sequence, n_replications=n_sim,
                base_seed=rep * 10000
            )
            elapsed = time.time() - start

            estimates_mean.append(mc['mean_makespan'])
            estimates_cvar.append(mc['cvar_95'])
            times.append(elapsed)

        mean_est = np.array(estimates_mean)
        cvar_est = np.array(estimates_cvar)

        results[n_sim] = {
            'mean_estimate': float(np.mean(mean_est)),
            'std_estimate': float(np.std(mean_est)),
            'bias_mean': float(np.mean(mean_est) - true_mean),
            'rmse_mean': float(np.sqrt(np.mean((mean_est - true_mean) ** 2))),
            'mean_cvar_estimate': float(np.mean(cvar_est)),
            'std_cvar_estimate': float(np.std(cvar_est)),
            'bias_cvar': float(np.mean(cvar_est) - true_cvar),
            'rmse_cvar': float(np.sqrt(np.mean((cvar_est - true_cvar) ** 2))),
            'avg_time_seconds': float(np.mean(times)),
            'total_time_seconds': float(np.sum(times)),
        }

    results['ground_truth'] = {
        'mean_makespan': true_mean,
        'cvar_95': true_cvar,
    }

    return results
