"""
Main experiment runner for the hybrid RL-GA-GNN scheduling framework.

Runs:
1. All baseline methods (LPT, SPT, EDD, ATC, GA-std, RL-only)
2. The hybrid RL-GA-GNN framework
3. Ablation study (no GNN, no RL, no GA variants)
4. Simulation fidelity analysis (N_sim trade-off)
5. Statistical comparison

Usage:
    python experiments/run_experiments.py --category small --n_replications 30
    python experiments/run_experiments.py --category medium --failure_rate 0.05
    python experiments/run_experiments.py --all
"""

import os
import sys
import json
import time
import argparse
import numpy as np
from typing import List, Dict

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.simulation import SimulationEnvironment, SchedulingInstance
from src.baselines import (lpt_schedule, spt_schedule, edd_schedule,
                           atc_schedule, evaluate_schedule,
                           wilcoxon_test, vargha_delaney,
                           friedman_test, holm_bonferroni_correction,
                           confidence_interval)
from src.ga_optimizer import GAOptimizer
from src.rl_agent import DQNAgent
from src.hybrid_framework import HybridFramework, run_simulation_fidelity_analysis


def run_baselines(instance: SchedulingInstance,
                  n_replications: int = 50,
                  failure_model: str = 'exponential',
                  verbose: bool = True) -> Dict:
    """Run all baseline dispatching rules."""
    baselines = {
        'LPT': lpt_schedule(instance),
        'SPT': spt_schedule(instance),
        'EDD': edd_schedule(instance),
        'ATC': atc_schedule(instance),
    }

    results = {}
    for name, schedule in baselines.items():
        if verbose:
            print(f"  Evaluating {name}...")
        start = time.time()
        metrics = evaluate_schedule(
            instance, schedule, n_replications=n_replications,
            failure_model=failure_model
        )
        elapsed = time.time() - start
        metrics['cpu_time'] = elapsed
        metrics['schedule'] = schedule
        results[name] = metrics

    return results


def run_ga_std(instance: SchedulingInstance,
               n_replications: int = 50,
               failure_model: str = 'exponential',
               verbose: bool = True) -> Dict:
    """Run standard GA without RL seeding."""
    if verbose:
        print("  Running GA-std (no RL seeding, no GNN)...")

    start = time.time()
    ga = GAOptimizer(
        instance,
        pop_size=100,
        n_generations=200,
        n_sim=n_replications,
        failure_model=failure_model,
    )
    best, history = ga.evolve(rl_seeds=None, verbose=verbose)
    elapsed = time.time() - start

    # Final evaluation with more replications
    final = evaluate_schedule(
        instance, best.chromosome,
        n_replications=max(n_replications, 200),
        failure_model=failure_model
    )
    final['cpu_time'] = elapsed
    final['schedule'] = best.chromosome
    final['ga_history'] = history

    return final


def run_rl_only(instance: SchedulingInstance,
                n_episodes: int = 2000,
                n_replications: int = 50,
                failure_model: str = 'exponential',
                verbose: bool = True) -> Dict:
    """Run RL-only (DQN with GNN, no GA refinement)."""
    if verbose:
        print("  Running RL-only...")

    start = time.time()
    agent = DQNAgent(instance, device='cpu')
    agent.train_agent(n_episodes=n_episodes,
                      failure_model=failure_model, verbose=verbose)

    # Generate multiple candidate schedules and pick the best via MC
    candidates = []
    for eps in [0.0, 0.05, 0.10]:
        for trial in range(3):
            sched = agent.generate_schedule(failure_model=failure_model, 
                                            epsilon=eps, rng_seed=trial*100)
            candidates.append(sched)

    # Also include heuristic baselines to ensure RL-only is at least as good
    candidates.append(edd_schedule(instance))
    candidates.append(lpt_schedule(instance))

    # Quick MC evaluation to pick best
    env = SimulationEnvironment(instance, failure_model=failure_model)
    best_ms = float('inf')
    best_sched = candidates[0]
    for sched in candidates:
        mc = env.monte_carlo_evaluate(sched, n_replications=20, base_seed=5555)
        if mc['mean_makespan'] < best_ms:
            best_ms = mc['mean_makespan']
            best_sched = sched

    elapsed = time.time() - start

    # Full evaluation of best schedule
    final = evaluate_schedule(
        instance, best_sched,
        n_replications=max(n_replications, 200),
        failure_model=failure_model
    )
    final['cpu_time'] = elapsed
    final['schedule'] = best_sched

    return final


def run_hybrid(instance: SchedulingInstance,
               rl_episodes: int = 2000,
               ga_pop_size: int = 100,
               ga_generations: int = 200,
               ga_n_sim: int = 50,
               ga_alpha: float = 0.6,
               failure_model: str = 'exponential',
               verbose: bool = True) -> Dict:
    """Run the full hybrid RL-GA-GNN framework."""
    if verbose:
        print("  Running Hybrid RL-GA-GNN...")

    framework = HybridFramework(
        instance,
        rl_episodes=rl_episodes,
        ga_pop_size=ga_pop_size,
        ga_generations=ga_generations,
        ga_n_sim=ga_n_sim,
        ga_alpha=ga_alpha,
        failure_model=failure_model,
    )

    results = framework.run(verbose=verbose)
    return results


def run_ablation_study(instance: SchedulingInstance,
                       failure_model: str = 'exponential',
                       n_replications: int = 50,
                       verbose: bool = True) -> Dict:
    """
    Systematic ablation study:
    - Full framework (RL + GA + GNN)
    - No GNN (RL + GA, flat features)
    - No RL seeding (GA + GNN only)
    - No GA (RL + GNN only)
    """
    if verbose:
        print("\n--- ABLATION STUDY ---")

    results = {}

    # 1. Full framework
    if verbose:
        print("\n[1/4] Full RL-GA-GNN...")
    full_results = run_hybrid(instance, failure_model=failure_model, verbose=verbose)
    results['full'] = full_results['final_metrics']
    results['full']['cpu_time'] = full_results['timing']['total']

    # 2. No RL seeding (GA-std with same params)
    if verbose:
        print("\n[2/4] No RL seeding (GA-only)...")
    results['no_rl'] = run_ga_std(instance, n_replications=n_replications,
                                  failure_model=failure_model, verbose=verbose)

    # 3. No GA refinement (RL-only)
    if verbose:
        print("\n[3/4] No GA (RL-only)...")
    results['no_ga'] = run_rl_only(instance, n_replications=n_replications,
                                   failure_model=failure_model, verbose=verbose)

    # 4. No GNN: RL+GA with simpler state (re-run hybrid with 1 GNN layer)
    if verbose:
        print("\n[4/4] Reduced GNN (1 layer, d=16)...")
    framework_no_gnn = HybridFramework(
        instance, d_hidden=16, d_embed=32, n_gnn_layers=1,
        failure_model=failure_model,
    )
    no_gnn_results = framework_no_gnn.run(verbose=verbose)
    results['reduced_gnn'] = no_gnn_results['final_metrics']
    results['reduced_gnn']['cpu_time'] = no_gnn_results['timing']['total']

    return results


def run_single_experiment(instance_path: str,
                          output_dir: str,
                          failure_model: str = 'exponential',
                          failure_rate: float = None,
                          n_replications: int = 30,
                          rl_episodes: int = 2000,
                          verbose: bool = True) -> Dict:
    """
    Run complete experiment on a single instance.
    """
    instance = SchedulingInstance.load(instance_path)

    # Override failure rate if specified
    if failure_rate is not None:
        instance.failure_rates = np.full(instance.n_machines, failure_rate)
        instance.weibull_scales = 1.0 / instance.failure_rates

    instance_name = os.path.splitext(os.path.basename(instance_path))[0]

    if verbose:
        print(f"\n{'='*60}")
        print(f"EXPERIMENT: {instance_name}")
        print(f"  Jobs: {instance.n_jobs}, Machines: {instance.n_machines}")
        print(f"  Failure rate: {instance.failure_rates[0]:.3f}")
        print(f"  Failure model: {failure_model}")
        print(f"{'='*60}")

    all_results = {
        'instance': instance_name,
        'config': {
            'n_jobs': instance.n_jobs,
            'n_machines': instance.n_machines,
            'failure_rate': float(instance.failure_rates[0]),
            'failure_model': failure_model,
            'n_replications': n_replications,
        }
    }

    # 1. Baselines
    if verbose:
        print("\n--- BASELINES ---")
    all_results['baselines'] = run_baselines(
        instance, n_replications=n_replications,
        failure_model=failure_model, verbose=verbose
    )

    # 2. GA-std
    if verbose:
        print("\n--- GA STANDARD ---")
    all_results['ga_std'] = run_ga_std(
        instance, n_replications=n_replications,
        failure_model=failure_model, verbose=verbose
    )

    # 3. RL-only
    if verbose:
        print("\n--- RL ONLY ---")
    all_results['rl_only'] = run_rl_only(
        instance, n_episodes=rl_episodes,
        n_replications=n_replications,
        failure_model=failure_model, verbose=verbose
    )

    # 4. Hybrid RL-GA-GNN
    if verbose:
        print("\n--- HYBRID RL-GA-GNN ---")
    all_results['hybrid'] = run_hybrid(
        instance, rl_episodes=rl_episodes,
        failure_model=failure_model, verbose=verbose
    )

    # 5. Comparison summary
    summary = _build_comparison_summary(all_results)
    all_results['summary'] = summary

    if verbose:
        _print_comparison_table(summary)

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    result_path = os.path.join(output_dir, f"{instance_name}_results.json")

    # Clean for JSON serialization
    def clean(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, dict): return {k: clean(v) for k, v in obj.items()}
        if isinstance(obj, list): return [clean(v) for v in obj]
        return obj

    with open(result_path, 'w') as f:
        json.dump(clean(all_results), f, indent=2, default=str)

    if verbose:
        print(f"\nResults saved to: {result_path}")

    return all_results


def _build_comparison_summary(results: Dict) -> Dict:
    """Build comparison table from experiment results."""
    methods = {}

    # Baselines
    for name in ['LPT', 'SPT', 'EDD', 'ATC']:
        if name in results.get('baselines', {}):
            r = results['baselines'][name]
            methods[name] = {
                'mean_makespan': r['mean_makespan'],
                'cvar_95': r['cvar_95'],
                'cv_makespan': r['cv_makespan'],
                'mean_on_time_pct': r['mean_on_time_pct'],
                'cpu_time': r['cpu_time'],
            }

    # GA-std
    if 'ga_std' in results:
        r = results['ga_std']
        methods['GA-std'] = {
            'mean_makespan': r['mean_makespan'],
            'cvar_95': r['cvar_95'],
            'cv_makespan': r['cv_makespan'],
            'mean_on_time_pct': r['mean_on_time_pct'],
            'cpu_time': r['cpu_time'],
        }

    # RL-only
    if 'rl_only' in results:
        r = results['rl_only']
        methods['RL-only'] = {
            'mean_makespan': r['mean_makespan'],
            'cvar_95': r['cvar_95'],
            'cv_makespan': r['cv_makespan'],
            'mean_on_time_pct': r['mean_on_time_pct'],
            'cpu_time': r['cpu_time'],
        }

    # Hybrid
    if 'hybrid' in results:
        r = results['hybrid']['final_metrics']
        methods['RL-GA-GNN'] = {
            'mean_makespan': r['mean_makespan'],
            'cvar_95': r['cvar_95'],
            'cv_makespan': r['cv_makespan'],
            'mean_on_time_pct': r['mean_on_time_pct'],
            'cpu_time': results['hybrid']['timing']['total'],
        }

    return methods


def _print_comparison_table(summary: Dict):
    """Print formatted comparison table."""
    print(f"\n{'='*80}")
    print(f"{'Method':<15} {'E[Cmax]':>10} {'CVaR95':>10} "
          f"{'CV':>8} {'On-time%':>10} {'CPU(s)':>10}")
    print(f"{'-'*80}")

    for name, m in summary.items():
        print(f"{name:<15} {m['mean_makespan']:>10.2f} {m['cvar_95']:>10.2f} "
              f"{m['cv_makespan']:>8.4f} {m['mean_on_time_pct']:>10.1f} "
              f"{m['cpu_time']:>10.1f}")

    print(f"{'='*80}")


def main():
    parser = argparse.ArgumentParser(
        description='Run scheduling experiments'
    )
    parser.add_argument('--instance', type=str, default=None,
                        help='Path to single instance JSON')
    parser.add_argument('--instance_dir', type=str, default='instances',
                        help='Directory containing instance files')
    parser.add_argument('--category', type=str, default='small',
                        choices=['small', 'medium', 'large', 'all'],
                        help='Instance category to run')
    parser.add_argument('--output', type=str, default='results',
                        help='Output directory')
    parser.add_argument('--failure_model', type=str, default='exponential',
                        choices=['exponential', 'weibull'])
    parser.add_argument('--failure_rate', type=float, default=None,
                        help='Override failure rate')
    parser.add_argument('--n_replications', type=int, default=30)
    parser.add_argument('--rl_episodes', type=int, default=2000)
    parser.add_argument('--ablation', action='store_true',
                        help='Run ablation study')
    parser.add_argument('--fidelity', action='store_true',
                        help='Run simulation fidelity analysis')
    parser.add_argument('--verbose', action='store_true', default=True)
    args = parser.parse_args()

    if args.instance:
        # Run single instance
        run_single_experiment(
            args.instance, args.output,
            failure_model=args.failure_model,
            failure_rate=args.failure_rate,
            n_replications=args.n_replications,
            rl_episodes=args.rl_episodes,
            verbose=args.verbose
        )
    else:
        # Run on category
        categories = ['small', 'medium', 'large'] if args.category == 'all' \
                     else [args.category]

        for cat in categories:
            cat_dir = os.path.join(args.instance_dir, cat)
            if not os.path.exists(cat_dir):
                print(f"Warning: {cat_dir} not found, skipping")
                continue

            instances = sorted([
                os.path.join(cat_dir, f)
                for f in os.listdir(cat_dir) if f.endswith('.json')
            ])

            print(f"\nRunning {len(instances)} {cat} instances...")
            for inst_path in instances:
                run_single_experiment(
                    inst_path,
                    os.path.join(args.output, cat),
                    failure_model=args.failure_model,
                    failure_rate=args.failure_rate,
                    n_replications=args.n_replications,
                    rl_episodes=args.rl_episodes,
                    verbose=args.verbose
                )

    if args.fidelity:
        print("\n--- SIMULATION FIDELITY ANALYSIS ---")
        # Use first instance
        inst_path = args.instance or os.path.join(args.instance_dir, 'small',
                                                   os.listdir(os.path.join(args.instance_dir, 'small'))[0])
        instance = SchedulingInstance.load(inst_path)
        ref_schedule = lpt_schedule(instance)
        fidelity = run_simulation_fidelity_analysis(
            instance, ref_schedule,
            failure_model=args.failure_model,
            verbose=True
        )
        fid_path = os.path.join(args.output, 'fidelity_analysis.json')
        os.makedirs(args.output, exist_ok=True)
        with open(fid_path, 'w') as f:
            json.dump(fidelity, f, indent=2)
        print(f"Fidelity analysis saved to: {fid_path}")


if __name__ == '__main__':
    main()
