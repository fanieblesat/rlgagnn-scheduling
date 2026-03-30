"""Fast experiment runner — skips RL-only baseline to save ~70% runtime."""
import os, sys, json, time, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.simulation import SchedulingInstance
from experiments.run_experiments import (
    run_baselines, run_ga_std, run_hybrid, _build_comparison_summary,
    _print_comparison_table
)
import numpy as np

def run_fast(instance_path, output_dir, failure_rate, failure_model='exponential',
             rl_episodes=2000, verbose=True):
    instance = SchedulingInstance.load(instance_path)
    if failure_rate is not None:
        instance.failure_rates = np.full(instance.n_machines, failure_rate)
        instance.weibull_scales = 1.0 / instance.failure_rates

    name = os.path.splitext(os.path.basename(instance_path))[0]
    result_path = os.path.join(output_dir, f"{name}_results.json")

    # Skip if already done
    if os.path.exists(result_path):
        print(f"SKIP (already done): {name}")
        return

    print(f"\n{'='*60}")
    print(f"EXPERIMENT: {name} (rate={failure_rate})")
    print(f"{'='*60}")

    results = {
        'instance': name,
        'config': {
            'n_jobs': instance.n_jobs, 'n_machines': instance.n_machines,
            'failure_rate': float(failure_rate), 'failure_model': failure_model,
        }
    }

    # Baselines (instant)
    results['baselines'] = run_baselines(instance, failure_model=failure_model, verbose=verbose)

    # GA-std (~7 min for medium)
    results['ga_std'] = run_ga_std(instance, failure_model=failure_model, verbose=verbose)

    # Hybrid RL-GA-GNN (~90 min for medium)
    results['hybrid'] = run_hybrid(instance, rl_episodes=rl_episodes,
                                    failure_model=failure_model, verbose=verbose)

    summary = _build_comparison_summary(results)
    results['summary'] = summary
    _print_comparison_table(summary)

    os.makedirs(output_dir, exist_ok=True)
    def clean(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, dict): return {k: clean(v) for k, v in obj.items()}
        if isinstance(obj, list): return [clean(v) for v in obj]
        return obj

    with open(result_path, 'w') as f:
        json.dump(clean(results), f, indent=2, default=str)
    print(f"Saved: {result_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--failure_rate', type=float, required=True)
    parser.add_argument('--rl_episodes', type=int, default=2000)
    args = parser.parse_args()
    run_fast(args.instance, args.output, args.failure_rate, rl_episodes=args.rl_episodes)
