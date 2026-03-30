"""
Instance generator for parallel machine scheduling under machine failures.

Generates instances across four categories:
  - Small:  10-30 jobs, 2-5 machines (30 instances)
  - Medium: 50-100 jobs, 5-10 machines (30 instances)
  - Large:  200-500 jobs, 10-20 machines (30 instances)

Each category includes three due-date tightness levels (tau = 0.5, 1.0, 1.5)
and three failure-rate regimes (low, medium, high).
"""

import os
import json
import numpy as np
from typing import List, Tuple

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.simulation import SchedulingInstance


def generate_instance(n_jobs: int, n_machines: int,
                      tau: float = 1.0,
                      failure_rate: float = 0.05,
                      repair_rate: float = 0.2,
                      weibull_shape: float = 1.0,
                      rng_seed: int = 0) -> SchedulingInstance:
    """
    Generate a single scheduling instance.

    Args:
        n_jobs: Number of jobs
        n_machines: Number of machines
        tau: Due-date tightness factor (lower = tighter)
        failure_rate: Machine failure rate (lambda)
        repair_rate: Machine repair rate (mu)
        weibull_shape: Weibull shape parameter (1.0 = exponential)
        rng_seed: Random seed

    Returns:
        SchedulingInstance
    """
    rng = np.random.RandomState(rng_seed)

    # Processing times: U[1, 100]
    processing_times = rng.uniform(1, 100, size=n_jobs)

    # Release dates: U[0, r_bar] where r_bar ~ total_workload / n_machines
    total_workload = np.sum(processing_times)
    r_bar = total_workload / (n_machines * 2)
    release_dates = rng.uniform(0, r_bar, size=n_jobs)

    # Due dates: d_j = r_j + p_j * (1 + U[0, tau])
    due_dates = release_dates + processing_times * (1 + rng.uniform(0, tau, size=n_jobs))

    # Weights: U[1, 10] (integer)
    weights = rng.randint(1, 11, size=n_jobs).astype(float)

    # Machine parameters (identical machines with slightly varying reliability)
    failure_rates = np.full(n_machines, failure_rate)
    # Add small variation (±20%)
    failure_rates *= (1 + rng.uniform(-0.2, 0.2, size=n_machines))
    failure_rates = np.clip(failure_rates, 0.001, 1.0)

    repair_rates = np.full(n_machines, repair_rate)
    repair_rates *= (1 + rng.uniform(-0.1, 0.1, size=n_machines))

    weibull_shapes = np.full(n_machines, weibull_shape)
    # Scale parameter derived from failure rate: eta = (1/lambda) for beta=1
    weibull_scales = 1.0 / failure_rates

    return SchedulingInstance(
        n_jobs=n_jobs,
        n_machines=n_machines,
        processing_times=processing_times,
        due_dates=due_dates,
        release_dates=release_dates,
        weights=weights,
        failure_rates=failure_rates,
        repair_rates=repair_rates,
        weibull_shapes=weibull_shapes,
        weibull_scales=weibull_scales,
    )


def generate_all_instances(output_dir: str, verbose: bool = True) -> List[str]:
    """
    Generate the complete test suite as specified in the paper.

    Categories:
      Small:  n in {10, 20, 30}, m in {2, 3, 5}     -> 30 instances
      Medium: n in {50, 75, 100}, m in {5, 8, 10}    -> 30 instances
      Large:  n in {200, 350, 500}, m in {10, 15, 20} -> 30 instances

    Each with tau in {0.5, 1.0, 1.5} and 3-4 seeds per config.
    Failure rates set to medium (lambda=0.05) by default;
    experiments vary this separately.
    """
    os.makedirs(output_dir, exist_ok=True)

    configs = {
        'small': {
            'jobs': [10, 20, 30],
            'machines': [2, 3, 5],
            'count_per_config': 4,
        },
        'medium': {
            'jobs': [50, 75, 100],
            'machines': [5, 8, 10],
            'count_per_config': 4,
        },
        'large': {
            'jobs': [200, 350, 500],
            'machines': [10, 15, 20],
            'count_per_config': 4,
        },
    }

    taus = [0.5, 1.0, 1.5]
    failure_rate = 0.05  # Medium rate (default)
    repair_rate = 0.2    # Mean repair time = 5

    all_files = []
    instance_id = 0

    for category, cfg in configs.items():
        cat_dir = os.path.join(output_dir, category)
        os.makedirs(cat_dir, exist_ok=True)

        if verbose:
            print(f"\nGenerating {category} instances...")

        count = 0
        for n in cfg['jobs']:
            for m in cfg['machines']:
                for tau in taus:
                    for seed_offset in range(cfg['count_per_config']):
                        # Skip some combos to stay near 30 per category
                        if count >= 30:
                            break

                        rng_seed = instance_id * 7 + seed_offset * 13 + 1000

                        inst = generate_instance(
                            n_jobs=n, n_machines=m, tau=tau,
                            failure_rate=failure_rate,
                            repair_rate=repair_rate,
                            rng_seed=rng_seed
                        )

                        filename = f"{category}_{n}j_{m}m_tau{tau}_s{seed_offset}.json"
                        filepath = os.path.join(cat_dir, filename)
                        inst.save(filepath)
                        all_files.append(filepath)

                        count += 1
                        instance_id += 1

                        if verbose:
                            print(f"  [{count:2d}] {filename} "
                                  f"(n={n}, m={m}, tau={tau})")

                    if count >= 30:
                        break
                if count >= 30:
                    break

    # Generate metadata file
    metadata = {
        'total_instances': len(all_files),
        'categories': {cat: len([f for f in all_files if cat in f])
                       for cat in configs},
        'failure_rate_default': failure_rate,
        'repair_rate_default': repair_rate,
        'due_date_tightness_values': taus,
        'failure_rate_regimes': {
            'low': 0.01,
            'medium': 0.05,
            'high': 0.10,
        },
        'weibull_shapes_tested': [1.0, 1.5, 2.0, 3.0],
    }

    meta_path = os.path.join(output_dir, 'metadata.json')
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    if verbose:
        print(f"\nTotal instances generated: {len(all_files)}")
        print(f"Metadata saved to: {meta_path}")

    return all_files


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Generate test instances')
    parser.add_argument('--output', type=str, default='instances',
                        help='Output directory')
    args = parser.parse_args()

    generate_all_instances(args.output)
