"""
Baseline dispatching rules and performance metrics.

Dispatching rules:
- LPT: Longest Processing Time first
- SPT: Shortest Processing Time first
- EDD: Earliest Due Date first
- ATC: Apparent Tardiness Cost

Statistical analysis utilities for experimental comparison.
"""

import numpy as np
from typing import List, Dict
from scipy import stats

from .simulation import SimulationEnvironment, SchedulingInstance


# ====================================================================
# Dispatching Rules
# ====================================================================

def lpt_schedule(instance: SchedulingInstance) -> List[int]:
    """Longest Processing Time first."""
    return sorted(range(instance.n_jobs),
                  key=lambda j: instance.processing_times[j],
                  reverse=True)


def spt_schedule(instance: SchedulingInstance) -> List[int]:
    """Shortest Processing Time first."""
    return sorted(range(instance.n_jobs),
                  key=lambda j: instance.processing_times[j])


def edd_schedule(instance: SchedulingInstance) -> List[int]:
    """Earliest Due Date first."""
    return sorted(range(instance.n_jobs),
                  key=lambda j: instance.due_dates[j])


def atc_schedule(instance: SchedulingInstance, k: float = 2.0) -> List[int]:
    """
    Apparent Tardiness Cost dispatching rule.
    Priority: (w_j / p_j) * exp(-max(0, d_j - p_j - t) / (k * p_bar))
    For static ordering, uses t=0 and average processing time.
    """
    p_bar = np.mean(instance.processing_times)
    n = instance.n_jobs
    priorities = []

    for j in range(n):
        p_j = instance.processing_times[j]
        d_j = instance.due_dates[j]
        w_j = instance.weights[j]

        slack = max(0, d_j - p_j)
        priority = (w_j / p_j) * np.exp(-slack / (k * p_bar + 1e-10))
        priorities.append(priority)

    return sorted(range(n), key=lambda j: priorities[j], reverse=True)


def random_schedule(instance: SchedulingInstance, rng_seed: int = 0) -> List[int]:
    """Random job ordering."""
    rng = np.random.RandomState(rng_seed)
    return rng.permutation(instance.n_jobs).tolist()


# ====================================================================
# Evaluation
# ====================================================================

def evaluate_schedule(instance: SchedulingInstance, job_sequence: List[int],
                      n_replications: int = 50, failure_model: str = 'exponential',
                      base_seed: int = 42) -> Dict:
    """
    Evaluate a schedule via Monte Carlo simulation.

    Returns dict with all performance metrics.
    """
    env = SimulationEnvironment(instance, failure_model=failure_model)
    return env.monte_carlo_evaluate(job_sequence, n_replications, base_seed)


# ====================================================================
# Statistical Tests
# ====================================================================

def wilcoxon_test(data1: List[float], data2: List[float]) -> Dict:
    """
    Wilcoxon signed-rank test for paired samples.
    """
    try:
        stat, p_value = stats.wilcoxon(data1, data2, alternative='two-sided')
    except ValueError:
        stat, p_value = 0.0, 1.0

    return {
        'statistic': float(stat),
        'p_value': float(p_value),
        'significant_005': p_value < 0.05,
        'significant_001': p_value < 0.01,
    }


def vargha_delaney(data1: List[float], data2: List[float]) -> Dict:
    """
    Vargha-Delaney A_12 effect size statistic.
    A_12 > 0.5 means data1 tends to be larger than data2.
    For minimization: A_12 < 0.5 means method1 is better.
    """
    n1, n2 = len(data1), len(data2)
    if n1 == 0 or n2 == 0:
        return {'a12': 0.5, 'effect': 'negligible'}

    r = 0
    for x in data1:
        for y in data2:
            if x > y:
                r += 1
            elif x == y:
                r += 0.5

    a12 = r / (n1 * n2)

    # Interpret
    if a12 >= 0.71 or a12 <= 0.29:
        effect = 'large'
    elif a12 >= 0.64 or a12 <= 0.36:
        effect = 'medium'
    elif a12 >= 0.56 or a12 <= 0.44:
        effect = 'small'
    else:
        effect = 'negligible'

    return {'a12': float(a12), 'effect': effect}


def friedman_test(data_matrix: np.ndarray) -> Dict:
    """
    Friedman test for comparing multiple methods across instances.

    Args:
        data_matrix: (n_instances, n_methods) array of performance values

    Returns:
        Test results
    """
    try:
        stat, p_value = stats.friedmanchisquare(
            *[data_matrix[:, i] for i in range(data_matrix.shape[1])]
        )
    except ValueError:
        stat, p_value = 0.0, 1.0

    return {
        'statistic': float(stat),
        'p_value': float(p_value),
        'significant_005': p_value < 0.05,
    }


def holm_bonferroni_correction(p_values: List[float], alpha: float = 0.05) -> List[bool]:
    """
    Holm-Bonferroni correction for multiple comparisons.

    Returns list of booleans indicating significance after correction.
    """
    n = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    results = [False] * n

    for rank, (orig_idx, p) in enumerate(indexed):
        adjusted_alpha = alpha / (n - rank)
        if p <= adjusted_alpha:
            results[orig_idx] = True
        else:
            break  # All subsequent are non-significant

    return results


def confidence_interval(data: List[float], confidence: float = 0.95) -> tuple:
    """Compute confidence interval for the mean."""
    n = len(data)
    if n < 2:
        return (np.mean(data), np.mean(data))
    mean = np.mean(data)
    se = stats.sem(data)
    h = se * stats.t.ppf((1 + confidence) / 2, n - 1)
    return (mean - h, mean + h)


def compute_cvar(values: List[float], gamma: float = 0.95) -> float:
    """Compute Conditional Value-at-Risk at level gamma."""
    sorted_vals = np.sort(values)
    k = int(np.ceil(gamma * len(sorted_vals)))
    if k >= len(sorted_vals):
        return sorted_vals[-1]
    return np.mean(sorted_vals[k:])
