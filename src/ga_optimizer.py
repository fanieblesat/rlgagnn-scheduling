"""
Genetic Algorithm (GA) optimizer for schedule refinement.

Features:
- RL-seeded population initialization
- Order Crossover (OX) for permutation encoding
- Swap and insertion mutations
- Monte Carlo simulation-based fitness evaluation with CRN
- Elitism preservation
"""

import numpy as np
from typing import List, Tuple, Optional, Callable
import copy
import time

from .simulation import SimulationEnvironment, SchedulingInstance


class Individual:
    """A candidate schedule represented as a job permutation."""

    def __init__(self, chromosome: List[int]):
        self.chromosome = list(chromosome)
        self.fitness = None
        self.mean_makespan = None
        self.cvar_95 = None
        self.metrics = None

    def __repr__(self):
        return f"Individual(fitness={self.fitness:.4f})" if self.fitness else "Individual(unevaluated)"


class GAOptimizer:
    """
    Genetic Algorithm for schedule optimization with Monte Carlo
    simulation-based fitness evaluation.
    """

    def __init__(self, instance: SchedulingInstance,
                 pop_size: int = 100,
                 n_generations: int = 200,
                 crossover_prob: float = 0.8,
                 mutation_swap_prob: float = 0.15,
                 mutation_insert_prob: float = 0.05,
                 tournament_size: int = 5,
                 elite_rate: float = 0.05,
                 rl_seed_fraction: float = 0.25,
                 n_sim: int = 50,
                 alpha: float = 0.6,
                 failure_model: str = 'exponential',
                 crn_base_seed: int = 42,
                 rng_seed: int = 0):
        """
        Args:
            instance: Scheduling problem instance
            pop_size: Population size
            n_generations: Number of GA generations
            crossover_prob: Probability of applying crossover
            mutation_swap_prob: Probability of swap mutation
            mutation_insert_prob: Probability of insertion mutation
            tournament_size: Tournament selection size
            elite_rate: Fraction of population preserved as elites
            rl_seed_fraction: Fraction of initial population from RL
            n_sim: Monte Carlo replications per fitness evaluation
            alpha: Trade-off parameter (makespan vs resilience)
            failure_model: 'exponential' or 'weibull'
            crn_base_seed: Base seed for Common Random Numbers
            rng_seed: Random seed for GA operations
        """
        self.instance = instance
        self.pop_size = pop_size
        self.n_generations = n_generations
        self.crossover_prob = crossover_prob
        self.mutation_swap_prob = mutation_swap_prob
        self.mutation_insert_prob = mutation_insert_prob
        self.tournament_size = tournament_size
        self.elite_count = max(1, int(elite_rate * pop_size))
        self.rl_seed_fraction = rl_seed_fraction
        self.n_sim = n_sim
        self.alpha = alpha
        self.failure_model = failure_model
        self.crn_base_seed = crn_base_seed
        self.rng = np.random.RandomState(rng_seed)
        self.n_jobs = instance.n_jobs

        # Tracking
        self.best_fitness_history = []
        self.avg_fitness_history = []
        self.best_individual = None

    def initialize_population(self, rl_seeds: Optional[List[List[int]]] = None) -> List[Individual]:
        """
        Initialize population with RL-seeded and random individuals.

        Args:
            rl_seeds: List of job permutations from RL policy

        Returns:
            Initial population
        """
        population = []

        # Insert RL elite seeds
        if rl_seeds:
            n_rl = min(len(rl_seeds), int(self.rl_seed_fraction * self.pop_size))
            for i in range(n_rl):
                population.append(Individual(rl_seeds[i]))

        # Fill remaining with random permutations
        while len(population) < self.pop_size:
            perm = self.rng.permutation(self.n_jobs).tolist()
            population.append(Individual(perm))

        return population

    def evaluate_fitness(self, individual: Individual,
                         generation_seed: int) -> float:
        """
        Evaluate fitness via Monte Carlo simulation.

        Uses CRN: base_seed + generation_seed ensures same failure
        scenarios across all individuals in a generation.

        Args:
            individual: Candidate schedule
            generation_seed: Generation-specific seed offset for CRN

        Returns:
            Fitness value (higher is better)
        """
        env = SimulationEnvironment(
            self.instance, failure_model=self.failure_model
        )

        crn_seed = self.crn_base_seed + generation_seed * 1000
        mc_results = env.monte_carlo_evaluate(
            individual.chromosome,
            n_replications=self.n_sim,
            base_seed=crn_seed
        )

        mean_ms = mc_results['mean_makespan']
        cvar = mc_results['cvar_95']

        # Fitness: weighted combination (higher is better)
        # Avoid division by zero
        if mean_ms > 0 and cvar > 0:
            fitness = self.alpha * (1.0 / mean_ms) + (1 - self.alpha) * (1.0 / cvar)
        else:
            fitness = 0.0

        individual.fitness = fitness
        individual.mean_makespan = mean_ms
        individual.cvar_95 = cvar
        individual.metrics = mc_results

        return fitness

    def tournament_selection(self, population: List[Individual]) -> Individual:
        """Select individual via tournament selection."""
        tournament = [
            population[self.rng.randint(0, len(population))]
            for _ in range(self.tournament_size)
        ]
        return max(tournament, key=lambda ind: ind.fitness or -float('inf'))

    @staticmethod
    def order_crossover(parent1: List[int], parent2: List[int],
                        rng: np.random.RandomState) -> Tuple[List[int], List[int]]:
        """
        Order Crossover (OX) for permutation-based chromosomes.

        1. Select a random substring from parent1
        2. Copy it to offspring1
        3. Fill remaining positions from parent2 in relative order
        """
        n = len(parent1)
        c1, c2 = sorted(rng.choice(n, 2, replace=False))

        # Offspring 1
        child1 = [-1] * n
        child1[c1:c2+1] = parent1[c1:c2+1]
        segment = set(parent1[c1:c2+1])
        fill = [g for g in parent2 if g not in segment]
        idx = 0
        for i in range(n):
            if child1[i] == -1:
                child1[i] = fill[idx]
                idx += 1

        # Offspring 2
        child2 = [-1] * n
        child2[c1:c2+1] = parent2[c1:c2+1]
        segment = set(parent2[c1:c2+1])
        fill = [g for g in parent1 if g not in segment]
        idx = 0
        for i in range(n):
            if child2[i] == -1:
                child2[i] = fill[idx]
                idx += 1

        return child1, child2

    def swap_mutation(self, chromosome: List[int]) -> List[int]:
        """Swap two random positions."""
        chrom = list(chromosome)
        i, j = self.rng.choice(len(chrom), 2, replace=False)
        chrom[i], chrom[j] = chrom[j], chrom[i]
        return chrom

    def insertion_mutation(self, chromosome: List[int]) -> List[int]:
        """Remove element and reinsert at random position."""
        chrom = list(chromosome)
        i = self.rng.randint(0, len(chrom))
        gene = chrom.pop(i)
        j = self.rng.randint(0, len(chrom) + 1)
        chrom.insert(j, gene)
        return chrom

    def evolve(self, rl_seeds: Optional[List[List[int]]] = None,
               verbose: bool = True) -> Tuple[Individual, dict]:
        """
        Run the full GA evolution.

        Args:
            rl_seeds: RL-generated elite schedules for seeding
            verbose: Print progress

        Returns:
            (best_individual, evolution_history)
        """
        start_time = time.time()

        # Initialize
        population = self.initialize_population(rl_seeds)

        if verbose:
            n_rl = len(rl_seeds) if rl_seeds else 0
            print(f"  GA initialized: {self.pop_size} individuals "
                  f"({n_rl} RL-seeded, {self.pop_size - n_rl} random)")

        # Evaluate initial population
        for ind in population:
            self.evaluate_fitness(ind, generation_seed=0)

        for gen in range(self.n_generations):
            # Sort by fitness
            population.sort(key=lambda x: x.fitness or -float('inf'), reverse=True)

            # Track best
            if self.best_individual is None or \
               population[0].fitness > (self.best_individual.fitness or -float('inf')):
                self.best_individual = copy.deepcopy(population[0])

            best_fit = population[0].fitness
            avg_fit = np.mean([ind.fitness for ind in population if ind.fitness])
            self.best_fitness_history.append(best_fit)
            self.avg_fitness_history.append(avg_fit)

            if verbose and (gen + 1) % 20 == 0:
                print(f"  Gen {gen+1}/{self.n_generations} | "
                      f"Best Fitness: {best_fit:.6f} | "
                      f"Best Makespan: {population[0].mean_makespan:.1f} | "
                      f"Avg Fitness: {avg_fit:.6f}")

            # Elitism: preserve top individuals
            new_population = [copy.deepcopy(ind) for ind in population[:self.elite_count]]

            # Generate offspring
            while len(new_population) < self.pop_size:
                parent1 = self.tournament_selection(population)
                parent2 = self.tournament_selection(population)

                if self.rng.random() < self.crossover_prob:
                    child1_chrom, child2_chrom = self.order_crossover(
                        parent1.chromosome, parent2.chromosome, self.rng
                    )
                else:
                    child1_chrom = list(parent1.chromosome)
                    child2_chrom = list(parent2.chromosome)

                # Mutations
                if self.rng.random() < self.mutation_swap_prob:
                    child1_chrom = self.swap_mutation(child1_chrom)
                if self.rng.random() < self.mutation_insert_prob:
                    child1_chrom = self.insertion_mutation(child1_chrom)

                if self.rng.random() < self.mutation_swap_prob:
                    child2_chrom = self.swap_mutation(child2_chrom)
                if self.rng.random() < self.mutation_insert_prob:
                    child2_chrom = self.insertion_mutation(child2_chrom)

                new_population.append(Individual(child1_chrom))
                if len(new_population) < self.pop_size:
                    new_population.append(Individual(child2_chrom))

            # Evaluate new individuals (skip elites that are unchanged)
            for ind in new_population[self.elite_count:]:
                self.evaluate_fitness(ind, generation_seed=gen + 1)

            population = new_population

        # Final sort
        population.sort(key=lambda x: x.fitness or -float('inf'), reverse=True)
        if population[0].fitness > (self.best_individual.fitness or -float('inf')):
            self.best_individual = copy.deepcopy(population[0])

        elapsed = time.time() - start_time

        history = {
            'best_fitness': self.best_fitness_history,
            'avg_fitness': self.avg_fitness_history,
            'elapsed_seconds': elapsed,
            'best_makespan': self.best_individual.mean_makespan,
            'best_cvar': self.best_individual.cvar_95,
        }

        if verbose:
            print(f"  GA complete in {elapsed:.1f}s | "
                  f"Best Makespan: {self.best_individual.mean_makespan:.1f} | "
                  f"CVaR95: {self.best_individual.cvar_95:.1f}")

        return self.best_individual, history
