"""
Discrete-Event Stochastic Simulation Environment for Parallel Machine Scheduling.

This module implements the simulation environment that serves three roles:
1. RL training environment (generating episodes with stochastic failures)
2. GA fitness evaluator (Monte Carlo estimation of makespan/CVaR)
3. Final robustness assessment (large-sample evaluation)

Supports Poisson (exponential) and Weibull failure processes with
preempt-resume semantics.
"""

import heapq
import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Tuple, Optional
import json
import copy


class EventType(Enum):
    JOB_START = 1
    JOB_COMPLETE = 2
    MACHINE_FAILURE = 3
    MACHINE_REPAIR = 4


@dataclass(order=True)
class Event:
    time: float
    event_type: EventType = field(compare=False)
    machine_id: int = field(compare=False)
    job_id: int = field(compare=False, default=-1)


@dataclass
class Job:
    job_id: int
    processing_time: float
    due_date: float
    release_date: float
    weight: float = 1.0
    remaining_time: float = 0.0  # For preempt-resume
    assigned_machine: int = -1
    start_time: float = -1.0
    completion_time: float = -1.0
    is_completed: bool = False
    is_assigned: bool = False

    def __post_init__(self):
        self.remaining_time = self.processing_time


@dataclass
class Machine:
    machine_id: int
    failure_rate: float = 0.05      # lambda for Poisson
    repair_rate: float = 0.2        # mu for repair
    weibull_shape: float = 1.0      # beta (1.0 = exponential)
    weibull_scale: float = 20.0     # eta
    is_available: bool = True
    current_job_id: int = -1
    available_at: float = 0.0
    total_load: float = 0.0
    failure_count: int = 0
    cumulative_downtime: float = 0.0


@dataclass
class SchedulingInstance:
    """A problem instance with jobs, machines, and failure parameters."""
    n_jobs: int
    n_machines: int
    processing_times: np.ndarray    # shape (n_jobs,) for identical machines
    due_dates: np.ndarray           # shape (n_jobs,)
    release_dates: np.ndarray       # shape (n_jobs,)
    weights: np.ndarray             # shape (n_jobs,)
    failure_rates: np.ndarray       # shape (n_machines,)
    repair_rates: np.ndarray        # shape (n_machines,)
    weibull_shapes: np.ndarray      # shape (n_machines,)
    weibull_scales: np.ndarray      # shape (n_machines,)

    def to_dict(self) -> dict:
        return {
            'n_jobs': self.n_jobs,
            'n_machines': self.n_machines,
            'processing_times': self.processing_times.tolist(),
            'due_dates': self.due_dates.tolist(),
            'release_dates': self.release_dates.tolist(),
            'weights': self.weights.tolist(),
            'failure_rates': self.failure_rates.tolist(),
            'repair_rates': self.repair_rates.tolist(),
            'weibull_shapes': self.weibull_shapes.tolist(),
            'weibull_scales': self.weibull_scales.tolist(),
        }

    @staticmethod
    def from_dict(d: dict) -> 'SchedulingInstance':
        return SchedulingInstance(
            n_jobs=d['n_jobs'],
            n_machines=d['n_machines'],
            processing_times=np.array(d['processing_times']),
            due_dates=np.array(d['due_dates']),
            release_dates=np.array(d['release_dates']),
            weights=np.array(d['weights']),
            failure_rates=np.array(d['failure_rates']),
            repair_rates=np.array(d['repair_rates']),
            weibull_shapes=np.array(d['weibull_shapes']),
            weibull_scales=np.array(d['weibull_scales']),
        )

    def save(self, filepath: str):
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @staticmethod
    def load(filepath: str) -> 'SchedulingInstance':
        with open(filepath, 'r') as f:
            return SchedulingInstance.from_dict(json.load(f))


class SimulationEnvironment:
    """
    Discrete-event simulation of parallel machine scheduling under
    stochastic machine failures.

    Used for:
    - RL training (step-by-step interaction)
    - GA fitness evaluation (full schedule execution)
    - Robustness assessment (batch evaluation)
    """

    def __init__(self, instance: SchedulingInstance,
                 failure_model: str = 'exponential',
                 rng_seed: Optional[int] = None):
        """
        Args:
            instance: Problem instance
            failure_model: 'exponential' or 'weibull'
            rng_seed: Random seed for reproducibility (used for CRN)
        """
        self.instance = instance
        self.failure_model = failure_model
        self.rng = np.random.RandomState(rng_seed)
        self.reset()

    def reset(self) -> dict:
        """Reset environment to initial state. Returns initial observation."""
        inst = self.instance

        self.jobs = [
            Job(job_id=j,
                processing_time=inst.processing_times[j],
                due_date=inst.due_dates[j],
                release_date=inst.release_dates[j],
                weight=inst.weights[j])
            for j in range(inst.n_jobs)
        ]

        self.machines = [
            Machine(machine_id=i,
                    failure_rate=inst.failure_rates[i],
                    repair_rate=inst.repair_rates[i],
                    weibull_shape=inst.weibull_shapes[i],
                    weibull_scale=inst.weibull_scales[i])
            for i in range(inst.n_machines)
        ]

        self.current_time = 0.0
        self.event_queue: List[Event] = []
        self.done = False
        self.n_assigned = 0
        self.n_completed = 0

        # Schedule initial failure events for each machine
        for m in self.machines:
            ttf = self._sample_time_to_failure(m)
            heapq.heappush(self.event_queue,
                           Event(ttf, EventType.MACHINE_FAILURE, m.machine_id))

        return self._get_state()

    def _sample_time_to_failure(self, machine: Machine) -> float:
        """Sample time to next failure from current time."""
        if self.failure_model == 'weibull':
            ttf = machine.weibull_scale * (
                self.rng.weibull(machine.weibull_shape)
            )
        else:  # exponential
            if machine.failure_rate > 0:
                ttf = self.rng.exponential(1.0 / machine.failure_rate)
            else:
                ttf = float('inf')
        return self.current_time + ttf

    def _sample_repair_time(self, machine: Machine) -> float:
        """Sample repair duration."""
        if machine.repair_rate > 0:
            return self.rng.exponential(1.0 / machine.repair_rate)
        return 0.0

    def _get_state(self) -> dict:
        """
        Return current state observation.
        Used by RL agent and for GNN embedding construction.
        """
        job_features = []
        for j in self.jobs:
            slack = j.due_date - self.current_time - j.remaining_time
            job_features.append({
                'job_id': j.job_id,
                'processing_time': j.processing_time,
                'remaining_time': j.remaining_time,
                'due_date': j.due_date,
                'release_date': j.release_date,
                'weight': j.weight,
                'slack': slack,
                'is_assigned': int(j.is_assigned),
                'is_completed': int(j.is_completed),
            })

        machine_features = []
        for m in self.machines:
            mtbf = 1.0 / m.failure_rate if m.failure_rate > 0 else float('inf')
            mttr = 1.0 / m.repair_rate if m.repair_rate > 0 else 0.0
            machine_features.append({
                'machine_id': m.machine_id,
                'is_available': int(m.is_available),
                'available_at': m.available_at,
                'total_load': m.total_load,
                'current_job_id': m.current_job_id,
                'mtbf': mtbf,
                'mttr': mttr,
                'failure_count': m.failure_count,
            })

        return {
            'current_time': self.current_time,
            'jobs': job_features,
            'machines': machine_features,
            'n_assigned': self.n_assigned,
            'n_completed': self.n_completed,
            'done': self.done,
        }

    def get_valid_actions(self) -> List[Tuple[int, int]]:
        """Return list of valid (job_id, machine_id) actions."""
        actions = []
        unassigned_jobs = [j for j in self.jobs
                          if not j.is_assigned and j.release_date <= self.current_time]
        available_machines = [m for m in self.machines if m.is_available]

        for j in unassigned_jobs:
            for m in available_machines:
                actions.append((j.job_id, m.machine_id))

        # If no jobs are released yet but some are unassigned, allow waiting
        if not actions and self.n_assigned < self.instance.n_jobs:
            # Find next release/repair event and advance time
            pass

        return actions

    def step(self, action: Tuple[int, int]) -> Tuple[dict, float, bool, dict]:
        """
        Execute one scheduling action (assign job to machine).

        Args:
            action: (job_id, machine_id)

        Returns:
            (next_state, reward, done, info)
        """
        job_id, machine_id = action
        job = self.jobs[job_id]
        machine = self.machines[machine_id]

        # Assign job
        job.is_assigned = True
        job.assigned_machine = machine_id
        self.n_assigned += 1

        # Schedule job start
        start_time = max(self.current_time, machine.available_at, job.release_date)
        job.start_time = start_time
        machine.available_at = start_time + job.processing_time
        machine.total_load += job.processing_time
        machine.current_job_id = job.job_id

        # Schedule completion event
        completion_time = start_time + job.processing_time
        heapq.heappush(self.event_queue,
                       Event(completion_time, EventType.JOB_COMPLETE,
                             machine_id, job_id))

        # Process any events (failures/repairs) that occur before this job starts
        # This is critical: without this, the RL agent never experiences failures
        while self.event_queue and self.event_queue[0].time <= start_time:
            event = heapq.heappop(self.event_queue)
            self.current_time = event.time
            self._handle_event(event)
        
        self.current_time = start_time

        # Compute reward
        reward = self._compute_reward(job, machine)

        # Check if all jobs assigned
        if self.n_assigned >= self.instance.n_jobs:
            # Process remaining events to completion
            self._process_all_remaining_events()
            self.done = True

        next_state = self._get_state()
        info = {'makespan': self._get_makespan() if self.done else None}

        return next_state, reward, self.done, info

    def _compute_reward(self, job: Job, machine: Machine) -> float:
        """
        Compute immediate reward for assigning job to machine.
        
        Design: reward is the negative of the INCREMENTAL makespan increase
        caused by this assignment, normalized by total workload. This gives
        clear learning signal: good assignments that balance load get higher
        rewards (closer to 0), while assignments that increase makespan
        get more negative rewards.
        """
        # Current makespan across all machines
        makespans = [m.available_at for m in self.machines]
        current_makespan = max(makespans)
        
        # What was makespan before this assignment?
        # machine.available_at already includes this job, so subtract it
        prev_avail = machine.available_at - job.processing_time
        prev_makespan = max(prev_avail if i == machine.machine_id else makespans[i]
                           for i, _ in enumerate(makespans))
        
        # Incremental makespan increase (0 if this didn't change the bottleneck)
        delta_makespan = max(0, current_makespan - prev_makespan)
        
        # Normalize by average processing time for scale-invariance
        avg_pt = np.mean(self.instance.processing_times)
        reward = -delta_makespan / avg_pt
        
        # Load balance bonus: reward choosing the least-loaded machine
        min_load = min(makespans)
        load_ratio = (machine.available_at - min_load) / (avg_pt + 1e-10)
        reward -= 0.1 * load_ratio
        
        # Terminal bonus: when all jobs are assigned, penalize final makespan
        if self.n_assigned >= self.instance.n_jobs:
            # Large reward proportional to how good the final makespan is
            # compared to a simple lower bound (total work / n_machines)
            lb = sum(self.instance.processing_times) / self.instance.n_machines
            ratio = current_makespan / lb
            reward += max(0, 5.0 * (2.0 - ratio))  # Bonus if within 2x of LB

        return reward

    def _process_events_until_decision(self):
        """Process failure/repair events that occur before next decision point."""
        while self.event_queue:
            event = self.event_queue[0]

            # Only process failure/repair events that happen before
            # all currently scheduled completions
            if event.event_type in (EventType.MACHINE_FAILURE, EventType.MACHINE_REPAIR):
                if event.time <= self.current_time:
                    heapq.heappop(self.event_queue)
                    self._handle_event(event)
                else:
                    break
            else:
                break

    def _process_all_remaining_events(self):
        """Process remaining events until all jobs are completed."""
        max_iterations = self.instance.n_jobs * 100  # Safety limit
        iterations = 0
        while self.event_queue and iterations < max_iterations:
            # Check if all jobs are completed
            if self.n_completed >= self.instance.n_jobs:
                break
            event = heapq.heappop(self.event_queue)
            self.current_time = event.time
            self._handle_event(event)
            iterations += 1

    def _handle_event(self, event: Event):
        """Handle a single simulation event."""
        if event.event_type == EventType.JOB_COMPLETE:
            self._handle_job_completion(event)
        elif event.event_type == EventType.MACHINE_FAILURE:
            self._handle_machine_failure(event)
        elif event.event_type == EventType.MACHINE_REPAIR:
            self._handle_machine_repair(event)

    def _handle_job_completion(self, event: Event):
        """Handle job completion event."""
        job = self.jobs[event.job_id]
        machine = self.machines[event.machine_id]

        job.completion_time = event.time
        job.is_completed = True
        job.remaining_time = 0.0
        machine.current_job_id = -1
        self.n_completed += 1

    def _handle_machine_failure(self, event: Event):
        """Handle machine failure event (preempt-resume)."""
        machine = self.machines[event.machine_id]

        if not machine.is_available:
            return  # Already failed

        machine.is_available = False
        machine.failure_count += 1

        # Interrupt current job if any
        if machine.current_job_id >= 0:
            job = self.jobs[machine.current_job_id]
            if not job.is_completed:
                # Calculate remaining processing time
                elapsed = event.time - job.start_time
                job.remaining_time = max(0, job.processing_time - elapsed)

                # Remove the pending completion event
                self.event_queue = [
                    e for e in self.event_queue
                    if not (e.event_type == EventType.JOB_COMPLETE
                            and e.job_id == job.job_id)
                ]
                heapq.heapify(self.event_queue)

        # Schedule repair
        repair_duration = self._sample_repair_time(machine)
        repair_time = event.time + repair_duration
        machine.cumulative_downtime += repair_duration
        heapq.heappush(self.event_queue,
                       Event(repair_time, EventType.MACHINE_REPAIR,
                             machine.machine_id))

    def _handle_machine_repair(self, event: Event):
        """Handle machine repair event."""
        machine = self.machines[event.machine_id]
        machine.is_available = True
        machine.available_at = event.time

        # Resume interrupted job if any
        if machine.current_job_id >= 0:
            job = self.jobs[machine.current_job_id]
            if not job.is_completed and job.remaining_time > 0:
                job.start_time = event.time
                completion_time = event.time + job.remaining_time
                machine.available_at = completion_time
                heapq.heappush(self.event_queue,
                               Event(completion_time, EventType.JOB_COMPLETE,
                                     machine.machine_id, job.job_id))

        # Schedule next failure — but only if we're within a reasonable
        # time horizon. This prevents unbounded failure cascades that can
        # produce makespans 10-100x larger than expected.
        total_work = np.sum(self.instance.processing_times)
        time_horizon = total_work * 3  # 3x total workload is generous
        if event.time < time_horizon:
            ttf = self._sample_time_to_failure(machine)
            if ttf < time_horizon:
                heapq.heappush(self.event_queue,
                               Event(ttf, EventType.MACHINE_FAILURE, machine.machine_id))

    def _get_makespan(self) -> float:
        """Return current makespan (max completion time)."""
        completed = [j.completion_time for j in self.jobs if j.is_completed]
        return max(completed) if completed else 0.0

    def get_metrics(self) -> dict:
        """Compute all performance metrics for the current schedule."""
        makespan = self._get_makespan()
        tardiness = sum(
            max(0, j.completion_time - j.due_date) * j.weight
            for j in self.jobs if j.is_completed
        )
        on_time = sum(
            1 for j in self.jobs
            if j.is_completed and j.completion_time <= j.due_date
        )
        total_downtime = sum(m.cumulative_downtime for m in self.machines)
        total_failures = sum(m.failure_count for m in self.machines)

        return {
            'makespan': makespan,
            'total_weighted_tardiness': tardiness,
            'on_time_count': on_time,
            'on_time_pct': on_time / self.instance.n_jobs * 100 if self.instance.n_jobs > 0 else 0,
            'total_downtime': total_downtime,
            'total_failures': total_failures,
        }

    def execute_schedule(self, job_sequence: List[int],
                         rng_seed: Optional[int] = None) -> dict:
        """
        Execute a complete schedule (given as job priority sequence)
        and return metrics. Used for GA fitness evaluation.

        Implements a simple list-scheduling decoder: assigns jobs in
        the given priority order, each to the machine with the
        earliest available time, handling failures via discrete events.
        """
        if rng_seed is not None:
            self.rng = np.random.RandomState(rng_seed)
        self.reset()

        # Track machine available times and failure windows
        machine_avail = [0.0] * self.instance.n_machines

        # Pre-generate failure/repair sequences for each machine
        failure_events = {}  # machine_id -> list of (fail_start, fail_end)
        for m in self.machines:
            failures = []
            t = 0.0
            max_time = np.sum(self.instance.processing_times) * 3  # upper bound
            while t < max_time:
                ttf = self._sample_time_to_failure(m)
                fail_start = ttf
                repair_dur = self._sample_repair_time(m)
                fail_end = fail_start + repair_dur
                if fail_start < max_time:
                    failures.append((fail_start, fail_end))
                    m_copy_rate = m.failure_rate
                    # Reset RNG time reference for next failure
                    t = fail_end
                    # For simplicity, generate from absolute times
                    break_after = fail_end
                    next_ttf = break_after + (
                        self.rng.exponential(1.0 / m.failure_rate)
                        if self.failure_model == 'exponential' and m.failure_rate > 0
                        else m.weibull_scale * self.rng.weibull(m.weibull_shape)
                        if self.failure_model == 'weibull'
                        else float('inf')
                    )
                    next_repair = self._sample_repair_time(m)
                    if next_ttf < max_time:
                        failures.append((next_ttf, next_ttf + next_repair))
                        t = next_ttf + next_repair
                    else:
                        break
                else:
                    break
                if len(failures) > 50:
                    break
            failure_events[m.machine_id] = failures

        def get_actual_completion(machine_id, start, processing_time):
            """Account for failures during processing (preempt-resume)."""
            remaining = processing_time
            current = start
            for f_start, f_end in failure_events.get(machine_id, []):
                if f_start >= current + remaining:
                    break  # Failure after job completes
                if f_end <= current:
                    continue  # Failure before job starts
                # Failure overlaps with processing
                if f_start > current:
                    remaining -= (f_start - current)
                    current = f_end  # Resume after repair
                else:
                    current = max(current, f_end)
            return current + remaining

        # Assign jobs in priority order
        job_completions = {}
        for job_id in job_sequence:
            p = self.instance.processing_times[job_id]
            r = self.instance.release_dates[job_id]

            # Find machine with earliest availability
            best_m = min(range(self.instance.n_machines),
                         key=lambda i: max(machine_avail[i], r))

            start = max(machine_avail[best_m], r)
            completion = get_actual_completion(best_m, start, p)

            job_completions[job_id] = completion
            machine_avail[best_m] = completion

        # Compute metrics
        makespan = max(job_completions.values()) if job_completions else 0.0
        tardiness = sum(
            max(0, job_completions[j] - self.instance.due_dates[j]) * self.instance.weights[j]
            for j in range(self.instance.n_jobs)
        )
        on_time = sum(
            1 for j in range(self.instance.n_jobs)
            if job_completions.get(j, float('inf')) <= self.instance.due_dates[j]
        )

        return {
            'makespan': makespan,
            'total_weighted_tardiness': tardiness,
            'on_time_count': on_time,
            'on_time_pct': on_time / self.instance.n_jobs * 100,
            'total_downtime': 0.0,  # Approximated in this fast path
            'total_failures': sum(len(v) for v in failure_events.values()),
        }

    def monte_carlo_evaluate(self, job_sequence: List[int],
                             n_replications: int = 50,
                             base_seed: int = 42) -> dict:
        """
        Evaluate a schedule via Monte Carlo simulation over multiple
        failure scenarios. Implements CRN by using sequential seeds.

        Args:
            job_sequence: Job priority permutation
            n_replications: Number of simulation replications (N_sim)
            base_seed: Base seed for CRN

        Returns:
            dict with aggregate statistics
        """
        makespans = []
        tardiness_vals = []
        on_time_pcts = []

        for rep in range(n_replications):
            seed = base_seed + rep  # CRN: same seeds across individuals
            self.rng = np.random.RandomState(seed)
            metrics = self.execute_schedule(job_sequence, rng_seed=seed)
            makespans.append(metrics['makespan'])
            tardiness_vals.append(metrics['total_weighted_tardiness'])
            on_time_pcts.append(metrics['on_time_pct'])

        makespans = np.array(makespans)
        gamma = 0.95  # CVaR confidence level

        # CVaR estimation
        sorted_ms = np.sort(makespans)
        k = int(np.ceil(gamma * n_replications))
        cvar = np.mean(sorted_ms[k:]) if k < len(sorted_ms) else sorted_ms[-1]

        return {
            'mean_makespan': float(np.mean(makespans)),
            'std_makespan': float(np.std(makespans)),
            'cv_makespan': float(np.std(makespans) / np.mean(makespans)) if np.mean(makespans) > 0 else 0,
            'cvar_95': float(cvar),
            'worst_makespan': float(np.max(makespans)),
            'best_makespan': float(np.min(makespans)),
            'mean_tardiness': float(np.mean(tardiness_vals)),
            'mean_on_time_pct': float(np.mean(on_time_pcts)),
            'makespans': makespans.tolist(),
        }
