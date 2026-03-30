"""
Deep Q-Network (DQN) Reinforcement Learning Agent with GNN state embeddings.

The agent learns a scheduling policy by interacting with the stochastic
simulation environment. States are encoded via the GNN as graph embeddings.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
from typing import List, Tuple, Optional

from .gnn_model import GNNEncoder, BipartiteGraphData, state_to_graph
from .simulation import SimulationEnvironment, SchedulingInstance


class ReplayBuffer:
    """Experience replay buffer for DQN."""

    def __init__(self, capacity: int = 50000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state_embed, action_idx, reward, next_state_embed, done):
        self.buffer.append((state_embed, action_idx, reward, next_state_embed, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.stack(states),
            torch.tensor(actions, dtype=torch.long),
            torch.tensor(rewards, dtype=torch.float32),
            torch.stack(next_states),
            torch.tensor(dones, dtype=torch.float32),
        )

    def __len__(self):
        return len(self.buffer)


class QNetwork(nn.Module):
    """Q-value network: takes graph embedding and outputs Q-values for all possible actions."""

    def __init__(self, embed_dim: int = 128, hidden_dim: int = 256, max_actions: int = 500):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, max_actions),
        )

    def forward(self, x):
        return self.network(x)


class DQNAgent:
    """
    DQN Agent with GNN-encoded states for parallel machine scheduling.
    """

    def __init__(self, instance: SchedulingInstance,
                 d_hidden: int = 64, d_embed: int = 128, n_gnn_layers: int = 3,
                 lr: float = 1e-3, gamma: float = 0.99,
                 epsilon_start: float = 1.0, epsilon_end: float = 0.05,
                 epsilon_decay_frac: float = 0.8,
                 buffer_size: int = 50000, batch_size: int = 64,
                 target_update_freq: int = 100,
                 device: str = 'cpu'):

        self.instance = instance
        self.device = torch.device(device)
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.max_actions = instance.n_jobs * instance.n_machines

        # Epsilon schedule
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_frac = epsilon_decay_frac

        # Networks
        self.gnn = GNNEncoder(
            d_job=8, d_machine=6, d_hidden=d_hidden,
            d_embed=d_embed, n_layers=n_gnn_layers
        ).to(self.device)

        self.q_network = QNetwork(
            embed_dim=d_embed, hidden_dim=256, max_actions=self.max_actions
        ).to(self.device)

        self.target_network = QNetwork(
            embed_dim=d_embed, hidden_dim=256, max_actions=self.max_actions
        ).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())

        # Joint optimizer for GNN + Q-network
        self.optimizer = optim.Adam(
            list(self.gnn.parameters()) + list(self.q_network.parameters()),
            lr=lr
        )

        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size)
        self.training_step = 0

    def _action_to_index(self, action: Tuple[int, int]) -> int:
        """Convert (job_id, machine_id) to flat index."""
        return action[0] * self.instance.n_machines + action[1]

    def _index_to_action(self, idx: int) -> Tuple[int, int]:
        """Convert flat index to (job_id, machine_id)."""
        job_id = idx // self.instance.n_machines
        machine_id = idx % self.instance.n_machines
        return (job_id, machine_id)

    def get_state_embedding(self, state: dict) -> torch.Tensor:
        """Compute GNN embedding for a state."""
        graph = state_to_graph(state, self.device)
        with torch.no_grad():
            embedding = self.gnn(graph)
        return embedding

    def select_action(self, state: dict, valid_actions: List[Tuple[int, int]],
                      training: bool = True) -> Tuple[int, int]:
        """
        Select action using epsilon-greedy policy.

        Args:
            state: Current simulation state
            valid_actions: List of valid (job_id, machine_id) pairs
            training: Whether to use exploration

        Returns:
            Selected (job_id, machine_id) action
        """
        if not valid_actions:
            return None

        if training and random.random() < self.epsilon:
            return random.choice(valid_actions)

        graph = state_to_graph(state, self.device)
        self.gnn.eval()
        self.q_network.eval()
        with torch.no_grad():
            embedding = self.gnn(graph)
            q_values = self.q_network(embedding.unsqueeze(0)).squeeze(0)
        self.gnn.train()
        self.q_network.train()

        # Mask invalid actions
        valid_indices = [self._action_to_index(a) for a in valid_actions]
        valid_q = {idx: q_values[idx].item() for idx in valid_indices if idx < len(q_values)}

        if not valid_q:
            return random.choice(valid_actions)

        best_idx = max(valid_q, key=valid_q.get)
        return self._index_to_action(best_idx)

    def update(self):
        """Perform one step of DQN training."""
        if len(self.replay_buffer) < self.batch_size:
            return 0.0

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        # Current Q-values
        current_q = self.q_network(states)
        current_q_actions = current_q.gather(1, actions.unsqueeze(1).clamp(max=current_q.shape[1]-1)).squeeze(1)

        # Target Q-values
        with torch.no_grad():
            next_q = self.target_network(next_states)
            max_next_q = next_q.max(dim=1)[0]
            target_q = rewards + self.gamma * max_next_q * (1 - dones)

        # Loss
        loss = nn.functional.mse_loss(current_q_actions, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.gnn.parameters()) + list(self.q_network.parameters()), 1.0
        )
        self.optimizer.step()
        self.training_step += 1

        return loss.item()

    def update_target_network(self):
        """Copy Q-network weights to target network."""
        self.target_network.load_state_dict(self.q_network.state_dict())

    def update_epsilon(self, episode: int, total_episodes: int):
        """Linear epsilon decay."""
        decay_episodes = int(total_episodes * self.epsilon_decay_frac)
        if episode < decay_episodes:
            self.epsilon = self.epsilon_start - (
                (self.epsilon_start - self.epsilon_end) * episode / decay_episodes
            )
        else:
            self.epsilon = self.epsilon_end

    def train_agent(self, n_episodes: int = 5000,
                    failure_model: str = 'exponential',
                    verbose: bool = True) -> List[float]:
        """
        Train the DQN agent through interaction with the simulation environment.
        """
        episode_rewards = []
        best_makespan = float('inf')

        for episode in range(n_episodes):
            env = SimulationEnvironment(
                self.instance, failure_model=failure_model,
                rng_seed=episode
            )
            state = env.reset()
            total_reward = 0.0
            prev_embedding = self.get_state_embedding(state)
            steps = 0
            max_steps = self.instance.n_jobs * 3  # Safety limit

            while not env.done and steps < max_steps:
                valid_actions = env.get_valid_actions()

                if not valid_actions:
                    # No valid actions: either jobs not released or machines down
                    # Advance time to the next relevant event
                    if env.event_queue:
                        import heapq
                        event = heapq.heappop(env.event_queue)
                        env.current_time = event.time
                        env._handle_event(event)
                        state = env._get_state()
                        prev_embedding = self.get_state_embedding(state)
                        steps += 1

                        # Also check if we can assign unreleased jobs by
                        # advancing time to their release
                        if not env.get_valid_actions() and env.n_assigned < self.instance.n_jobs:
                            unassigned = [j for j in env.jobs if not j.is_assigned]
                            if unassigned:
                                next_release = min(j.release_date for j in unassigned)
                                if next_release > env.current_time:
                                    # Process events up to release
                                    while env.event_queue and env.event_queue[0].time <= next_release:
                                        ev = heapq.heappop(env.event_queue)
                                        env.current_time = ev.time
                                        env._handle_event(ev)
                                    env.current_time = next_release
                    else:
                        break  # No events left, shouldn't happen
                    continue

                action = self.select_action(state, valid_actions, training=True)
                if action is None:
                    break

                next_state, reward, done, info = env.step(action)
                total_reward += reward

                next_embedding = self.get_state_embedding(next_state)
                action_idx = self._action_to_index(action)

                self.replay_buffer.push(
                    prev_embedding.cpu(), action_idx, reward,
                    next_embedding.cpu(), float(done)
                )

                loss = self.update()
                state = next_state
                prev_embedding = next_embedding
                steps += 1

            # Update epsilon
            self.update_epsilon(episode, n_episodes)

            # Update target network periodically
            if (episode + 1) % self.target_update_freq == 0:
                self.update_target_network()

            episode_rewards.append(total_reward)
            
            # Track best makespan
            ms = info.get('makespan', float('inf')) if info else float('inf')
            if ms and ms < best_makespan:
                best_makespan = ms

            if verbose and (episode + 1) % 100 == 0:
                avg_reward = np.mean(episode_rewards[-100:])
                print(f"  Episode {episode+1}/{n_episodes} | "
                      f"Avg Reward: {avg_reward:.3f} | "
                      f"Makespan: {ms:.1f} | "
                      f"Best: {best_makespan:.1f} | "
                      f"Epsilon: {self.epsilon:.3f}")

        return episode_rewards

    def generate_schedule(self, failure_model: str = 'exponential',
                          epsilon: float = 0.0,
                          rng_seed: int = 0) -> List[int]:
        """
        Generate a schedule using the learned policy.

        Uses a DETERMINISTIC state progression (no failures) to extract
        the policy's preferred job ordering. The schedule is then evaluated
        for robustness separately via Monte Carlo simulation.

        This avoids the problem of noisy schedules from single failure
        realizations — the policy encodes robustness from training across
        many episodes; we just need the ordering it has learned.
        """
        old_epsilon = self.epsilon
        self.epsilon = epsilon

        # Create a no-failure environment to extract pure policy ordering
        # The policy was TRAINED with failures, so it has internalized
        # robustness; we just need a clean extraction of its preferences
        instance_copy = SchedulingInstance(
            n_jobs=self.instance.n_jobs,
            n_machines=self.instance.n_machines,
            processing_times=self.instance.processing_times.copy(),
            due_dates=self.instance.due_dates.copy(),
            release_dates=self.instance.release_dates.copy(),
            weights=self.instance.weights.copy(),
            failure_rates=np.zeros(self.instance.n_machines),  # No failures
            repair_rates=self.instance.repair_rates.copy(),
            weibull_shapes=self.instance.weibull_shapes.copy(),
            weibull_scales=np.full(self.instance.n_machines, 1e10),  # No failures
        )

        env = SimulationEnvironment(
            instance_copy, failure_model=failure_model,
            rng_seed=rng_seed
        )
        state = env.reset()
        job_sequence = []
        steps = 0

        while not env.done and len(job_sequence) < self.instance.n_jobs and steps < self.instance.n_jobs * 3:
            valid_actions = env.get_valid_actions()
            if not valid_actions:
                # Advance time to next release
                import heapq
                unassigned = [j for j in env.jobs if not j.is_assigned]
                if unassigned:
                    next_release = min(j.release_date for j in unassigned)
                    env.current_time = max(env.current_time, next_release)
                    state = env._get_state()
                    steps += 1
                    continue
                else:
                    break

            # Use the ORIGINAL instance's state features for GNN embedding
            # but with the clean environment for action selection
            action = self.select_action(state, valid_actions, training=False)
            if action is None:
                break

            job_sequence.append(action[0])
            state, _, _, _ = env.step(action)
            steps += 1

        self.epsilon = old_epsilon

        # Ensure all jobs are included
        assigned = set(job_sequence)
        for j in range(self.instance.n_jobs):
            if j not in assigned:
                job_sequence.append(j)

        return job_sequence

    def save(self, filepath: str):
        """Save model weights."""
        torch.save({
            'gnn_state': self.gnn.state_dict(),
            'q_network_state': self.q_network.state_dict(),
            'target_network_state': self.target_network.state_dict(),
        }, filepath)

    def load(self, filepath: str):
        """Load model weights."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.gnn.load_state_dict(checkpoint['gnn_state'])
        self.q_network.load_state_dict(checkpoint['q_network_state'])
        self.target_network.load_state_dict(checkpoint['target_network_state'])
