"""
Graph Neural Network (GNN) for bipartite job-machine graph embeddings.

Implements a GraphSAGE-style message-passing architecture adapted for
bipartite graphs. Produces fixed-size state embeddings for the RL agent.

Uses plain PyTorch (no PyTorch Geometric dependency) for portability.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional


class BipartiteGraphData:
    """
    Represents a bipartite job-machine graph with node and edge features.
    """

    def __init__(self, job_features: torch.Tensor, machine_features: torch.Tensor,
                 edge_index: torch.Tensor, edge_features: Optional[torch.Tensor] = None):
        """
        Args:
            job_features: (n_jobs, d_J) tensor
            machine_features: (n_machines, d_M) tensor
            edge_index: (2, n_edges) tensor, [0] = job indices, [1] = machine indices
            edge_features: (n_edges, d_E) optional tensor
        """
        self.job_features = job_features
        self.machine_features = machine_features
        self.edge_index = edge_index
        self.edge_features = edge_features
        self.n_jobs = job_features.shape[0]
        self.n_machines = machine_features.shape[0]


def state_to_graph(state: dict, device: torch.device = torch.device('cpu')) -> BipartiteGraphData:
    """
    Convert a simulation state dict to a BipartiteGraphData object.

    Job features: [processing_time, remaining_time, due_date, release_date,
                   weight, slack, is_assigned, is_completed]
    Machine features: [is_available, available_at, total_load, mtbf, mttr, failure_count]
    """
    jobs = state['jobs']
    machines = state['machines']
    current_time = state['current_time']

    # Normalize features
    max_pt = max(j['processing_time'] for j in jobs) if jobs else 1.0
    max_dd = max(j['due_date'] for j in jobs) if jobs else 1.0
    max_time = max(max_pt, max_dd, current_time, 1.0)

    job_feats = []
    for j in jobs:
        job_feats.append([
            j['processing_time'] / max_time,
            j['remaining_time'] / max_time,
            j['due_date'] / max_time,
            j['release_date'] / max_time,
            j['weight'],
            j['slack'] / max_time,
            float(j['is_assigned']),
            float(j['is_completed']),
        ])

    machine_feats = []
    for m in machines:
        mtbf_norm = min(m['mtbf'] / max_time, 10.0) if m['mtbf'] != float('inf') else 10.0
        machine_feats.append([
            float(m['is_available']),
            m['available_at'] / max_time,
            m['total_load'] / max_time,
            mtbf_norm,
            m['mttr'] / max_time if m['mttr'] != float('inf') else 0.0,
            m['failure_count'] / 10.0,  # Normalize
        ])

    job_tensor = torch.tensor(job_feats, dtype=torch.float32, device=device)
    machine_tensor = torch.tensor(machine_feats, dtype=torch.float32, device=device)

    # Full bipartite connectivity: each job connects to each machine
    n_jobs = len(jobs)
    n_machines = len(machines)
    job_idx = torch.arange(n_jobs, device=device).repeat_interleave(n_machines)
    machine_idx = torch.arange(n_machines, device=device).repeat(n_jobs)
    edge_index = torch.stack([job_idx, machine_idx], dim=0)

    # Edge features: processing time affinity
    edge_feats = []
    for j in range(n_jobs):
        for m in range(n_machines):
            edge_feats.append([
                jobs[j]['processing_time'] / max_time,
                1.0,  # Affinity placeholder (identical machines)
            ])
    edge_tensor = torch.tensor(edge_feats, dtype=torch.float32, device=device)

    return BipartiteGraphData(job_tensor, machine_tensor, edge_index, edge_tensor)


class BipartiteSAGELayer(nn.Module):
    """
    Single GraphSAGE-style message-passing layer for bipartite graphs.
    Updates job nodes by aggregating from machine neighbors and vice versa.
    """

    def __init__(self, in_dim_job: int, in_dim_machine: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        # Job update: aggregate from machines
        self.W_job = nn.Linear(in_dim_job + in_dim_machine, out_dim)
        # Machine update: aggregate from jobs
        self.W_machine = nn.Linear(in_dim_machine + in_dim_job, out_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm_j = nn.LayerNorm(out_dim)
        self.layer_norm_m = nn.LayerNorm(out_dim)

    def forward(self, job_h: torch.Tensor, machine_h: torch.Tensor,
                edge_index: torch.Tensor) -> tuple:
        """
        Args:
            job_h: (n_jobs, d_j)
            machine_h: (n_machines, d_m)
            edge_index: (2, n_edges) - [0] job indices, [1] machine indices

        Returns:
            updated (job_h, machine_h)
        """
        job_idx, machine_idx = edge_index[0], edge_index[1]
        n_jobs = job_h.shape[0]
        n_machines = machine_h.shape[0]

        # Aggregate machine features to jobs (mean aggregation)
        machine_msgs = machine_h[machine_idx]  # (n_edges, d_m)
        job_agg = torch.zeros(n_jobs, machine_h.shape[1],
                              device=job_h.device, dtype=job_h.dtype)
        job_counts = torch.zeros(n_jobs, 1, device=job_h.device, dtype=job_h.dtype)
        job_agg.scatter_add_(0, machine_idx.unsqueeze(1).expand_as(machine_msgs)
                             .clamp(max=n_jobs - 1),
                             machine_msgs)
        # Use proper scatter for job aggregation
        job_agg = torch.zeros(n_jobs, machine_h.shape[1],
                              device=job_h.device, dtype=job_h.dtype)
        for i in range(edge_index.shape[1]):
            j, m = job_idx[i], machine_idx[i]
            job_agg[j] += machine_h[m]
        job_deg = torch.bincount(job_idx, minlength=n_jobs).float().clamp(min=1)
        job_agg = job_agg / job_deg.unsqueeze(1)

        # Aggregate job features to machines (mean aggregation)
        machine_agg = torch.zeros(n_machines, job_h.shape[1],
                                  device=machine_h.device, dtype=machine_h.dtype)
        for i in range(edge_index.shape[1]):
            j, m = job_idx[i], machine_idx[i]
            machine_agg[m] += job_h[j]
        machine_deg = torch.bincount(machine_idx, minlength=n_machines).float().clamp(min=1)
        machine_agg = machine_agg / machine_deg.unsqueeze(1)

        # Update nodes
        new_job_h = self.W_job(torch.cat([job_h, job_agg], dim=1))
        new_job_h = self.layer_norm_j(F.relu(new_job_h))
        new_job_h = self.dropout(new_job_h)

        new_machine_h = self.W_machine(torch.cat([machine_h, machine_agg], dim=1))
        new_machine_h = self.layer_norm_m(F.relu(new_machine_h))
        new_machine_h = self.dropout(new_machine_h)

        return new_job_h, new_machine_h


class GNNEncoder(nn.Module):
    """
    Multi-layer GNN encoder for bipartite job-machine graphs.
    Produces a fixed-size graph embedding from variable-size inputs.
    """

    def __init__(self, d_job: int = 8, d_machine: int = 6, d_hidden: int = 64,
                 d_embed: int = 128, n_layers: int = 3, dropout: float = 0.1):
        """
        Args:
            d_job: Dimension of job node features
            d_machine: Dimension of machine node features
            d_hidden: Hidden dimension of GNN layers
            d_embed: Output embedding dimension
            n_layers: Number of message-passing layers
            dropout: Dropout rate
        """
        super().__init__()

        self.d_embed = d_embed

        # Input projections to common hidden dim
        self.job_input = nn.Linear(d_job, d_hidden)
        self.machine_input = nn.Linear(d_machine, d_hidden)

        # Message-passing layers
        self.layers = nn.ModuleList([
            BipartiteSAGELayer(d_hidden, d_hidden, d_hidden, dropout)
            for _ in range(n_layers)
        ])

        # Readout MLP
        self.readout = nn.Sequential(
            nn.Linear(2 * d_hidden, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_embed),
        )

    def forward(self, graph: BipartiteGraphData) -> torch.Tensor:
        """
        Compute graph-level embedding.

        Args:
            graph: BipartiteGraphData object

        Returns:
            embedding: (d_embed,) tensor
        """
        # Project inputs
        job_h = F.relu(self.job_input(graph.job_features))
        machine_h = F.relu(self.machine_input(graph.machine_features))

        # Message passing
        for layer in self.layers:
            job_h, machine_h = layer(job_h, machine_h, graph.edge_index)

        # Graph-level readout (mean pooling over both node sets)
        job_pool = job_h.mean(dim=0)        # (d_hidden,)
        machine_pool = machine_h.mean(dim=0)  # (d_hidden,)

        # Concatenate and project
        graph_repr = torch.cat([job_pool, machine_pool], dim=0)
        embedding = self.readout(graph_repr)

        return embedding

    def forward_with_node_embeddings(self, graph: BipartiteGraphData):
        """
        Return both graph embedding and individual node embeddings.
        Node embeddings are used for action-level Q-values.
        """
        job_h = F.relu(self.job_input(graph.job_features))
        machine_h = F.relu(self.machine_input(graph.machine_features))

        for layer in self.layers:
            job_h, machine_h = layer(job_h, machine_h, graph.edge_index)

        job_pool = job_h.mean(dim=0)
        machine_pool = machine_h.mean(dim=0)
        graph_repr = torch.cat([job_pool, machine_pool], dim=0)
        embedding = self.readout(graph_repr)

        return embedding, job_h, machine_h
