from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class GraphLearnerConfig:
    hidden_dim: int = 64
    metric_type: str = "weighted_cosine"
    num_heads: int = 4
    topk: int | None = None
    epsilon: float | None = None
    # Required when ``metric_type`` is one of the learned matrix variants.
    node_count: int | None = None
    # Hidden width of the hypernetwork used by ``instance_conditional``.
    # Defaults to ``hidden_dim`` when None.
    hyper_hidden_dim: int | None = None

    def __post_init__(self):
        if self.metric_type in {"learned_weight", "instance_conditional"}:
            if self.node_count is None or self.node_count <= 0:
                raise ValueError(
                    f"metric_type={self.metric_type!r} requires node_count to be a positive int; "
                    f"got {self.node_count!r}"
                )
        if self.hyper_hidden_dim is not None and self.hyper_hidden_dim <= 0:
            raise ValueError(
                f"hyper_hidden_dim must be a positive int when set; got {self.hyper_hidden_dim!r}"
            )


class GraphLearner(nn.Module):
    """
    Learn a dense operator-level graph from node states.

    The implementation is intentionally compact and keeps the metric variants
    used by the pretraining workflow. Two variants operate on a learnable
    interaction-weight matrix:

    - ``learned_weight``: ``A = σ((W + Wᵀ) / 2)`` with a single learnable
      ``W ∈ R^{n×n}`` shared per operator-type (one ``GraphLearner`` is
      instantiated per operator type, so the parameter is automatically
      type-shared by construction).
    - ``instance_conditional``: ``W`` is produced by a small hypernetwork
      conditioned on the mean-pooled initial node features. Used for the
      type-shared default.
    """

    def __init__(self, config: GraphLearnerConfig):
        super().__init__()
        self.config = config

        if config.metric_type == "weighted_cosine":
            self.weight_tensor = nn.Parameter(torch.empty(config.num_heads, config.hidden_dim))
            nn.init.xavier_uniform_(self.weight_tensor)
        elif config.metric_type == "attention":
            self.projections = nn.ModuleList(
                nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
                for _ in range(config.num_heads)
            )
        elif config.metric_type == "learned_weight":
            n = config.node_count
            self.interaction_weight = nn.Parameter(torch.empty(n, n))
            nn.init.xavier_uniform_(self.interaction_weight)
        elif config.metric_type == "instance_conditional":
            n = config.node_count
            hyper_hidden = config.hyper_hidden_dim or config.hidden_dim
            # Output the upper-triangular entries (incl. diagonal) of W; we
            # then symmetrise them into a full n×n matrix below. This is the
            # simplest valid hypernetwork output and produces an exactly
            # symmetric W by construction.
            self._tri_size = n * (n + 1) // 2
            self.hypernet = nn.Sequential(
                nn.Linear(config.hidden_dim, hyper_hidden),
                nn.ReLU(),
                nn.Linear(hyper_hidden, self._tri_size),
            )
            tri_indices = torch.triu_indices(n, n)
            # Buffers move with .to(device) and are saved in state_dict.
            self.register_buffer("_tri_rows", tri_indices[0], persistent=False)
            self.register_buffer("_tri_cols", tri_indices[1], persistent=False)
        else:
            raise ValueError(f"Unsupported graph learner metric: {config.metric_type}")

    def forward(self, node_states: torch.Tensor) -> torch.Tensor:
        squeeze_batch = False
        if node_states.dim() == 2:
            node_states = node_states.unsqueeze(0)
            squeeze_batch = True
        if node_states.dim() != 3:
            raise ValueError(
                f"Expected node_states to have shape [N, D] or [B, N, D], got {tuple(node_states.shape)}"
            )

        metric = self.config.metric_type
        if metric == "weighted_cosine":
            adjacency = self._weighted_cosine(node_states)
            adjacency = torch.relu(adjacency)
        elif metric == "attention":
            adjacency = self._attention(node_states)
            adjacency = torch.relu(adjacency)
        elif metric == "learned_weight":
            adjacency = self._learned_weight(node_states)
            # Output of σ is already in [0, 1]; do not apply ReLU.
        elif metric == "instance_conditional":
            adjacency = self._instance_conditional(node_states)
            # σ output already in [0, 1]; do not apply ReLU.
        else:  # pragma: no cover - guarded in __init__
            raise ValueError(f"Unsupported graph learner metric: {metric}")

        adjacency = self._remove_self_loops(adjacency)
        adjacency = self._apply_epsilon(adjacency)
        adjacency = self._apply_topk(adjacency)
        return adjacency.squeeze(0) if squeeze_batch else adjacency

    def _weighted_cosine(self, node_states: torch.Tensor) -> torch.Tensor:
        weighted_states = node_states.unsqueeze(0) * self.weight_tensor[:, None, None, :]
        normalized = F.normalize(weighted_states, p=2, dim=-1)
        return torch.matmul(normalized, normalized.transpose(-1, -2)).mean(dim=0)

    def _attention(self, node_states: torch.Tensor) -> torch.Tensor:
        adjacency = 0.0
        for projection in self.projections:
            projected = torch.relu(projection(node_states))
            adjacency = adjacency + torch.matmul(projected, projected.transpose(-1, -2))
        return adjacency / len(self.projections)

    def _learned_weight(self, node_states: torch.Tensor) -> torch.Tensor:
        batch_size, node_count, _ = node_states.shape
        if node_count != self.config.node_count:
            raise ValueError(
                f"learned_weight expects node_states with N={self.config.node_count}, got N={node_count}"
            )
        w = self.interaction_weight
        symmetric = 0.5 * (w + w.transpose(-1, -2))
        adjacency = torch.sigmoid(symmetric)
        # Broadcast the shared adjacency to the batch dimension.
        return adjacency.unsqueeze(0).expand(batch_size, -1, -1)

    def _instance_conditional(self, node_states: torch.Tensor) -> torch.Tensor:
        batch_size, node_count, _ = node_states.shape
        if node_count != self.config.node_count:
            raise ValueError(
                f"instance_conditional expects node_states with N={self.config.node_count}, "
                f"got N={node_count}"
            )
        # Mean-pool over nodes -> [B, D]; small hypernet -> upper triangle.
        pooled = node_states.mean(dim=1)
        tri = self.hypernet(pooled)  # [B, tri_size]

        n = self.config.node_count
        w = node_states.new_zeros(batch_size, n, n)
        rows = self._tri_rows
        cols = self._tri_cols
        w[:, rows, cols] = tri
        # The hypernetwork output fills only the upper triangle; the lower
        # triangle starts as zeros. After 0.5*(W+Wᵀ), off-diagonal entries
        # become 0.5*upper_ij (effectively half the raw hypernet output) and
        # diagonal entries stay as upper_ii. This yields an exactly symmetric
        # matrix; the half-scaling is absorbed by the hypernet's output layer.
        w_symmetric = 0.5 * (w + w.transpose(-1, -2))
        return torch.sigmoid(w_symmetric)

    def _remove_self_loops(self, adjacency: torch.Tensor) -> torch.Tensor:
        batch_size, node_count, _ = adjacency.shape
        mask = torch.eye(node_count, device=adjacency.device, dtype=torch.bool).unsqueeze(0).expand(batch_size, -1, -1)
        return adjacency.masked_fill(mask, 0.0)

    def _apply_epsilon(self, adjacency: torch.Tensor) -> torch.Tensor:
        if self.config.epsilon is None:
            return adjacency
        return torch.where(adjacency >= self.config.epsilon, adjacency, torch.zeros_like(adjacency))

    def _apply_topk(self, adjacency: torch.Tensor) -> torch.Tensor:
        if self.config.topk is None:
            return adjacency

        topk = min(self.config.topk, adjacency.size(-1))
        values, indices = torch.topk(adjacency, k=topk, dim=-1)
        masked = torch.zeros_like(adjacency)
        return masked.scatter(-1, indices, values)
