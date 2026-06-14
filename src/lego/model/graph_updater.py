from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class GraphUpdateConfig:
    graph_skip_conn: float = 0.5
    update_adj_ratio: float = 0.2
    include_self: bool = False
    normalize_rows: bool = True


class GraphUpdater:
    """
    Apply the two update stages that already exist in the legacy artifact:

    1. Blend the learned graph with the initial CAG template.
    2. During iterative refinement, blend each new graph with the first
       refined graph to keep updates stable.
    """

    def __init__(self, config: GraphUpdateConfig):
        self.config = config

    def build_first_graph(self, initial_adjacency: torch.Tensor, learned_adjacency: torch.Tensor) -> torch.Tensor:
        blended = self.config.graph_skip_conn * initial_adjacency + (1.0 - self.config.graph_skip_conn) * learned_adjacency
        return self._finalize(blended)

    def build_iterative_graph(
        self,
        initial_adjacency: torch.Tensor,
        learned_adjacency: torch.Tensor,
        first_refined_adjacency: torch.Tensor,
    ) -> torch.Tensor:
        blended = self.config.graph_skip_conn * initial_adjacency + (1.0 - self.config.graph_skip_conn) * learned_adjacency
        updated = self.config.update_adj_ratio * first_refined_adjacency + (1.0 - self.config.update_adj_ratio) * blended
        return self._finalize(updated)

    def adjacency_delta(self, previous_adjacency: torch.Tensor, next_adjacency: torch.Tensor) -> float:
        return torch.mean(torch.abs(next_adjacency - previous_adjacency)).item()

    def _finalize(self, adjacency: torch.Tensor) -> torch.Tensor:
        adjacency = torch.clamp(adjacency, min=0.0)

        if self.config.include_self:
            adjacency = self._add_self_loops(adjacency)

        if self.config.normalize_rows:
            adjacency = self._normalize_rows(adjacency)

        return adjacency

    def _add_self_loops(self, adjacency: torch.Tensor) -> torch.Tensor:
        node_count = adjacency.size(-1)
        eye = torch.eye(node_count, device=adjacency.device, dtype=adjacency.dtype)
        if adjacency.dim() == 3:
            eye = eye.unsqueeze(0).expand(adjacency.size(0), -1, -1)
        return adjacency + eye

    def _normalize_rows(self, adjacency: torch.Tensor) -> torch.Tensor:
        row_sum = adjacency.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        return adjacency / row_sum
