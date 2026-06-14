from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class NodeEncoderConfig:
    node_count: int
    hidden_dim: int = 64
    node_identity_dim: int = 16
    dropout: float = 0.0


class ScalarNodeEncoder(nn.Module):
    """
    Encode scalar CAG node values into dense node states.

    The encoder keeps node identity explicit with a learned node embedding so the
    model can distinguish nodes that happen to share similar scalar values.

    ``TypedNodeEncoder`` supersedes this encoder with per-type routing
    (categorical, log_scale, z_score). This class remains available for
    checkpoint compatibility.
    """

    def __init__(self, config: NodeEncoderConfig):
        super().__init__()
        self.config = config
        self.node_embedding = nn.Embedding(config.node_count, config.node_identity_dim)
        self.value_mlp = nn.Sequential(
            nn.Linear(1 + config.node_identity_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
        )

    def forward(self, node_values: torch.Tensor) -> torch.Tensor:
        if node_values.dim() == 1:
            node_values = node_values.unsqueeze(0)
        if node_values.dim() != 2:
            raise ValueError(f"Expected node_values to have shape [N] or [B, N], got {tuple(node_values.shape)}")

        batch_size, node_count = node_values.shape
        if node_count != self.config.node_count:
            raise ValueError(
                f"Node count mismatch: encoder expects {self.config.node_count}, got {node_count}"
            )

        node_ids = torch.arange(node_count, device=node_values.device).unsqueeze(0).expand(batch_size, -1)
        node_identity = self.node_embedding(node_ids)
        scalar_values = node_values.unsqueeze(-1)
        return self.value_mlp(torch.cat([scalar_values, node_identity], dim=-1))
