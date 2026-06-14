"""SimCLR-style two-layer MLP for the contrastive objective.

The projection head maps the operator
embedding ``z`` to a contrast vector ``u`` used by the InfoNCE loss. Following
SimCLR (Chen et al. 2020, eqn around the InfoNCE definition), g is used **only
at training time** and is discarded at inference. Downstream adapters consume
``z`` directly, never ``u``.

This module is intentionally a thin wrapper:
  * Two linear layers with a ReLU between.
  * Optional L2-normalization of the output, since InfoNCE cosine similarity
    expects unit vectors.
  * No softmax or temperature inside; those live in the loss.

The head's parameters are saved during training (so warm-start fine-tunes can
load them) but inference ignores them when constructing an embedding model.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class ProjectionHeadConfig:
    """Configuration for the SimCLR-style projector.

    ``input_dim`` should match the operator embedding dimension
    (``OperatorEncoderConfig.output_dim``). ``hidden_dim`` and ``output_dim``
    are independent — SimCLR commonly uses ``hidden_dim == input_dim`` and
    ``output_dim`` smaller (e.g. 128). For LEGO the contrastive batch is
    typically modest, so we keep ``output_dim == input_dim`` by default.
    """
    input_dim: int = 64
    hidden_dim: int = 64
    output_dim: int = 64
    dropout: float = 0.0
    l2_normalize: bool = True


class ProjectionHead(nn.Module):
    """Two-layer MLP projector ``z -> u``.

    Args:
        config: head sizing and behavior.

    Forward signature: ``forward(z) -> u`` where ``z`` has shape
    ``[..., input_dim]`` and ``u`` has shape ``[..., output_dim]``. When
    ``l2_normalize`` is True (default), ``u`` lies on the unit sphere so
    InfoNCE can use a plain dot product as cosine similarity.
    """

    def __init__(self, config: ProjectionHeadConfig):
        super().__init__()
        self.config = config
        self.layers = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.output_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        u = self.layers(z)
        if self.config.l2_normalize:
            u = F.normalize(u, dim=-1, eps=1e-8)
        return u
