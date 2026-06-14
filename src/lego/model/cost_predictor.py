"""CostHead maps operator embedding ``z`` to scalar cost.

This head is used as a supervised branch during pretraining and can also be
used at inference for cost-only deployment.

History: this module used to be ``CostPredictor`` and bundled message
passing + pool + cost head into one class. The propagate/pool concerns
moved to ``operator_encoder.py``; this file now hosts the lightweight
cost-prediction MLP only. ``CostPredictor`` is retained as a deprecated
alias so code reading legacy checkpoints keeps working until the loader
finishes the split (see ``inference/checkpoint_loader.py``).
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class CostHeadConfig:
    """Lightweight cost-regression head configuration.

    ``input_dim`` should match ``OperatorEncoderConfig.output_dim`` (i.e.
    ``hidden_dim`` for non-concat pooling, ``hidden_dim * 3`` for concat).
    """
    input_dim: int = 64
    hidden_dim: int = 64
    dropout: float = 0.0


class CostHead(nn.Module):
    """Two-layer MLP mapping operator embedding ``z`` to a positive scalar cost."""

    def __init__(self, config: CostHeadConfig):
        super().__init__()
        self.config = config
        self.head = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, 1),
        )
        self.output_activation = nn.Softplus()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.output_activation(self.head(z)).squeeze(-1)


# ============================================================================
# Backward-compatible alias for legacy callers.
#
# Old code paths (training scripts, checkpoint loaders) referred to
# ``CostPredictor`` and ``CostPredictorConfig`` and treated the class as a
# "GNN + pool + cost head" unit. Path B splits that into ``OperatorEncoder``
# (in operator_encoder.py) and ``CostHead`` (above). To minimise blast radius
# on the migration commit, we keep these symbols importable but emit a
# DeprecationWarning at construction time so callers can find them.
# ============================================================================


@dataclass(frozen=True)
class CostPredictorConfig:
    """Deprecated. Use ``OperatorEncoderConfig`` + ``CostHeadConfig`` instead."""
    hidden_dim: int = 64
    num_message_passing_layers: int = 2
    pool_type: str = "mean"
    dropout: float = 0.0


class CostPredictor(nn.Module):
    """Deprecated wrapper that combines OperatorEncoder and CostHead."""

    def __init__(self, config: CostPredictorConfig):
        super().__init__()
        warnings.warn(
            "CostPredictor is deprecated; use OperatorEncoder + CostHead "
            "(operator_encoder.py + cost_predictor.CostHead).",
            DeprecationWarning,
            stacklevel=2,
        )
        from .operator_encoder import OperatorEncoder, OperatorEncoderConfig

        self.config = config
        self._encoder = OperatorEncoder(
            OperatorEncoderConfig(
                hidden_dim=config.hidden_dim,
                num_message_passing_layers=config.num_message_passing_layers,
                pool_type=config.pool_type,
                dropout=config.dropout,
            )
        )
        self._head = CostHead(
            CostHeadConfig(
                input_dim=self._encoder.config.output_dim,
                hidden_dim=config.hidden_dim,
                dropout=config.dropout,
            )
        )

    @property
    def layers(self):  # noqa: D401 — preserve old attribute access
        return self._encoder.layers

    @property
    def head(self):
        return self._head.head

    @property
    def output_activation(self):
        return self._head.output_activation

    def propagate(self, node_states: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        return self._encoder.propagate(node_states, adjacency)

    def pool_states(self, node_states: torch.Tensor) -> torch.Tensor:
        return self._encoder.readout(node_states)

    def predict_from_states(self, node_states: torch.Tensor) -> torch.Tensor:
        z = self._encoder.readout(node_states)
        return self._head(z)

    def forward(
        self,
        node_states: torch.Tensor,
        adjacency: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        z, propagated = self._encoder.encode(node_states, adjacency)
        prediction = self._head(z)
        return prediction, propagated
