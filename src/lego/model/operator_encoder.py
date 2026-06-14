"""OperatorEncoder turns a refined OBG into operator embedding ``z``.

Architecture:

    node_states → [pre_mp MLP] → K message-passing layers → readout/pool → [post_mp MLP] → z

Both the cost head and the contrastive projection head sit *outside* this module;
they consume ``z`` and are discarded at inference.

Pre-MP and post-MP layers are optional (default disabled) for full backward
compatibility with existing checkpoints.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


# --------------------------------------------------------------------------- #
# FlexibleMLP implementation used by the operator encoder.
# --------------------------------------------------------------------------- #

class FlexibleMLP(nn.Module):
    """Configurable multi-layer MLP with optional activation.

    The last layer has no activation regardless of ``has_act``.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 1,
        has_act: bool = True,
        has_bias: bool = True,
    ):
        super().__init__()
        layers = []
        prev_dim = in_dim
        for i in range(num_layers - 1):
            layers.append(nn.Linear(prev_dim, hidden_dim, bias=has_bias))
            if has_act:
                layers.append(nn.ReLU())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, out_dim, bias=has_bias))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class OperatorEncoderConfig:
    encoder_mode: str = "graph"
    hidden_dim: int = 64
    num_message_passing_layers: int = 2
    pool_type: str = "mean"  # mean | sum | max | concat | attention
    dropout: float = 0.0

    # --- pre-MP (feature transformation before message passing) ---
    pre_mp_layers: int = 0           # 0 = skip pre_mp
    pre_mp_hidden_dim: int = 64

    # --- post-MP (projection after pooling → z) ---
    post_mp_layers: int = 0          # 0 = skip post_mp (backward-compat)
    post_mp_hidden_dim: int = 64
    post_mp_output_dim: int = 0      # 0 = auto (hidden_dim or hidden_dim*3 for concat)

    # Retained for old checkpoint config compatibility. The release path uses
    # graph mode.
    flat_input_dim: int = 0

    # Retained for old checkpoint config compatibility.
    ft_num_layers: int = 2
    ft_num_heads: int = 4
    ft_dropout: float = 0.1

    @property
    def output_dim(self) -> int:
        """Dimensionality of the produced ``z`` vector."""
        if self.post_mp_output_dim > 0:
            return self.post_mp_output_dim
        if self.post_mp_layers > 0 and self.pool_type != "concat":
            return self.post_mp_hidden_dim
        return self.hidden_dim * 3 if self.pool_type == "concat" else self.hidden_dim


# --------------------------------------------------------------------------- #
# Message-passing layer
# --------------------------------------------------------------------------- #

class _MessagePassingLayer(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float):
        super().__init__()
        self.self_linear = nn.Linear(hidden_dim, hidden_dim)
        self.neighbor_linear = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, node_states: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        messages = torch.matmul(adjacency, self.neighbor_linear(node_states))
        updated = self.self_linear(node_states) + messages
        return torch.relu(self.dropout(updated))


# --------------------------------------------------------------------------- #
# Operator Encoder
# --------------------------------------------------------------------------- #

class OperatorEncoder(nn.Module):
    """Backbone that produces ``z`` from refined OBG inputs.

    Pipeline::

        node_states
          │
          ├─ [pre_mp]  ── optional MLP (pre_mp_layers > 0)
          │
          ├─ propagate ── K message-passing layers
          │
          ├─ readout   ── pool to graph-level vector
          │
          ├─ [post_mp] ── optional MLP (post_mp_layers > 0) → z
          │
          └─ z

    When both pre_mp and post_mp are disabled, this remains compatible with
    older graph-mode checkpoints.
    """

    def __init__(self, config: OperatorEncoderConfig):
        super().__init__()
        self.config = config

        if config.encoder_mode != "graph":
            raise ValueError(f"Unsupported OperatorEncoder encoder_mode={config.encoder_mode!r}")

        # --- pre-MP ---
        mp_input_dim = config.hidden_dim
        if config.pre_mp_layers > 0:
            self.pre_mp = FlexibleMLP(
                in_dim=config.hidden_dim,
                out_dim=config.pre_mp_hidden_dim,
                hidden_dim=config.pre_mp_hidden_dim,
                num_layers=config.pre_mp_layers,
                has_act=True,
            )
            mp_input_dim = config.pre_mp_hidden_dim
        else:
            self.pre_mp = None

        # --- message-passing layers ---
        self.layers = nn.ModuleList(
            _MessagePassingLayer(mp_input_dim if i == 0 else mp_input_dim, config.dropout)
            for i in range(config.num_message_passing_layers)
        )
        self._mp_dim = mp_input_dim

        # --- post-MP ---
        if config.post_mp_layers > 0:
            pool_dim = mp_input_dim * 3 if config.pool_type == "concat" else mp_input_dim
            out_dim = config.post_mp_output_dim if config.post_mp_output_dim > 0 else config.post_mp_hidden_dim
            self.post_mp = FlexibleMLP(
                in_dim=pool_dim,
                out_dim=out_dim,
                hidden_dim=config.post_mp_hidden_dim,
                num_layers=config.post_mp_layers,
                has_act=True,
            )
        else:
            self.post_mp = None

    def propagate(self, node_states: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        if self.pre_mp is not None:
            node_states = self.pre_mp(node_states)
        hidden = node_states
        for layer in self.layers:
            hidden = layer(hidden, adjacency)
        return hidden

    def readout(self, node_states: torch.Tensor) -> torch.Tensor:
        """Pool node states to a graph-level vector."""
        pool_type = self.config.pool_type.lower()

        if pool_type == "mean":
            return node_states.mean(dim=1)
        if pool_type in {"sum", "add"}:
            return node_states.sum(dim=1)
        if pool_type == "max":
            return node_states.max(dim=1).values
        if pool_type == "attention":
            return self._attention_pool(node_states)
        if pool_type == "concat":
            mean_pool = node_states.mean(dim=1)
            max_pool = node_states.max(dim=1).values
            sum_pool = node_states.sum(dim=1)
            return torch.cat([mean_pool, max_pool, sum_pool], dim=-1)

        raise ValueError(f"Unsupported pool type: {self.config.pool_type}")

    def _attention_pool(self, x: torch.Tensor) -> torch.Tensor:
        """Attention pooling over nodes (ported from HomoParamGCN)."""
        if not hasattr(self, "_attn_linear"):
            self._attn_linear = nn.Linear(x.size(-1), 1).to(x.device)
        scores = self._attn_linear(x)          # [B, N, 1]
        weights = torch.softmax(scores, dim=1)  # [B, N, 1]
        return (x * weights).sum(dim=1)         # [B, D]

    def encode(
        self,
        node_states: torch.Tensor,
        adjacency: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        propagated = self.propagate(node_states, adjacency)
        pooled = self.readout(propagated)
        if self.post_mp is not None:
            pooled = self.post_mp(pooled)
        return pooled, propagated

    def forward(
        self,
        node_states: torch.Tensor,
        adjacency: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encode(node_states, adjacency)
