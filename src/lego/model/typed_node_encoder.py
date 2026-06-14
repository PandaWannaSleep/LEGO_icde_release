"""Type-specific node value encoder.

The legacy :class:`lego.model.encoders.ScalarNodeEncoder` is a single shared MLP
that processes every CAG node value uniformly after concatenating a learned
identity embedding. That conflates three semantically different value
distributions found in the OBG node schema:

1. ``categorical`` — operator-type / parent-type / child-type / strategy
   nodes. The "value" is a class label, not a scalar magnitude. A numeric MLP
   on integer codes is the wrong inductive bias; we want an embedding lookup.
2. ``log_scale`` — cardinalities (Rows / LeftRows / RightRows), page counts
   (TablePages / IndexTreePages), KB sizes, hash batch counts, recent tuple
   counters. These are heavy-tailed positive scalars; ``log1p`` makes them
   learnable for a small MLP.
3. ``z_score`` — CPU %, load averages, ratios, IO counts, selectivity, etc.
   Continuous, often multi-modal; ``(x - mean) / std`` is the natural
   pre-processing. ``mean`` / ``std`` live as :func:`torch.nn.Module.register_buffer`
   buffers, default to ``0`` / ``1`` (identity normalisation), and may be
   replaced by population statistics fitted from data.

Forward signature is **deliberately compatible** with the legacy encoder:
``forward(node_values: FloatTensor[B, N]) -> FloatTensor[B, N, hidden_dim]``.
Internally the encoder dispatches each column to its type-specific path based
on ``schema.value_types()``, then projects the per-type outputs into a shared
``R^{hidden_dim}`` space via :class:`MLP_share`.

Categorical pre-encoding
------------------------

The upstream :class:`lego.data.operator_context_extractor.OperatorContextExtractor`
emits **string** values for ``ParentOp / LeftOp / RightOp / Strategy`` (see
``_init_behavior_features`` and ``_extract_strategy``). The forward of this
encoder, however, takes a single floating-point ``node_values`` tensor —
matching the existing ``ScalarNodeEncoder`` interface. Categorical columns are
expected to be pre-encoded as integer indices in that float tensor and are cast
to ``long`` before the embedding lookup. Out-of-range values are clamped to
``[0, vocab_size - 1]`` so a stale extractor never crashes the encoder.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import torch
from torch import nn

from lego.cag.node_schema import (
    NodeSchema,
    VALID_VALUE_TYPES,
    VALUE_TYPE_CATEGORICAL,
    VALUE_TYPE_LOG_SCALE,
    VALUE_TYPE_Z_SCORE,
)


@dataclass(frozen=True)
class TypedNodeEncoderConfig:
    """Configuration for :class:`TypedNodeEncoder`.

    Attributes:
        hidden_dim: dimension of the per-type per-node hidden state produced
            by the type-specific MLPs (``MLP_cat``, ``MLP_log``, ``MLP_zscore``).
            Also the input dim to the shared projector.
        shared_hidden_dim: width of the shared projector's hidden layer.
            Defaults to ``hidden_dim``.
        output_dim: width of the final per-node embedding. Defaults to
            ``hidden_dim`` so the encoder is a drop-in replacement for
            :class:`ScalarNodeEncoder` when both are configured to the same
            ``hidden_dim``.
        embedding_dim: width of the per-node categorical embedding lookup.
            Defaults to ``hidden_dim`` so the categorical path skips a
            projection.
        categorical_vocab_size: vocabulary size shared by all categorical
            nodes. The 4 default categorical nodes (ParentOp / LeftOp /
            RightOp / Strategy) draw from a small finite set of operator
            names plus join / sort / aggregate strategy strings — 64 is a
            generous cap and any OOV indices are clamped at forward time.
        dropout: dropout probability applied inside every MLP path.
    """

    hidden_dim: int = 64
    shared_hidden_dim: int | None = None
    output_dim: int | None = None
    embedding_dim: int | None = None
    categorical_vocab_size: int = 64
    dropout: float = 0.0


def _make_mlp(in_dim: int, hidden_dim: int, out_dim: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, out_dim),
        nn.ReLU(),
    )


class TypedNodeEncoder(nn.Module):
    """Type-specific node value encoder.

    Splits the ``[B, N]`` input column-wise by per-node value type:

    * ``categorical`` → :class:`nn.Embedding` lookup → ``MLP_cat``
    * ``log_scale``   → ``log1p`` → ``MLP_log``
    * ``z_score``     → ``(x - mean) / std`` → ``MLP_zscore``

    Per-type outputs are scattered back into per-column slots, then the shared
    ``MLP_share`` projects every column into the final ``hidden_dim`` space.

    Output shape: ``[B, N, output_dim]`` (defaults to ``[B, N, hidden_dim]``).
    """

    # Class-level guard so the OOV warning is emitted at most once per process
    # (across all encoder instances). Subsequent OOV events are silently
    # clamped — see the categorical forward path in ``forward``.
    _oov_warned: bool = False

    def __init__(self, schema: NodeSchema, config: TypedNodeEncoderConfig | None = None):
        super().__init__()
        self.config = config or TypedNodeEncoderConfig()
        self.schema = schema

        node_order = schema.node_order
        if not node_order:
            raise ValueError("TypedNodeEncoder requires a non-empty NodeSchema.node_order")

        value_types = schema.value_types()
        # Detect missing / unknown tags early — the encoder must know how to
        # route every column.
        missing = [n for n in node_order if n not in value_types]
        if missing:
            raise ValueError(
                f"NodeSchema.value_types() is missing entries for: {missing[:5]}"
                + (" ..." if len(missing) > 5 else "")
            )
        unknown = {n: value_types[n] for n in node_order if value_types[n] not in VALID_VALUE_TYPES}
        if unknown:
            raise ValueError(
                f"NodeSchema.value_types() has unrecognised tags (must be one of "
                f"{sorted(VALID_VALUE_TYPES)}): {unknown}"
            )

        # Resolve config dims.
        hidden = self.config.hidden_dim
        shared_hidden = self.config.shared_hidden_dim or hidden
        output = self.config.output_dim if self.config.output_dim is not None else hidden
        embed_dim = self.config.embedding_dim if self.config.embedding_dim is not None else hidden
        dropout = self.config.dropout
        if hidden <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden}")
        if self.config.categorical_vocab_size <= 0:
            raise ValueError(
                f"categorical_vocab_size must be positive, got {self.config.categorical_vocab_size}"
            )

        # Pre-compute per-type column index tensors (registered as buffers so
        # ``.to(device)`` moves them along with the module). Stored as long
        # tensors for index_select.
        cat_idx = [i for i, n in enumerate(node_order) if value_types[n] == VALUE_TYPE_CATEGORICAL]
        log_idx = [i for i, n in enumerate(node_order) if value_types[n] == VALUE_TYPE_LOG_SCALE]
        zsc_idx = [i for i, n in enumerate(node_order) if value_types[n] == VALUE_TYPE_Z_SCORE]
        self.register_buffer(
            "categorical_indices", torch.tensor(cat_idx, dtype=torch.long), persistent=False
        )
        self.register_buffer(
            "log_indices", torch.tensor(log_idx, dtype=torch.long), persistent=False
        )
        self.register_buffer(
            "zscore_indices", torch.tensor(zsc_idx, dtype=torch.long), persistent=False
        )
        self._node_count = len(node_order)

        # Categorical path: shared embedding table for all categorical
        # operator/schema slots.
        if cat_idx:
            self.categorical_embedding = nn.Embedding(
                self.config.categorical_vocab_size, embed_dim
            )
            self.cat_mlp = _make_mlp(embed_dim, hidden, hidden, dropout)
        else:
            self.categorical_embedding = None
            self.cat_mlp = None

        # log_scale path.
        self.log_mlp = _make_mlp(1, hidden, hidden, dropout) if log_idx else None

        # z_score path. mean/std are per-node buffers so a future trainer can
        # replace them with population statistics. Defaults are mean=0, std=1
        # (identity normalisation).
        if zsc_idx:
            self.zscore_mlp = _make_mlp(1, hidden, hidden, dropout)
            self.register_buffer(
                "zscore_mean", torch.zeros(len(zsc_idx)), persistent=True
            )
            self.register_buffer(
                "zscore_std", torch.ones(len(zsc_idx)), persistent=True
            )
        else:
            self.zscore_mlp = None

        # Shared projector applied to every column post-dispatch.
        self.shared_mlp = nn.Sequential(
            nn.Linear(hidden, shared_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(shared_hidden, output),
        )

        self._hidden_dim = hidden
        self._output_dim = output

    @property
    def hidden_dim(self) -> int:
        """Final per-node embedding dimension (matches ``ScalarNodeEncoder``)."""
        return self._output_dim

    def set_zscore_stats(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        """Override the z-score buffers in-place.

        Both tensors must have shape ``[num_zscore_nodes]`` and ``std`` must
        be strictly positive (a tiny floor is applied at forward-time as well
        for safety).
        """
        if self.zscore_mlp is None:
            raise RuntimeError("Encoder has no z-score nodes; nothing to set.")
        expected = self.zscore_mean.shape
        if mean.shape != expected or std.shape != expected:
            raise ValueError(
                f"Expected mean/std with shape {tuple(expected)}, got mean={tuple(mean.shape)} std={tuple(std.shape)}"
            )
        if torch.any(std <= 0):
            raise ValueError("z-score std must be strictly positive")
        with torch.no_grad():
            self.zscore_mean.copy_(mean)
            self.zscore_std.copy_(std)

    def forward(self, node_values: torch.Tensor) -> torch.Tensor:
        if node_values.dim() == 1:
            node_values = node_values.unsqueeze(0)
        if node_values.dim() != 2:
            raise ValueError(
                f"Expected node_values to have shape [N] or [B, N], got {tuple(node_values.shape)}"
            )

        batch_size, node_count = node_values.shape
        if node_count != self._node_count:
            raise ValueError(
                f"Node count mismatch: encoder expects {self._node_count}, got {node_count}"
            )

        # Allocate the per-column hidden representation; columns get filled in
        # via ``index_copy_`` per type. Using the input's dtype (float) for
        # the buffer keeps downstream ops in the same dtype; categorical
        # columns are cast to long only for the embedding lookup itself.
        out = node_values.new_zeros(batch_size, node_count, self._hidden_dim)

        if self.categorical_embedding is not None and self.categorical_indices.numel() > 0:
            cat_cols = node_values.index_select(1, self.categorical_indices)
            # Round-then-cast so a float index like 3.0 maps to 3, and clamp
            # to vocab range so a stale upstream (or an unset default like
            # 0.0) never raises in the embedding lookup. We still detect OOV
            # before the clamp so we can warn — silently clamping every OOV
            # to ``vocab-1`` would corrupt that embedding row by sharing it
            # with garbage and silently degrade contrastive quality.
            raw_long = cat_cols.detach().round().to(torch.long)
            oov_mask = (raw_long < 0) | (raw_long >= self.config.categorical_vocab_size)
            if oov_mask.any() and not type(self)._oov_warned:
                type(self)._oov_warned = True
                warnings.warn(
                    f"TypedNodeEncoder saw {int(oov_mask.sum())} categorical "
                    f"value(s) outside [0, {self.config.categorical_vocab_size - 1}]; "
                    f"clamping to vocab range. This usually means the data pipeline "
                    f"is producing string node values that need integer pre-encoding. "
                    f"Subsequent OOV events suppressed.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            cat_long = raw_long.clamp_(0, self.config.categorical_vocab_size - 1)
            embedded = self.categorical_embedding(cat_long)  # [B, K, embed_dim]
            cat_hidden = self.cat_mlp(embedded)  # [B, K, hidden_dim]
            out = out.index_copy(1, self.categorical_indices, cat_hidden)

        if self.log_mlp is not None and self.log_indices.numel() > 0:
            log_cols = node_values.index_select(1, self.log_indices)
            # log1p is defined for log_cols >= -1; clamp negatives to 0 so a
            # malformed upstream never produces NaN.
            log_input = torch.log1p(torch.clamp(log_cols, min=0.0)).unsqueeze(-1)
            log_hidden = self.log_mlp(log_input)
            out = out.index_copy(1, self.log_indices, log_hidden)

        if self.zscore_mlp is not None and self.zscore_indices.numel() > 0:
            z_cols = node_values.index_select(1, self.zscore_indices)
            # Floor std for numerical safety regardless of set_zscore_stats.
            std_safe = torch.clamp(self.zscore_std, min=1e-6)
            z_input = ((z_cols - self.zscore_mean) / std_safe).unsqueeze(-1)
            z_hidden = self.zscore_mlp(z_input)
            out = out.index_copy(1, self.zscore_indices, z_hidden)

        return self.shared_mlp(out)
