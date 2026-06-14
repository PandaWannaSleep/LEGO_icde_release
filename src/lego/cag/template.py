from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .node_schema import NodeSchema


@dataclass(frozen=True)
class CAGTemplate:
    operator_type: str
    node_schema: NodeSchema
    node_order: tuple[str, ...]
    initial_adjacency: np.ndarray
    build_method: str
    categorical_encoders: dict[str, dict[str, int]] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        n = len(self.node_order)
        if self.initial_adjacency.shape != (n, n):
            raise ValueError(
                f"Adjacency shape {self.initial_adjacency.shape} does not match node order size {n}"
            )

