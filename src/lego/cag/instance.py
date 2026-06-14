from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from lego.data.operator_context import OperatorContext
from .template import CAGTemplate


@dataclass(frozen=True)
class OperatorCAG:
    context: OperatorContext
    template: CAGTemplate
    node_values: np.ndarray
    initial_adjacency: np.ndarray
    current_adjacency: np.ndarray

    def __post_init__(self) -> None:
        n = len(self.template.node_order)
        if self.node_values.shape != (n,):
            raise ValueError(
                f"Node value vector shape {self.node_values.shape} does not match template size {n}"
            )
        for name, adj in {
            "initial_adjacency": self.initial_adjacency,
            "current_adjacency": self.current_adjacency,
        }.items():
            if adj.shape != (n, n):
                raise ValueError(f"{name} shape {adj.shape} does not match template size {(n, n)}")

    def with_refined_adjacency(self, refined_adjacency: np.ndarray) -> "OperatorCAG":
        return replace(self, current_adjacency=refined_adjacency)


@dataclass(frozen=True)
class BatchOperatorCAG:
    """Batch of same-operator CAG instances.

    All rows in a batch share the same operator template, so their node-value
    vectors and adjacency matrices have compatible shapes and can be stacked
    for a single encoder forward pass.
    """

    operator_type: str
    contexts: tuple[OperatorContext, ...]
    template: CAGTemplate
    node_values: np.ndarray
    initial_adjacency: np.ndarray
    current_adjacency: np.ndarray

    def __post_init__(self) -> None:
        n = len(self.template.node_order)
        b = len(self.contexts)
        if self.node_values.shape != (b, n):
            raise ValueError(
                f"Batch node value shape {self.node_values.shape} does not match {(b, n)}"
            )
        for name, adj in {
            "initial_adjacency": self.initial_adjacency,
            "current_adjacency": self.current_adjacency,
        }.items():
            if adj.shape != (b, n, n):
                raise ValueError(f"{name} shape {adj.shape} does not match {(b, n, n)}")
