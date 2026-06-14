from __future__ import annotations

import numpy as np

from lego.data.operator_context import OperatorContext
from .instance import OperatorCAG
from .template import CAGTemplate


class OperatorCAGBuilder:
    """Instantiate an operator-specific CAG from a context and a template."""

    def build(self, context: OperatorContext, template: CAGTemplate, strict: bool = True) -> OperatorCAG:
        if context.operator_type != template.operator_type:
            raise ValueError(
                f"Operator mismatch: context={context.operator_type!r}, template={template.operator_type!r}"
            )

        raw_values = context.to_ordered_feature_values(
            node_order=template.node_order,
            categorical_encoders=template.categorical_encoders,
            strict=strict,
        )
        normalization_stats = template.metadata.get("normalization_stats", {})
        normalized_values: list[float] = []
        for node_name, value in zip(template.node_order, raw_values):
            stats = normalization_stats.get(node_name)
            if stats is None:
                normalized_values.append(float(value))
                continue
            mean = float(stats.get("mean", 0.0))
            std = max(float(stats.get("std", 1.0)), 1e-6)
            normalized_values.append((float(value) - mean) / std)

        node_values = np.array(normalized_values, dtype=np.float32)

        initial_adj = template.initial_adjacency.astype(np.float32, copy=True)
        return OperatorCAG(
            context=context,
            template=template,
            node_values=node_values,
            initial_adjacency=initial_adj,
            current_adjacency=initial_adj.copy(),
        )

