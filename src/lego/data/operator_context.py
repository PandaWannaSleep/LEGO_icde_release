from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Union


FeatureValue = Union[float, int, str]


@dataclass(frozen=True)
class OperatorLabels:
    """Raw operator-level labels kept before task-specific target transforms."""

    actual_startup_time: float = 0.0
    actual_total_time: float = 0.0
    optimizer_startup_cost: float = 0.0
    optimizer_total_cost: float = 0.0


@dataclass(frozen=True)
class OperatorMetadata:
    """Non-feature metadata attached to an operator sample."""

    operator_type: str
    parent_operator: str
    relation_name: str | None = None
    query_text: str | None = None
    source_path: str | None = None
    raw_plan: dict[str, Any] | None = None
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class OperatorContext:
    """A first-class representation of a single operator execution context."""

    operator_type: str
    behavior_features: dict[str, FeatureValue]
    resource_features: dict[str, FeatureValue]
    table_heat_features: dict[str, FeatureValue]
    labels: OperatorLabels
    metadata: OperatorMetadata

    def feature_dict(self) -> dict[str, FeatureValue]:
        merged: dict[str, FeatureValue] = {}
        for group in (
            self.behavior_features,
            self.resource_features,
            self.table_heat_features,
        ):
            overlap = set(merged).intersection(group)
            if overlap:
                raise ValueError(f"Duplicate feature names across groups: {sorted(overlap)}")
            merged.update(group)
        return merged

    def to_ordered_feature_values(
        self,
        node_order: list[str] | tuple[str, ...],
        categorical_encoders: dict[str, dict[str, int]] | None = None,
        strict: bool = True,
    ) -> list[float]:
        """Encode context features into a node-value vector aligned with node_order."""

        features = self.feature_dict()
        encoded_values: list[float] = []
        categorical_encoders = categorical_encoders or {}

        for node_name in node_order:
            if node_name not in features:
                if strict:
                    raise KeyError(f"Feature {node_name!r} missing from operator context")
                encoded_values.append(0.0)
                continue

            value = features[node_name]
            if isinstance(value, str):
                encoder = categorical_encoders.get(node_name)
                if encoder is None:
                    if strict:
                        raise KeyError(f"No categorical encoder registered for {node_name!r}")
                    encoded_values.append(0.0)
                else:
                    encoded_values.append(float(encoder.get(value, 0)))
            else:
                encoded_values.append(float(value))

        return encoded_values

