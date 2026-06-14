from __future__ import annotations

from collections import defaultdict

import numpy as np
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.metrics import mutual_info_score

from lego.artifact.operator_catalog import DEFAULT_CHILD_OPERATOR, DEFAULT_PARENT_OPERATOR, DEFAULT_STRATEGY
from lego.cag.node_schema import NodeSchema, default_node_schema
from lego.cag.template import CAGTemplate
from .operator_context import OperatorContext


def _build_numeric_normalization_stats(
    contexts: list[OperatorContext],
    schema: NodeSchema,
    encoders: dict[str, dict[str, int]],
) -> dict[str, dict[str, float]]:
    if not contexts:
        return {}

    features = np.array(
        [
            context.to_ordered_feature_values(
                schema.node_order,
                categorical_encoders=encoders,
                strict=False,
            )
            for context in contexts
        ],
        dtype=np.float32,
    )
    stats: dict[str, dict[str, float]] = {}
    for index, node_name in enumerate(schema.node_order):
        if node_name in schema.categorical_nodes:
            continue
        column = features[:, index].astype(np.float64)
        mean = float(np.mean(column))
        std = float(np.std(column))
        stats[node_name] = {
            "mean": mean,
            "std": max(std, 1e-6),
        }
    return stats


class MutualInformationCAGTemplateBuilder:
    def __init__(self, threshold: float = 0.4):
        self.threshold = threshold

    def build(
        self,
        operator_type: str,
        contexts: list[OperatorContext],
        node_schema: NodeSchema | None = None,
    ) -> CAGTemplate:
        schema = node_schema or default_node_schema()
        encoders = _build_categorical_encoders(contexts)
        features = np.array(
            [
                context.to_ordered_feature_values(
                    schema.node_order,
                    categorical_encoders=encoders,
                    strict=False,
                )
                for context in contexts
            ],
            dtype=np.float32,
        )
        if features.size == 0:
            raise ValueError("Cannot build MI template from an empty context list")

        transposed = features.T
        feature_names = list(schema.node_order)
        adjacency = np.zeros((len(feature_names), len(feature_names)), dtype=np.float32)

        for i in range(len(feature_names)):
            for j in range(i + 1, len(feature_names)):
                mi = _compute_pairwise_mi(
                    transposed[i],
                    transposed[j],
                    feature_names[i] in schema.categorical_nodes,
                    feature_names[j] in schema.categorical_nodes,
                )
                if mi > self.threshold:
                    adjacency[i, j] = 1.0
                    adjacency[j, i] = 1.0

        normalization_stats = _build_numeric_normalization_stats(contexts, schema, encoders)
        return CAGTemplate(
            operator_type=operator_type,
            node_schema=schema,
            node_order=schema.node_order,
            initial_adjacency=adjacency,
            build_method="mutual_information",
            categorical_encoders=encoders,
            metadata={
                "mi_threshold": self.threshold,
                "num_contexts": len(contexts),
                "normalization_stats": normalization_stats,
            },
        )


def _build_categorical_encoders(contexts: list[OperatorContext]) -> dict[str, dict[str, int]]:
    categories: dict[str, set[str]] = defaultdict(set)
    categories["ParentOp"].add(DEFAULT_PARENT_OPERATOR)
    categories["LeftOp"].add(DEFAULT_CHILD_OPERATOR)
    categories["RightOp"].add(DEFAULT_CHILD_OPERATOR)
    categories["Strategy"].add(DEFAULT_STRATEGY)

    for context in contexts:
        feature_dict = context.feature_dict()
        for key in ("ParentOp", "LeftOp", "RightOp", "Strategy"):
            value = feature_dict.get(key)
            if isinstance(value, str):
                categories[key].add(value)

    return {
        key: {value: idx for idx, value in enumerate(sorted(values))}
        for key, values in categories.items()
    }


def _compute_pairwise_mi(
    left: np.ndarray,
    right: np.ndarray,
    left_is_categorical: bool,
    right_is_categorical: bool,
) -> float:
    n_samples = len(left)
    if n_samples < 2:
        return 0.0

    if np.all(left == left[0]) or np.all(right == right[0]):
        return 0.0

    n_neighbors = max(1, min(3, n_samples - 1))

    if left_is_categorical and right_is_categorical:
        return float(mutual_info_score(left, right))

    if not left_is_categorical and not right_is_categorical:
        try:
            return float(
                mutual_info_regression(
                    left.reshape(-1, 1),
                    right,
                    random_state=42,
                    n_neighbors=n_neighbors,
                )[0]
            )
        except ValueError:
            return 0.0

    if left_is_categorical:
        labels, counts = np.unique(left, return_counts=True)
        if len(labels) < 2 or np.max(counts) < 2:
            return 0.0
        try:
            return float(
                mutual_info_classif(
                    right.reshape(-1, 1),
                    left,
                    discrete_features=False,
                    random_state=42,
                    n_neighbors=n_neighbors,
                )[0]
            )
        except ValueError:
            return 0.0

    labels, counts = np.unique(right, return_counts=True)
    if len(labels) < 2 or np.max(counts) < 2:
        return 0.0
    try:
        return float(
            mutual_info_classif(
                left.reshape(-1, 1),
                right,
                discrete_features=False,
                random_state=42,
                n_neighbors=n_neighbors,
            )[0]
        )
    except ValueError:
        return 0.0
