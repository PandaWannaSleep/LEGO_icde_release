from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from lego.artifact.operator_catalog import (
    DEFAULT_CHILD_OPERATOR,
    DEFAULT_PARENT_OPERATOR,
    DEFAULT_STRATEGY,
)
from lego.cag.io import save_cag_template
from lego.cag.node_schema import default_node_schema
from lego.cag.template import CAGTemplate


DEFAULT_OPERATORS = (
    "Seq Scan",
    "Index Scan",
    "Index Only Scan",
    "Nested Loop",
    "Hash Join",
    "Hash",
    "Aggregate",
    "Gather",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build fully connected CAG templates")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--operator-type", action="append", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    operators = tuple(args.operator_type) if args.operator_type else DEFAULT_OPERATORS
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    schema = default_node_schema()
    n_nodes = schema.size
    adjacency = np.ones((n_nodes, n_nodes), dtype=np.float32) - np.eye(n_nodes, dtype=np.float32)
    encoders = {
        "ParentOp": {DEFAULT_PARENT_OPERATOR: 0},
        "LeftOp": {DEFAULT_CHILD_OPERATOR: 0},
        "RightOp": {DEFAULT_CHILD_OPERATOR: 0},
        "Strategy": {DEFAULT_STRATEGY: 0},
    }

    for operator_type in operators:
        template = CAGTemplate(
            operator_type=operator_type,
            node_schema=schema,
            node_order=schema.node_order,
            initial_adjacency=adjacency.copy(),
            build_method="fully_connected",
            categorical_encoders=encoders,
            metadata={"normalization_stats": {}},
        )
        output_path = output_dir / f"{operator_type.replace(' ', '_')}.pkl"
        save_cag_template(template, output_path)
        print(
            f"saved_template operator={operator_type} "
            f"node_count={n_nodes} edge_count={int((adjacency != 0).sum())} "
            f"path={output_path}"
        )


if __name__ == "__main__":
    main()
