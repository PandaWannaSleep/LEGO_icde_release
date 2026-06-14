from __future__ import annotations

import argparse
from pathlib import Path

from lego.inference.manifest_writer import build_manifest


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


def _slug(operator_type: str) -> str:
    return "".join(
        ch for ch in operator_type.lower().replace(" ", "_")
        if ch.isalnum() or ch == "_"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a LEGO checkpoint manifest")
    parser.add_argument("--task", required=True, choices=("runtime_cost", "startup_cost"))
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--tag", default="", help="Optional suffix used in checkpoint directory names")
    parser.add_argument(
        "--checkpoint-prefix",
        default="synthetic_pak_lw",
        help="Checkpoint directory prefix before the operator slug",
    )
    parser.add_argument("--operator-type", action="append", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    operators = tuple(args.operator_type) if args.operator_type else DEFAULT_OPERATORS
    checkpoint_root = Path(args.checkpoint_root)
    mapping = {}
    for operator_type in operators:
        suffix = f"_{args.tag}" if args.tag else ""
        checkpoint_dir = checkpoint_root / f"{args.checkpoint_prefix}_{_slug(operator_type)}{suffix}"
        mapping[operator_type] = checkpoint_dir
    build_manifest(args.task, mapping, args.output_path)
    print(f"manifest_path={args.output_path}")


if __name__ == "__main__":
    main()
