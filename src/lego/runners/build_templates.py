from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from lego.data.cag_template_builder import (
    MutualInformationCAGTemplateBuilder,
)
from lego.data.operator_context_extractor import OperatorContextExtractor
from lego.data.plan_loader import iter_plan_records_with_source
from lego.data.schema_stats import load_schema_stats
from lego.cag.io import save_cag_template


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build operator-type CAG templates")
    parser.add_argument("--plan-file", nargs="+", required=True, help="One or more JSONL plan files")
    parser.add_argument("--db-name", required=True, help="Database name, e.g. imdb")
    parser.add_argument("--legacy-root", help="Path to legacy artifact root")
    parser.add_argument("--schema-cache", help="Explicit schema stats cache artifact (.pickle or .json)")
    parser.add_argument("--output-dir", required=True, help="Directory to save generated templates")
    parser.add_argument("--limit", type=int, default=0, help="Maximum number of plans to load; 0 means all")
    parser.add_argument("--skip-invalid-plans", action="store_true", help="Skip invalid plan records instead of failing")
    parser.add_argument("--mi-threshold", type=float, default=0.4, help="MI threshold for edge creation")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    schema_stats = load_schema_stats(
        db_name=args.db_name,
        schema_cache=args.schema_cache,
        legacy_root=args.legacy_root,
    )
    extractor = OperatorContextExtractor(schema_stats=schema_stats)
    grouped_contexts: dict[str, list] = defaultdict(list)
    skipped_by_file: dict[str, int] = defaultdict(int)

    def on_skip(path: Path, _line_number: int, _reason: str) -> None:
        skipped_by_file[str(path)] += 1

    for plan_index, (source_path, plan_record) in enumerate(
        iter_plan_records_with_source(
            args.plan_file,
            strict=not args.skip_invalid_plans,
            on_skip=on_skip if args.skip_invalid_plans else None,
        ),
        start=1,
    ):
        for context in extractor.extract_plan(plan_record, source_path=str(source_path)):
            grouped_contexts[context.operator_type].append(context)
        if args.limit and plan_index >= args.limit:
            break

    builder = MutualInformationCAGTemplateBuilder(threshold=args.mi_threshold)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    skipped_total = sum(skipped_by_file.values())
    print(f"template_build_skipped={skipped_total}")

    for operator_type, contexts in sorted(grouped_contexts.items()):
        template = builder.build(operator_type=operator_type, contexts=contexts)
        target_path = output_dir / f"{operator_type.replace(' ', '_')}.pkl"
        save_cag_template(template, target_path)
        print(
            f"saved_template operator={operator_type} num_contexts={len(contexts)} "
            f"node_count={len(template.node_order)} path={target_path}"
        )


if __name__ == "__main__":
    main()
