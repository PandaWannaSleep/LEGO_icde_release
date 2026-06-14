from __future__ import annotations

import argparse

from lego.data.pair_assembler import (
    PAIRING_POLICIES,
    RECORD_CONCURRENCY_LEVEL,
    assemble_pairs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Assemble paired and single operator-view datasets from raw JSONL files")
    parser.add_argument("--s1-file", nargs="*", default=[])
    parser.add_argument("--s4-file", nargs="*", default=[])
    parser.add_argument("--s8-file", nargs="*", default=[])
    parser.add_argument(
        "--raw-file",
        nargs="*",
        default=[],
        help="Raw JSONL files treated as one irregular observation pool.",
    )
    parser.add_argument(
        "--irregular-file",
        nargs="*",
        default=[],
        help="Alias for --raw-file; intended for data/<benchmark>/raw/*_irregular_*.jsonl.",
    )
    parser.add_argument(
        "--record-level-file",
        nargs="*",
        default=[],
        help="Compatibility mode: use each record's concurrency_level field as the level label.",
    )
    parser.add_argument("--pairs-path", required=True)
    parser.add_argument("--singles-path", required=True)
    parser.add_argument("--data-quality-path", required=True)
    parser.add_argument(
        "--pairing-policy",
        choices=PAIRING_POLICIES,
        default="plan_position",
        help="Positive-pair matching policy; the default preserves the existing same-plan-position protocol.",
    )
    parser.add_argument(
        "--count-only",
        action="store_true",
        help="Compute data-quality and pair-count statistics without writing full pair/single view records.",
    )
    parser.add_argument("--skip-invalid-plans", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_files_by_level = {
        level: files
        for level, files in {
            "s1": args.s1_file,
            "s4": args.s4_file,
            "s8": args.s8_file,
        }.items()
        if files
    }
    irregular_files = [*args.raw_file, *args.irregular_file]
    if irregular_files:
        raw_files_by_level.setdefault("irregular", []).extend(irregular_files)
    record_level_files = [*args.record_level_file]
    if record_level_files:
        raw_files_by_level.setdefault(RECORD_CONCURRENCY_LEVEL, []).extend(record_level_files)
    if not raw_files_by_level:
        raise ValueError("At least one of --raw-file, --irregular-file, --s1-file, --s4-file, or --s8-file is required")
    result = assemble_pairs(
        raw_files_by_level=raw_files_by_level,
        pairs_path=args.pairs_path,
        singles_path=args.singles_path,
        data_quality_path=args.data_quality_path,
        strict=not args.skip_invalid_plans,
        pairing_policy=args.pairing_policy,
        write_records=not args.count_only,
    )
    print(f"pairs_path={result.pairs_path}")
    print(f"singles_path={result.singles_path}")
    print(f"data_quality_path={result.data_quality_path}")


if __name__ == "__main__":
    main()
