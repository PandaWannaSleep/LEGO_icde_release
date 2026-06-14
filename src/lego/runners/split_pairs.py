from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split singles.jsonl into train/valid/test by template_id")
    parser.add_argument("--singles-path", required=True)
    parser.add_argument("--train-path", required=True)
    parser.add_argument("--valid-path", required=True)
    parser.add_argument("--test-path", required=True)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--valid-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    singles_path = Path(args.singles_path)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with singles_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            grouped[str(record.get("template_id", "unknown"))].append(record)

    template_ids = sorted(grouped.keys())
    rng = random.Random(args.seed)
    rng.shuffle(template_ids)

    total = len(template_ids)
    train_cut = int(total * args.train_ratio)
    valid_cut = train_cut + int(total * args.valid_ratio)
    train_ids = set(template_ids[:train_cut])
    valid_ids = set(template_ids[train_cut:valid_cut])
    test_ids = set(template_ids[valid_cut:])

    train_records: list[dict[str, Any]] = []
    valid_records: list[dict[str, Any]] = []
    test_records: list[dict[str, Any]] = []
    for template_id, records in grouped.items():
        if template_id in train_ids:
            train_records.extend(records)
        elif template_id in valid_ids:
            valid_records.extend(records)
        else:
            test_records.extend(records)

    train_path = Path(args.train_path)
    valid_path = Path(args.valid_path)
    test_path = Path(args.test_path)
    _write_jsonl(train_path, train_records)
    _write_jsonl(valid_path, valid_records)
    _write_jsonl(test_path, test_records)

    print(f"train_path={train_path} rows={len(train_records)} templates={len(train_ids)}")
    print(f"valid_path={valid_path} rows={len(valid_records)} templates={len(valid_ids)}")
    print(f"test_path={test_path} rows={len(test_records)} templates={len(test_ids)}")


if __name__ == "__main__":
    main()
