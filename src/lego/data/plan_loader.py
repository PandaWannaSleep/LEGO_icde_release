from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Callable

from .plan_preprocessor import normalize_plan_record

SkipCallback = Callable[[Path, int, str], None]


def iter_plan_records_with_source(
    paths: list[str] | list[Path],
    *,
    strict: bool = True,
    on_skip: SkipCallback | None = None,
) -> Iterator[tuple[Path, dict[str, Any]]]:
    for path in paths:
        plan_path = Path(path)
        with plan_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    raw_record = json.loads(line)
                except json.JSONDecodeError as exc:
                    if strict:
                        raise ValueError(f"{plan_path}:{line_number}: JSON decode error: {exc.msg}") from exc
                    if on_skip is not None:
                        on_skip(plan_path, line_number, f"JSON decode error: {exc.msg}")
                    continue
                try:
                    normalized_record = normalize_plan_record(raw_record)
                except ValueError as exc:
                    if strict:
                        raise ValueError(f"{plan_path}:{line_number}: {exc}") from exc
                    if on_skip is not None:
                        on_skip(plan_path, line_number, str(exc))
                    continue
                yield plan_path, normalized_record


def iter_plan_records(
    paths: list[str] | list[Path],
    *,
    strict: bool = True,
    on_skip: SkipCallback | None = None,
) -> Iterator[dict[str, Any]]:
    for _, record in iter_plan_records_with_source(paths, strict=strict, on_skip=on_skip):
        yield record


def load_plan_records(
    paths: list[str] | list[Path],
    *,
    strict: bool = True,
    on_skip: SkipCallback | None = None,
) -> list[dict[str, Any]]:
    return list(iter_plan_records(paths, strict=strict, on_skip=on_skip))

