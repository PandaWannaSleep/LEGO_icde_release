from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable


_WORKLOAD_NAME_RE = re.compile(r"^(?P<template>\d+)(?P<instance>[A-Za-z][A-Za-z0-9_-]*)$")


@dataclass(frozen=True)
class WorkloadEntry:
    benchmark: str
    template_id: str
    instance_id: str
    sql: str
    source_path: str


def _normalize_sql(sql: str) -> str:
    collapsed = " ".join(sql.split())
    if not collapsed:
        raise ValueError("SQL is empty after normalization")
    return collapsed


def parse_workload_line(line: str, *, benchmark: str, source_path: str, line_number: int) -> WorkloadEntry:
    stripped = line.strip()
    if not stripped:
        raise ValueError("Empty workload line")
    parts = stripped.split("\t", 2)
    if len(parts) == 3:
        template_id, instance_id, sql = parts
        return WorkloadEntry(
            benchmark=benchmark,
            template_id=template_id.strip() or "unknown",
            instance_id=instance_id.strip() or str(line_number),
            sql=_normalize_sql(sql),
            source_path=f"{source_path}:{line_number}",
        )
    return WorkloadEntry(
        benchmark=benchmark,
        template_id="unknown",
        instance_id=str(line_number),
        sql=_normalize_sql(stripped),
        source_path=f"{source_path}:{line_number}",
    )


def load_workload_manifest(path: str | Path, *, benchmark: str, limit: int = 0) -> list[WorkloadEntry]:
    plan_path = Path(path)
    entries: list[WorkloadEntry] = []
    with plan_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            entries.append(
                parse_workload_line(
                    stripped,
                    benchmark=benchmark,
                    source_path=str(plan_path),
                    line_number=line_number,
                )
            )
            if limit and len(entries) >= limit:
                break
    return entries


def _parse_job_path(path: Path) -> tuple[str, str]:
    match = _WORKLOAD_NAME_RE.match(path.stem)
    if not match:
        raise ValueError(
            f"JOB workload filename must look like <template><instance>.sql (e.g. 6f.sql), got {path.name!r}"
        )
    return match.group("template"), match.group("instance")


def load_job_directory_manifest(directory: str | Path, *, benchmark: str = "job", limit: int = 0) -> list[WorkloadEntry]:
    root = Path(directory)
    entries: list[WorkloadEntry] = []
    for path in sorted(root.glob("*.sql")):
        template_id, instance_id = _parse_job_path(path)
        sql = _normalize_sql(path.read_text(encoding="utf-8"))
        entries.append(
            WorkloadEntry(
                benchmark=benchmark,
                template_id=template_id,
                instance_id=instance_id,
                sql=sql,
                source_path=str(path),
            )
        )
        if limit and len(entries) >= limit:
            break
    return entries


def dump_workload_manifest(entries: Iterable[WorkloadEntry], path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(f"{entry.template_id}\t{entry.instance_id}\t{entry.sql}\n")
