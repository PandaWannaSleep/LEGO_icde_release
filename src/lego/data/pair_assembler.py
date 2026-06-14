from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from .plan_loader import iter_plan_records_with_source
from .plan_preprocessor import iter_plan_nodes


PAIRING_POLICIES = (
    "plan_position",
    "positive_anchor_key",
    "positive_anchor_key_breaker_plan_position",
)
RECORD_CONCURRENCY_LEVEL = "__record_concurrency_level__"

# Matches the PostgreSQL breaker policy used by the positive-anchor key
# implementation. Both Material spellings are accepted for JSON plans.
PIPELINE_BREAKERS = frozenset(
    {
        "Hash",
        "Sort",
        "Incremental Sort",
        "Material",
        "Materialize",
        "Aggregate",
        "WindowAgg",
        "SetOp",
        "Unique",
        "Gather",
        "Gather Merge",
        "Recursive Union",
        "CTE Scan",
    }
)

_COLUMN_FIELDS = (
    "Filter",
    "Index Cond",
    "Recheck Cond",
    "Hash Cond",
    "Merge Cond",
    "Join Filter",
    "Sort Key",
    "Group Key",
    "Output",
)
_SQL_WORDS = frozenset(
    {
        "and",
        "or",
        "not",
        "null",
        "true",
        "false",
        "is",
        "in",
        "any",
        "all",
        "like",
        "between",
        "case",
        "when",
        "then",
        "else",
        "end",
        "asc",
        "desc",
        "date",
        "timestamp",
        "time",
        "interval",
        "integer",
        "int",
        "int4",
        "int8",
        "numeric",
        "text",
        "varchar",
        "bpchar",
        "double",
        "precision",
        "real",
        "boolean",
        "oid",
        "partial",
    }
)
_IDENTIFIER_RE = re.compile(
    r'(?:"(?P<quoted_qual>[A-Za-z_][A-Za-z0-9_$]*)"|(?P<qual>[A-Za-z_][A-Za-z0-9_$]*))\.'
    r'(?:"(?P<quoted_col>[A-Za-z_][A-Za-z0-9_$]*)"|(?P<col>[A-Za-z_][A-Za-z0-9_$]*))'
    r'|(?:"(?P<quoted_single>[A-Za-z_][A-Za-z0-9_$]*)"|(?P<single>[A-Za-z_][A-Za-z0-9_$]*))'
)


@dataclass(frozen=True)
class PairAssemblyResult:
    pairs_path: Path
    singles_path: Path
    data_quality_path: Path


@dataclass
class _AnchorStats:
    operator_type: str
    key_payload: dict[str, Any]
    views_per_level: Counter[str]

    @property
    def view_count(self) -> int:
        return sum(self.views_per_level.values())

    @property
    def possible_pairs(self) -> int:
        return _choose_two(self.view_count)

    @property
    def cross_condition_pairs(self) -> int:
        levels = sorted(self.views_per_level)
        return sum(
            self.views_per_level[left] * self.views_per_level[right]
            for idx, left in enumerate(levels)
            for right in levels[idx + 1 :]
        )


def _choose_two(count: int) -> int:
    return count * (count - 1) // 2


def _record_key(record: dict[str, Any]) -> tuple[str, str, str, int]:
    return (
        str(record.get("benchmark", "unknown")),
        str(record.get("template_id", "unknown")),
        str(record.get("instance_id", "unknown")),
        int(record.get("repeat_id", 0)),
    )


def _resolve_level(level: str, record: dict[str, Any]) -> str:
    if level == RECORD_CONCURRENCY_LEVEL:
        return str(record.get("concurrency_level") or "unknown")
    return level


def _plan_signature(plan_record: dict[str, Any]) -> tuple[str, ...]:
    return tuple(node.get("Node Type", "?") for node in iter_plan_nodes(plan_record["planinfo"]["Plan"]))


def _node_view(record: dict[str, Any], node: dict[str, Any]) -> dict[str, Any]:
    return {
        "node": copy.deepcopy(node),
        "query": record.get("query"),
        "config": record.get("config", {}),
        "table_heat_metrics": record.get("table_heat_metrics", {}),
        "planinfo": {
            "System Metrics at Plan Time": copy.deepcopy(
                record.get("planinfo", {}).get("System Metrics at Plan Time")
            )
        },
    }


def _string_values(value: Any) -> Iterator[str]:
    if isinstance(value, list):
        for item in value:
            yield str(item)
    elif value is not None:
        yield str(value)


def _referenced_columns(node: dict[str, Any]) -> tuple[str, ...]:
    """Extract a normalized referenced-column set for the anchor key."""
    columns: set[str] = set()
    for field in _COLUMN_FIELDS:
        for expression in _string_values(node.get(field)):
            # Quoted string constants and PG type casts must not become columns.
            expression = re.sub(r"'(?:''|[^'])*'", " ", expression)
            expression = re.sub(r"::\s*[A-Za-z_][A-Za-z0-9_]*(?:\[\])?", " ", expression)
            for match in _IDENTIFIER_RE.finditer(expression):
                candidate = (
                    match.group("quoted_col")
                    or match.group("col")
                    or match.group("quoted_single")
                    or match.group("single")
                )
                if not candidate:
                    continue
                candidate = candidate.lower()
                if candidate in _SQL_WORDS:
                    continue
                tail = expression[match.end() :].lstrip()
                if "." not in match.group(0) and tail.startswith("("):
                    # A bare function/operator name such as count(...) is not a column.
                    continue
                columns.add(candidate)
    return tuple(sorted(columns))


def _operator_anchor_key(node: dict[str, Any]) -> tuple[Any, ...]:
    return (
        str(node.get("Node Type", "?")),
        str(node.get("Relation Name") or "<none>"),
        _referenced_columns(node),
        str(node.get("Index Name") or "<none>"),
        str(node.get("Strategy") or "<none>"),
    )


def _actual_cardinality_bucket(node: dict[str, Any]) -> str:
    """Log-scale bucket over the observed output cardinality."""
    value = node.get("Actual Rows")
    if value is None:
        return "unknown"
    try:
        rows = float(value)
    except (TypeError, ValueError):
        return "unknown"
    if not math.isfinite(rows) or rows < 0:
        return "unknown"
    if rows == 0:
        return "zero"
    exponent = int(math.floor(math.log10(rows)))
    return f"10^{exponent}"


def _is_pipeline_breaker(node: dict[str, Any]) -> bool:
    return str(node.get("Node Type", "?")) in PIPELINE_BREAKERS


def _iter_nodes_with_pipeline_path(
    plan_root: dict[str, Any],
) -> Iterator[tuple[int, dict[str, Any], tuple[tuple[str, str], ...]]]:
    """Yield postorder position plus the breaker-to-node path."""
    positions = {id(node): index for index, node in enumerate(iter_plan_nodes(plan_root))}

    def walk(
        node: dict[str, Any],
        parent_path: tuple[tuple[str, str], ...] | None,
        incoming_role: str,
    ) -> Iterator[tuple[int, dict[str, Any], tuple[tuple[str, str], ...]]]:
        node_type = str(node.get("Node Type", "?"))
        if _is_pipeline_breaker(node):
            path = (("breaker", node_type),)
        elif parent_path is None:
            path = (("breaker", "<ROOT_PIPELINE>"), ("root", node_type))
        else:
            path = parent_path + ((incoming_role, node_type),)
        yield positions[id(node)], node, path
        for child_index, child in enumerate(node.get("Plans", [])):
            role = "left" if child_index == 0 else "right" if child_index == 1 else f"child_{child_index}"
            yield from walk(child, path, role)

    yield from walk(plan_root, None, "root")


def _anchor_key_components(
    node: dict[str, Any],
    *,
    location_key: tuple[Any, ...],
    location_payload: dict[str, Any],
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    operator_key = _operator_anchor_key(node)
    bucket = _actual_cardinality_bucket(node)
    key = (operator_key, location_key, bucket)
    payload = {
        "operator_anchor": {
            "node_type": operator_key[0],
            "relation_name": operator_key[1],
            "referenced_columns": list(operator_key[2]),
            "index_name": operator_key[3],
            "strategy": operator_key[4],
        },
        "actual_cardinality_bucket": bucket,
    }
    payload.update(location_payload)
    return key, payload


def _anchor_key_components_for_policy(
    node: dict[str, Any],
    *,
    plan_position: int,
    pipeline_path: tuple[tuple[str, str], ...],
    pairing_policy: str,
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    if pairing_policy == "positive_anchor_key":
        return _anchor_key_components(
            node,
            location_key=("intra_pipeline_path", pipeline_path),
            location_payload={
                "intra_pipeline_path": [
                    {"direction": direction, "node_type": node_type}
                    for direction, node_type in pipeline_path
                ],
            },
        )
    if pairing_policy == "positive_anchor_key_breaker_plan_position":
        if _is_pipeline_breaker(node):
            return _anchor_key_components(
                node,
                location_key=("pipeline_breaker_plan_position", int(plan_position)),
                location_payload={
                    "location_component": "pipeline_breaker_plan_position",
                    "pipeline_breaker_node_type": str(node.get("Node Type", "?")),
                    "plan_position": int(plan_position),
                },
            )
        return _anchor_key_components(
            node,
            location_key=("intra_pipeline_path", pipeline_path),
            location_payload={
                "location_component": "intra_pipeline_path",
                "intra_pipeline_path": [
                    {"direction": direction, "node_type": node_type}
                    for direction, node_type in pipeline_path
                ],
            },
        )
    raise ValueError(f"Unsupported anchor-key pairing policy: {pairing_policy!r}")


def _anchor_id_from_payload(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return f"pak_{hashlib.sha256(encoded).hexdigest()[:24]}"


def _empty_output_files(
    pairs_path: Path,
    singles_path: Path,
    data_quality_path: Path,
) -> None:
    for path in (pairs_path, singles_path, data_quality_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    pairs_path.write_text("", encoding="utf-8")
    singles_path.write_text("", encoding="utf-8")


def _assemble_plan_position_pairs(
    *,
    raw_files_by_level: dict[str, list[str] | list[Path]],
    pairs_path: Path,
    singles_path: Path,
    quality_path: Path,
    strict: bool,
    write_records: bool,
) -> None:
    grouped: dict[tuple[str, str, str, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    total_records = 0
    for level, paths in raw_files_by_level.items():
        for _source_path, record in iter_plan_records_with_source(list(paths), strict=strict):
            record_level = _resolve_level(level, record)
            grouped[_record_key(record)][record_level] = record
            total_records += 1

    discarded_plan_shape_mismatch = 0
    discarded_per_template: Counter[str] = Counter()
    anchors_per_op_type: Counter[str] = Counter()
    view_records_per_op_type: Counter[str] = Counter()
    potential_pairs_per_op_type: Counter[str] = Counter()
    instances_with_three_levels = 0

    with pairs_path.open("w", encoding="utf-8") as pair_handle, singles_path.open(
        "w", encoding="utf-8"
    ) as single_handle:
        for key in sorted(grouped.keys()):
            views_by_level = grouped[key]
            benchmark, template_id, instance_id, repeat_id = key
            signatures = {level: _plan_signature(record) for level, record in views_by_level.items()}
            levels_sorted = sorted(views_by_level.keys())
            if len(levels_sorted) == 3:
                instances_with_three_levels += 1

            shape_match = len({sig for sig in signatures.values()}) == 1
            if len(levels_sorted) >= 2 and shape_match:
                shared_nodes = [
                    (idx, node, node.get("Node Type", "?"))
                    for idx, node in enumerate(iter_plan_nodes(views_by_level[levels_sorted[0]]["planinfo"]["Plan"]))
                ]
                for plan_position, _node, operator_type in shared_nodes:
                    anchor = {
                        "anchor_id": f"{benchmark}_{template_id}_{instance_id}_r{repeat_id}_p{plan_position}",
                        "benchmark": benchmark,
                        "template_id": template_id,
                        "instance_id": instance_id,
                        "repeat_id": repeat_id,
                        "plan_position": plan_position,
                        "operator_type": operator_type,
                        "views": {},
                    }
                    if write_records:
                        for level in levels_sorted:
                            level_nodes = list(iter_plan_nodes(views_by_level[level]["planinfo"]["Plan"]))
                            anchor["views"][level] = _node_view(views_by_level[level], level_nodes[plan_position])
                        pair_handle.write(json.dumps(anchor, ensure_ascii=False) + "\n")
                    anchors_per_op_type[operator_type] += 1
                    view_records_per_op_type[operator_type] += len(levels_sorted)
                    potential_pairs_per_op_type[operator_type] += _choose_two(len(levels_sorted))
            elif len(levels_sorted) >= 2 and not shape_match:
                discarded_plan_shape_mismatch += 1
                discarded_per_template[template_id] += 1

            if write_records:
                for level in levels_sorted:
                    nodes = list(iter_plan_nodes(views_by_level[level]["planinfo"]["Plan"]))
                    for plan_position, node in enumerate(nodes):
                        single = {
                            "instance_id": f"{benchmark}_{template_id}_{instance_id}_r{repeat_id}_{level}_p{plan_position}",
                            "benchmark": benchmark,
                            "template_id": template_id,
                            "repeat_id": repeat_id,
                            "plan_position": plan_position,
                            "concurrency_level": level,
                            "operator_type": node.get("Node Type", "?"),
                            "view": _node_view(views_by_level[level], node),
                        }
                        single_handle.write(json.dumps(single, ensure_ascii=False) + "\n")

    potential_pairs = sum(potential_pairs_per_op_type.values())
    quality_payload = {
        "pairing_policy": "plan_position",
        "total_records": total_records,
        "total_instances": len(grouped),
        "instances_with_three_levels": instances_with_three_levels,
        "discarded_plan_shape_mismatch": discarded_plan_shape_mismatch,
        "discarded_per_template": dict(discarded_per_template),
        "anchor_groups": sum(anchors_per_op_type.values()),
        "positive_view_records": sum(view_records_per_op_type.values()),
        "potential_positive_pairs": potential_pairs,
        "cross_condition_positive_pairs": potential_pairs,
        "same_condition_positive_pairs": 0,
        "anchors_per_op_type": dict(anchors_per_op_type),
        "positive_view_records_per_op_type": dict(view_records_per_op_type),
        "potential_positive_pairs_per_op_type": dict(potential_pairs_per_op_type),
        "available_levels": sorted(raw_files_by_level.keys()),
        "records_written": write_records,
    }
    with quality_path.open("w", encoding="utf-8") as handle:
        json.dump(quality_payload, handle, indent=2)


def _iter_positive_anchor_observations(
    raw_files_by_level: dict[str, list[str] | list[Path]],
    strict: bool,
    pairing_policy: str,
) -> Iterator[tuple[str, Path, int, dict[str, Any], int, dict[str, Any], tuple[Any, ...], dict[str, Any]]]:
    ordinal = 0
    for level, paths in raw_files_by_level.items():
        for source_path, record in iter_plan_records_with_source(list(paths), strict=strict):
            record_level = _resolve_level(level, record)
            root = record["planinfo"]["Plan"]
            for plan_position, node, pipeline_path in _iter_nodes_with_pipeline_path(root):
                anchor_key, payload = _anchor_key_components_for_policy(
                    node,
                    plan_position=plan_position,
                    pipeline_path=pipeline_path,
                    pairing_policy=pairing_policy,
                )
                yield record_level, source_path, ordinal, record, plan_position, node, anchor_key, payload
                ordinal += 1


def _assemble_positive_anchor_key_pairs(
    *,
    raw_files_by_level: dict[str, list[str] | list[Path]],
    pairs_path: Path,
    singles_path: Path,
    quality_path: Path,
    strict: bool,
    write_records: bool,
    pairing_policy: str,
) -> None:
    stats_by_key: dict[tuple[Any, ...], _AnchorStats] = {}
    total_observations = 0
    total_records_by_level: Counter[str] = Counter()
    seen_record_identities: set[tuple[str, str, str, str, int]] = set()
    for level, source_path, _ordinal, record, _position, node, anchor_key, payload in _iter_positive_anchor_observations(
        raw_files_by_level, strict, pairing_policy
    ):
        total_observations += 1
        seen_record_identities.add(
            (
                level,
                str(source_path),
                str(record.get("template_id", "unknown")),
                str(record.get("instance_id", "unknown")),
                int(record.get("repeat_id", 0)),
            )
        )
        stats = stats_by_key.get(anchor_key)
        if stats is None:
            stats = _AnchorStats(
                operator_type=str(node.get("Node Type", "?")),
                key_payload=payload,
                views_per_level=Counter(),
            )
            stats_by_key[anchor_key] = stats
        stats.views_per_level[level] += 1
    for level, *_rest in seen_record_identities:
        total_records_by_level[level] += 1

    eligible = {key: stats for key, stats in stats_by_key.items() if stats.view_count >= 2}
    anchors_per_op_type: Counter[str] = Counter()
    view_records_per_op_type: Counter[str] = Counter()
    potential_pairs_per_op_type: Counter[str] = Counter()
    cross_pairs_per_op_type: Counter[str] = Counter()
    for stats in eligible.values():
        anchors_per_op_type[stats.operator_type] += 1
        view_records_per_op_type[stats.operator_type] += stats.view_count
        potential_pairs_per_op_type[stats.operator_type] += stats.possible_pairs
        cross_pairs_per_op_type[stats.operator_type] += stats.cross_condition_pairs

    if write_records:
        with pairs_path.open("w", encoding="utf-8") as pair_handle, singles_path.open(
            "w", encoding="utf-8"
        ) as single_handle:
            for level, source_path, ordinal, record, plan_position, node, anchor_key, payload in _iter_positive_anchor_observations(
                raw_files_by_level, strict, pairing_policy
            ):
                benchmark = str(record.get("benchmark", "unknown"))
                template_id = str(record.get("template_id", "unknown"))
                instance_id = str(record.get("instance_id", "unknown"))
                repeat_id = int(record.get("repeat_id", 0))
                operator_type = str(node.get("Node Type", "?"))
                view = _node_view(record, node)
                single = {
                    "instance_id": f"{benchmark}_{template_id}_{instance_id}_r{repeat_id}_{level}_p{plan_position}",
                    "benchmark": benchmark,
                    "template_id": template_id,
                    "repeat_id": repeat_id,
                    "plan_position": plan_position,
                    "concurrency_level": level,
                    "operator_type": operator_type,
                    "view": view,
                }
                single_handle.write(json.dumps(single, ensure_ascii=False) + "\n")
                if anchor_key not in eligible:
                    continue
                anchor_id = _anchor_id_from_payload(payload)
                pair_view = {
                    "record_kind": "positive_anchor_view",
                    "pairing_policy": pairing_policy,
                    "anchor_id": anchor_id,
                    "anchor_key": payload,
                    "positive_anchor_key": payload,
                    "view_id": f"{level}:{source_path.name}:{ordinal}:p{plan_position}",
                    "benchmark": benchmark,
                    "template_id": template_id,
                    "instance_id": instance_id,
                    "repeat_id": repeat_id,
                    "plan_position": plan_position,
                    "concurrency_level": level,
                    "operator_type": operator_type,
                    "view": view,
                }
                pair_handle.write(json.dumps(pair_view, ensure_ascii=False) + "\n")

    total_pairs = sum(potential_pairs_per_op_type.values())
    cross_pairs = sum(cross_pairs_per_op_type.values())
    largest_anchor_groups = [
        {
            "anchor_id": _anchor_id_from_payload(stats.key_payload),
            "operator_type": stats.operator_type,
            "view_count": stats.view_count,
            "views_per_level": dict(stats.views_per_level),
            "potential_positive_pairs": stats.possible_pairs,
            "cross_condition_positive_pairs": stats.cross_condition_pairs,
            "positive_anchor_key": stats.key_payload,
        }
        for stats in sorted(
            eligible.values(),
            key=lambda item: (item.possible_pairs, item.view_count),
            reverse=True,
        )[:10]
    ]
    quality_payload = {
        "pairing_policy": pairing_policy,
        "anchor_key_definition": {
            "operator_anchor": [
                "node_type",
                "relation_name",
                "referenced_column_set",
                "index_name",
                "strategy",
            ],
            "cardinality_bucket": "floor(log10(Actual Rows)); zero and unknown are separate buckets",
        },
        "total_records": sum(total_records_by_level.values()),
        "total_records_per_level": dict(total_records_by_level),
        "total_operator_observations": total_observations,
        "candidate_anchor_keys": len(stats_by_key),
        "singleton_anchor_keys": len(stats_by_key) - len(eligible),
        "anchor_groups": len(eligible),
        "positive_view_records": sum(view_records_per_op_type.values()),
        "potential_positive_pairs": total_pairs,
        "cross_condition_positive_pairs": cross_pairs,
        "same_condition_positive_pairs": total_pairs - cross_pairs,
        "anchors_per_op_type": dict(anchors_per_op_type),
        "positive_view_records_per_op_type": dict(view_records_per_op_type),
        "potential_positive_pairs_per_op_type": dict(potential_pairs_per_op_type),
        "cross_condition_positive_pairs_per_op_type": dict(cross_pairs_per_op_type),
        "largest_anchor_groups": largest_anchor_groups,
        "available_levels": sorted(total_records_by_level.keys()),
        "records_written": write_records,
    }
    if pairing_policy == "positive_anchor_key":
        quality_payload["anchor_key_definition"][
            "intra_pipeline_path"
        ] = "nearest pipeline breaker to target node, with child direction and node type"
        quality_payload["anchor_key_definition"]["pipeline_breakers"] = sorted(PIPELINE_BREAKERS)
    elif pairing_policy == "positive_anchor_key_breaker_plan_position":
        quality_payload["anchor_key_definition"][
            "pipeline_breaker_plan_position"
        ] = "postorder plan node position, used only when the target node itself is a pipeline breaker"
        quality_payload["anchor_key_definition"][
            "intra_pipeline_path"
        ] = "non-breaker nodes still use nearest pipeline breaker to target node, with child direction and node type"
        quality_payload["anchor_key_definition"]["pipeline_breakers"] = sorted(PIPELINE_BREAKERS)
    quality_payload["positive_anchor_key_definition"] = quality_payload["anchor_key_definition"]
    with quality_path.open("w", encoding="utf-8") as handle:
        json.dump(quality_payload, handle, indent=2)


def assemble_pairs(
    *,
    raw_files_by_level: dict[str, list[str] | list[Path]],
    pairs_path: str | Path,
    singles_path: str | Path,
    data_quality_path: str | Path,
    strict: bool = True,
    pairing_policy: str = "plan_position",
    write_records: bool = True,
) -> PairAssemblyResult:
    """Assemble positive anchors using a selected pairing policy.

    ``plan_position`` preserves the existing same-query, same-plan-position
    protocol. ``positive_anchor_key`` implements the structured key. The
    ``positive_anchor_key_breaker_plan_position`` ablates only the location
    component of pipeline-breaker nodes: breakers use postorder plan position,
    while non-breakers keep the intra-pipeline path. Anchor-key policies
    write a streamed ``positive_anchor_view`` format so large anchor groups
    remain tractable. The paired dataset reader accepts both formats.
    """
    if pairing_policy not in PAIRING_POLICIES:
        raise ValueError(f"pairing_policy must be one of {PAIRING_POLICIES}, got {pairing_policy!r}")

    pairs_out = Path(pairs_path)
    singles_out = Path(singles_path)
    quality_out = Path(data_quality_path)
    _empty_output_files(pairs_out, singles_out, quality_out)

    if pairing_policy == "plan_position":
        _assemble_plan_position_pairs(
            raw_files_by_level=raw_files_by_level,
            pairs_path=pairs_out,
            singles_path=singles_out,
            quality_path=quality_out,
            strict=strict,
            write_records=write_records,
        )
    else:
        _assemble_positive_anchor_key_pairs(
            raw_files_by_level=raw_files_by_level,
            pairs_path=pairs_out,
            singles_path=singles_out,
            quality_path=quality_out,
            strict=strict,
            write_records=write_records,
            pairing_policy=pairing_policy,
        )

    return PairAssemblyResult(
        pairs_path=pairs_out,
        singles_path=singles_out,
        data_quality_path=quality_out,
    )
