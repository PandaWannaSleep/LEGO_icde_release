from __future__ import annotations

import copy
import logging
import re
from typing import Any

from lego.artifact.operator_catalog import (
    DEFAULT_CHILD_OPERATOR,
    DEFAULT_PARENT_OPERATOR,
    DEFAULT_STRATEGY,
)
from lego.cag.node_schema import NodeSchema, default_node_schema
from .operator_context import OperatorContext, OperatorLabels, OperatorMetadata
from .schema_stats import SchemaStatsSnapshot


logger = logging.getLogger(__name__)


def _unwrap_system_metrics(container: dict[str, Any] | None) -> dict[str, Any] | None:
    """Return the inner ``System Metrics`` dict from a plan node or planinfo container.

    PostgreSQL 13.3 (patched) emits sys metrics in two wrappers:

      * per-node:    ``container["Pre-execution Metrics"]["System Metrics"]``
      * plan-level:  ``container["System Metrics at Plan Time"]["System Metrics"]``

    The patched ``explain.c`` originally used ``ExplainOpenGroup(..., labeled=false, ...)``
    for the outer wrappers, which emitted a JSON ``[`` array containing a labeled
    ``"System Metrics": {...}`` member — syntactically invalid JSON. The fix in
    ``explain.c:661`` and ``:2080`` switched the outer ``labeled`` flag to ``true``,
    so wrappers are now JSON objects. This helper accepts both shapes (dict from
    fixed binary, list from legacy/buggy binary) plus the legacy flat shape
    (``container["System Metrics"]``) and returns ``None`` when no sys metrics
    are present.
    """
    if not isinstance(container, dict):
        return None

    def _resolve_outer(outer: Any) -> dict[str, Any] | None:
        if isinstance(outer, dict):
            sm = outer.get("System Metrics")
            if isinstance(sm, dict):
                return sm
            # Outer dict already IS the SystemMetrics body (5 category sub-objects);
            # accept this shape too for forward-compat against future PG emit changes.
            if any(k in outer for k in ("CPU", "Load Average", "Memory")):
                return outer
            return None
        if isinstance(outer, list) and outer and isinstance(outer[0], dict):
            return _resolve_outer(outer[0])
        return None

    sm = _resolve_outer(container.get("Pre-execution Metrics"))
    if sm is not None:
        return sm
    sm = _resolve_outer(container.get("System Metrics at Plan Time"))
    if sm is not None:
        return sm
    flat = container.get("System Metrics")
    if isinstance(flat, dict):
        return flat
    return None


class OperatorContextExtractor:
    """
    Extract a minimal OperatorContext from a normalized plan record.

    This first-stage extractor intentionally depends only on plan-native fields.
    DB-aware enrichment such as index statistics or schema-derived offsets can
    be added later through dedicated enrichers without changing the CAG object
    boundary.
    """

    def __init__(
        self,
        node_schema: NodeSchema | None = None,
        schema_stats: SchemaStatsSnapshot | None = None,
    ):
        self.node_schema = node_schema or default_node_schema()
        self.schema_stats = schema_stats
        # Diagnostic counters reset at every extract_plan entry. Surface as
        # logger.warning when the miss rate exceeds 50% on a single record.
        self._missing_resource_count = 0
        self._total_resource_calls = 0

    def extract_plan(self, plan_record: dict[str, Any], source_path: str | None = None) -> list[OperatorContext]:
        plan_root = copy.deepcopy(plan_record["planinfo"]["Plan"])
        plan_level_metrics = _unwrap_system_metrics(plan_record.get("planinfo"))
        self._missing_resource_count = 0
        self._total_resource_calls = 0
        contexts: list[OperatorContext] = []
        self._walk(
            node=plan_root,
            contexts=contexts,
            query_text=plan_record.get("query"),
            config=plan_record.get("config", {}),
            table_heat_metrics=plan_record.get("table_heat_metrics", {}),
            source_path=source_path,
            plan_level_metrics=plan_level_metrics,
        )
        if self._total_resource_calls and self._missing_resource_count:
            miss_rate = self._missing_resource_count / self._total_resource_calls
            if miss_rate >= 0.5:
                logger.warning(
                    "extractor missed system metrics on %d/%d nodes (%.0f%%); "
                    "record source=%s. If running against patched PG-13.3 the JSON should "
                    "contain Pre-execution Metrics; otherwise OS resource features will be 0.",
                    self._missing_resource_count,
                    self._total_resource_calls,
                    100 * miss_rate,
                    source_path,
                )
        return contexts

    def extract_operator(
        self,
        node: dict[str, Any],
        query_text: str | None = None,
        config: dict[str, Any] | None = None,
        table_heat_metrics: dict[str, Any] | None = None,
        source_path: str | None = None,
        plan_level_metrics: dict[str, Any] | None = None,
    ) -> OperatorContext:
        return self._extract_single(
            node=node,
            query_text=query_text,
            config=config or {},
            table_heat_metrics=table_heat_metrics or {},
            source_path=source_path,
            plan_level_metrics=plan_level_metrics,
        )

    def _walk(
        self,
        node: dict[str, Any],
        contexts: list[OperatorContext],
        query_text: str | None,
        config: dict[str, Any],
        table_heat_metrics: dict[str, Any],
        source_path: str | None,
        plan_level_metrics: dict[str, Any] | None = None,
    ) -> None:
        for child in node.get("Plans", []):
            if "Subplan Name" not in child:
                self._walk(
                    child, contexts, query_text, config, table_heat_metrics,
                    source_path, plan_level_metrics,
                )

        contexts.append(
            self._extract_single(
                node, query_text, config, table_heat_metrics, source_path,
                plan_level_metrics=plan_level_metrics,
            )
        )

    def _extract_single(
        self,
        node: dict[str, Any],
        query_text: str | None,
        config: dict[str, Any],
        table_heat_metrics: dict[str, Any],
        source_path: str | None,
        plan_level_metrics: dict[str, Any] | None = None,
    ) -> OperatorContext:
        behavior = self._init_behavior_features()
        behavior.update(self._extract_behavior(node))
        resource = self._extract_resource_features(node, plan_level_fallback=plan_level_metrics)
        table_heat = self._extract_table_heat(node, table_heat_metrics)

        labels = OperatorLabels(
            actual_startup_time=float(node.get("Actual Startup Time", 0.0) or 0.0),
            actual_total_time=float(node.get("Actual Total Time", 0.0) or 0.0),
            optimizer_startup_cost=float(node.get("Startup Cost", 0.0) or 0.0),
            optimizer_total_cost=float(node.get("Total Cost", 0.0) or 0.0),
        )
        metadata = OperatorMetadata(
            operator_type=node["Node Type"],
            parent_operator=node.get("parent", DEFAULT_PARENT_OPERATOR),
            relation_name=node.get("Relation Name"),
            query_text=query_text,
            source_path=source_path,
            raw_plan=node,
            extras={"config": config},
        )
        return OperatorContext(
            operator_type=node["Node Type"],
            behavior_features=behavior,
            resource_features=resource,
            table_heat_features=table_heat,
            labels=labels,
            metadata=metadata,
        )

    def _init_behavior_features(self) -> dict[str, float | str]:
        behavior: dict[str, float | str] = {}
        for node_name in self.node_schema.behavior_nodes:
            if node_name in self.node_schema.categorical_nodes:
                behavior[node_name] = DEFAULT_STRATEGY if node_name == "Strategy" else DEFAULT_CHILD_OPERATOR
            else:
                behavior[node_name] = 0.0
        behavior["ParentOp"] = DEFAULT_PARENT_OPERATOR
        return behavior

    def _extract_behavior(self, node: dict[str, Any]) -> dict[str, float | str]:
        res: dict[str, float | str] = {}
        child_nodes = [child for child in node.get("Plans", []) if "Subplan Name" not in child]

        res["ParentOp"] = node.get("parent", DEFAULT_PARENT_OPERATOR)
        res["SoutAvg"] = float(node.get("Plan Width", 0.0) or 0.0)
        res["Rows"] = float(node.get("Actual Rows", node.get("Plan Rows", 0.0)) or 0.0)
        res["Loops"] = float(node.get("Actual Loops", 0.0) or 0.0)

        left = child_nodes[0] if len(child_nodes) >= 1 else None
        right = child_nodes[1] if len(child_nodes) >= 2 else None
        res["LeftOp"] = left["Node Type"] if left else DEFAULT_CHILD_OPERATOR
        res["RightOp"] = right["Node Type"] if right else DEFAULT_CHILD_OPERATOR
        res["LeftSoutAvg"] = float(left.get("Plan Width", 0.0) if left else 0.0)
        res["LeftRows"] = float(left.get("Actual Rows", left.get("Plan Rows", 0.0)) if left else 0.0)
        res["LeftLoops"] = float(left.get("Actual Loops", 0.0) if left else 0.0)
        res["RightSoutAvg"] = float(right.get("Plan Width", 0.0) if right else 0.0)
        res["RightRows"] = float(right.get("Actual Rows", right.get("Plan Rows", 0.0)) if right else 0.0)
        res["RightLoops"] = float(right.get("Actual Loops", 0.0) if right else 0.0)
        res["InnerUnique"] = 1.0 if node.get("Inner Unique") is True else 0.0

        strategy = self._extract_strategy(node)
        res["Strategy"] = strategy

        res["FilterNum"] = float(self._count_predicates(node.get("Filter")))
        res["CondNum"] = float(
            self._count_predicates(
                node.get("Index Cond")
                or node.get("Hash Cond")
                or node.get("Merge Cond")
                or node.get("Sort Key")
                or node.get("Group Key")
            )
        )
        res.update(self._extract_filter_cond_stats(node))
        res.update(self._extract_index_stats(node))
        res.update(self._extract_table_stats(node))

        if node["Node Type"] == "Hash Join":
            hash_node = right
            if hash_node is not None:
                res["BucketsNum"] = float(hash_node.get("Hash Buckets", 1) or 1)
                res["BatchesNum"] = float(hash_node.get("Hash Batches", 1) or 1)
        elif node["Node Type"] == "Aggregate":
            res["BatchesNum"] = float(node.get("HashAgg Batches", 1) or 1)

        return res

    def _extract_filter_cond_stats(self, node: dict[str, Any]) -> dict[str, float]:
        stats = {
            "FilterNum": 0.0,
            "FilterOffset": 0.0,
            "FilterIntegerRatio": 0.0,
            "FilterFloatRatio": 0.0,
            "FilterStrRatio": 0.0,
            "FilterColumnNum": 0.0,
            "CondNum": 0.0,
            "CondOffset": 0.0,
            "CondIntegerRatio": 0.0,
            "CondFloatRatio": 0.0,
            "CondStrRatio": 0.0,
            "CondColumnNum": 0.0,
        }

        filter_info = node.get("Filter")
        if filter_info is not None:
            stats.update(self._feature_stats_from_expression(filter_info, prefix="Filter"))

        cond_info = (
            node.get("Index Cond")
            or node.get("Hash Cond")
            or node.get("Merge Cond")
            or node.get("Sort Key")
            or node.get("Group Key")
        )
        if cond_info is not None:
            stats.update(self._feature_stats_from_expression(cond_info, prefix="Cond"))

        return stats

    def _extract_index_stats(self, node: dict[str, Any]) -> dict[str, float]:
        stats = {
            "IndexCorrelation": 0.0,
            "IndexTreeHeight": 0.0,
            "IndexTreePages": 0.0,
            "IndexTreeUniqueValues": 0.0,
        }
        if self.schema_stats is None or node["Node Type"] not in {"Index Scan", "Index Only Scan"}:
            return stats

        index_name = node.get("Index Name")
        index_info = self.schema_stats.index_features.get(index_name, {})
        stats.update(
            {
                "IndexCorrelation": float(index_info.get("indexCorrelation", 0.0) or 0.0),
                "IndexTreeHeight": float(index_info.get("tree_height", 0.0) or 0.0),
                "IndexTreePages": float(index_info.get("pages", 0.0) or 0.0),
                "IndexTreeUniqueValues": float(index_info.get("distinctnum", 0.0) or 0.0),
            }
        )
        return stats

    def _extract_table_stats(self, node: dict[str, Any]) -> dict[str, float]:
        stats = {
            "TablePages": 0.0,
            "TuplesPerBlock": 0.0,
            "Selectivity": 0.0,
        }
        if self.schema_stats is None:
            return stats

        relation_name = node.get("Relation Name")
        if not relation_name:
            return stats

        table_info = self.schema_stats.table_features.get(relation_name)
        if not table_info:
            return stats

        table_pages = float(table_info.get("table_pages", 0.0) or 0.0)
        tuple_num = float(table_info.get("tuple_num", 0.0) or 0.0)
        stats["TablePages"] = table_pages
        if table_pages > 0:
            stats["TuplesPerBlock"] = tuple_num / table_pages

        if node["Node Type"] in {"Seq Scan", "Index Scan", "Index Only Scan"} and tuple_num > 0:
            rows_removed = 0.0
            rows_removed += float(node.get("Rows Removed by Filter", 0.0) or 0.0)
            rows_removed += float(node.get("Rows Removed by Index Recheck", 0.0) or 0.0)
            observed_rows = float(node.get("Actual Rows", node.get("Plan Rows", 0.0)) or 0.0)
            stats["Selectivity"] = (observed_rows + rows_removed) / tuple_num

        return stats

    def _extract_strategy(self, node: dict[str, Any]) -> str:
        if "Join Type" in node:
            return str(node["Join Type"])
        if "Sort Method" in node:
            return str(node["Sort Method"])
        if "Strategy" in node:
            return str(node["Strategy"])
        if "Scan Direction" in node:
            return str(node["Scan Direction"])
        return DEFAULT_STRATEGY

    def _feature_stats_from_expression(self, predicate_info: Any, prefix: str) -> dict[str, float]:
        filters = self._split_predicates(predicate_info)
        if not filters:
            return {
                f"{prefix}Num": 0.0,
                f"{prefix}Offset": 0.0,
                f"{prefix}IntegerRatio": 0.0,
                f"{prefix}FloatRatio": 0.0,
                f"{prefix}StrRatio": 0.0,
                f"{prefix}ColumnNum": 0.0,
            }

        int_num = 0
        float_num = 0
        str_num = 0
        largest_offset = 0.0
        columns: set[str] = set()

        for predicate in filters:
            column_info = self._infer_column_info(predicate)
            if not column_info:
                str_num += 1
                continue

            columns.add(column_info["column"])
            largest_offset = max(largest_offset, float(column_info.get("offset", 0.0) or 0.0))
            column_type = column_info.get("type")
            if column_type == "int4" and "numeric" not in predicate:
                int_num += 1
            elif "numeric" in predicate or column_type == "numeric":
                float_num += 1
            else:
                str_num += 1

        total = int_num + float_num + str_num
        if total == 0:
            total = 1

        return {
            f"{prefix}Num": float(len(filters)),
            f"{prefix}Offset": float(largest_offset),
            f"{prefix}IntegerRatio": int_num / total,
            f"{prefix}FloatRatio": float_num / total,
            f"{prefix}StrRatio": str_num / total,
            f"{prefix}ColumnNum": float(len(columns)),
        }

    def _split_predicates(self, predicate_info: Any) -> list[str]:
        if predicate_info is None:
            return []
        if isinstance(predicate_info, list):
            return [str(item) for item in predicate_info if str(item).strip()]
        if isinstance(predicate_info, str):
            atoms: list[str] = []
            for and_part in re.split(r"\sAND\s", predicate_info):
                atoms.extend(re.split(r"\sOR\s", and_part))
            return [atom.strip() for atom in atoms if atom.strip()]
        return []

    def _infer_column_info(self, predicate: str) -> dict[str, Any]:
        if self.schema_stats is None:
            return {}

        token = re.split(r"[<>!=)~]|(\sIS\s)", predicate)[0].replace("(", "").strip()
        pieces = token.split(".")
        if len(pieces) >= 2:
            candidate = f"{pieces[0].strip()}.{pieces[1].strip()}"
        else:
            candidate = token
        candidate = candidate.replace('"', "")
        return self.schema_stats.get_column_info(candidate)

    def _count_predicates(self, predicate_info: Any) -> int:
        if predicate_info is None:
            return 0
        if isinstance(predicate_info, list):
            return len(predicate_info)
        if isinstance(predicate_info, str):
            atoms: list[str] = []
            for and_part in re.split(r"\sAND\s", predicate_info):
                atoms.extend(re.split(r"\sOR\s", and_part))
            return len([atom for atom in atoms if atom.strip()])
        return 0

    def _extract_resource_features(
        self,
        node: dict[str, Any],
        plan_level_fallback: dict[str, Any] | None = None,
    ) -> dict[str, float]:
        self._total_resource_calls += 1
        resource = {name: 0.0 for name in self.node_schema.resource_nodes}
        sys_metrics = _unwrap_system_metrics(node) or plan_level_fallback
        if not sys_metrics:
            self._missing_resource_count += 1
            return resource

        cpu = sys_metrics.get("CPU", {})
        load = sys_metrics.get("Load Average", {})
        memory = sys_metrics.get("Memory", {})
        pg = sys_metrics.get("PostgreSQL Process", {})
        disk = sys_metrics.get("Disk IO", {})

        resource.update(
            {
                "cpu_user_percent": float(cpu.get("User Percent", 0.0) or 0.0),
                "cpu_system_percent": float(cpu.get("System Percent", 0.0) or 0.0),
                "cpu_idle_percent": float(cpu.get("Idle Percent", 0.0) or 0.0),
                "load_avg_1min": float(load.get("1 Minute", 0.0) or 0.0),
                "load_avg_5min": float(load.get("5 Minutes", 0.0) or 0.0),
                "load_avg_15min": float(load.get("15 Minutes", 0.0) or 0.0),
                "memory_total_kb": float(memory.get("Total KB", 0.0) or 0.0),
                "memory_used_kb": float(memory.get("Used KB", 0.0) or 0.0),
                "memory_used_percent": float(memory.get("Used Percent", 0.0) or 0.0),
                "memory_free_kb": float(memory.get("Free KB", 0.0) or 0.0),
                "postgresql_process_memory_kb": float(pg.get("Memory KB", 0.0) or 0.0),
                "postgresql_process_cpu_percent": float(pg.get("CPU Percent", 0.0) or 0.0),
                "disk_io_read_count": float(disk.get("Read Count", 0.0) or 0.0),
                "disk_io_read_kb": float(disk.get("Read KB", 0.0) or 0.0),
                "disk_io_write_count": float(disk.get("Write Count", 0.0) or 0.0),
                "disk_io_write_kb": float(disk.get("Write KB", 0.0) or 0.0),
            }
        )
        return resource

    def _extract_table_heat(self, node: dict[str, Any], table_heat_metrics: dict[str, Any]) -> dict[str, float]:
        table_heat = {name: 0.0 for name in self.node_schema.table_heat_nodes}
        relation_name = node.get("Relation Name")
        if not relation_name:
            return table_heat

        for relation in table_heat_metrics.values():
            if relation_name != relation.get("table_name"):
                continue
            for key in self.node_schema.table_heat_nodes:
                table_heat[key] = float(relation.get(key, 0.0) or 0.0)
            break
        return table_heat
