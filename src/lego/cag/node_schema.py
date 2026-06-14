from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping

from lego.artifact.operator_catalog import OPERATOR_CATEGORICAL_FEATURES


QUERY_BEHAVIOR_NODES = (
    "FilterNum",
    "FilterOffset",
    "FilterIntegerRatio",
    "FilterFloatRatio",
    "FilterStrRatio",
    "FilterColumnNum",
    "CondNum",
    "CondOffset",
    "CondIntegerRatio",
    "CondFloatRatio",
    "CondStrRatio",
    "CondColumnNum",
    "IndexCorrelation",
    "IndexTreeHeight",
    "IndexTreePages",
    "IndexTreeUniqueValues",
    "ParentOp",
    "SoutAvg",
    "Rows",
    "Loops",
    "LeftOp",
    "RightOp",
    "LeftSoutAvg",
    "LeftRows",
    "LeftLoops",
    "RightSoutAvg",
    "RightRows",
    "RightLoops",
    "InnerUnique",
    "TablePages",
    "TuplesPerBlock",
    "Selectivity",
    "BucketsNum",
    "BatchesNum",
    "Strategy",
)

SYSTEM_RESOURCE_NODES = (
    "cpu_user_percent",
    "cpu_system_percent",
    "cpu_idle_percent",
    "load_avg_1min",
    "load_avg_5min",
    "load_avg_15min",
    "memory_total_kb",
    "memory_used_kb",
    "memory_used_percent",
    "memory_free_kb",
    "postgresql_process_memory_kb",
    "postgresql_process_cpu_percent",
    "disk_io_read_count",
    "disk_io_read_kb",
    "disk_io_write_count",
    "disk_io_write_kb",
)

TABLE_HEAT_NODES = (
    "recent_seq_scan",
    "recent_seq_tup_read",
    "recent_idx_scan",
    "recent_idx_tup_fetch",
)


# Per-node value-type tags. The encoder dispatches each column to a
# type-specific path: categorical embedding, log1p, or z-score.
VALUE_TYPE_CATEGORICAL = "categorical"
VALUE_TYPE_LOG_SCALE = "log_scale"
VALUE_TYPE_Z_SCORE = "z_score"
VALID_VALUE_TYPES = frozenset(
    {VALUE_TYPE_CATEGORICAL, VALUE_TYPE_LOG_SCALE, VALUE_TYPE_Z_SCORE}
)

# Heavy-tailed positive scalars where ``log1p`` is a natural normalisation.
# Cardinalities, page counts, KB sizes, and a handful of cumulative IO counters
# whose distribution is dominated by their tail.
LOG_SCALE_NODES = frozenset(
    {
        # Cardinality / row counts.
        "Rows",
        "LeftRows",
        "RightRows",
        # Index / table page counts.
        "IndexTreePages",
        "TablePages",
        # Hash join / aggregate batching counts.
        "BucketsNum",
        "BatchesNum",
        # Memory sizes (KB, heavy-tailed positive).
        "memory_total_kb",
        "memory_used_kb",
        "memory_free_kb",
        "postgresql_process_memory_kb",
        # Disk IO byte counters (KB, heavy-tailed positive).
        "disk_io_read_kb",
        "disk_io_write_kb",
        # Cumulative table heat counters (recent tuple counts).
        "recent_seq_tup_read",
        "recent_idx_tup_fetch",
    }
)


@dataclass(frozen=True)
class NodeSchema:
    behavior_nodes: tuple[str, ...]
    resource_nodes: tuple[str, ...]
    table_heat_nodes: tuple[str, ...]
    categorical_nodes: frozenset[str]
    # Per-node value type. Stored as an immutable tuple of ``(node_name,
    # type_tag)`` pairs so the dataclass remains hashable. ``None`` means
    # "not set"; callers should use :meth:`value_types` to get a Mapping.
    node_value_types: tuple[tuple[str, str], ...] | None = field(default=None)

    @property
    def node_order(self) -> tuple[str, ...]:
        return self.behavior_nodes + self.resource_nodes + self.table_heat_nodes

    @property
    def size(self) -> int:
        return len(self.node_order)

    def value_types(self) -> Mapping[str, str]:
        """Return a value-type map for every node in ``node_order``.

        If ``node_value_types`` was not provided at construction time, this
        derives one via :func:`default_node_value_types`. Always returns a
        complete mapping (one entry per node in ``node_order``) wrapped in a
        ``MappingProxyType`` so callers cannot mutate the schema's view.
        """
        if self.node_value_types is not None:
            return MappingProxyType(dict(self.node_value_types))
        return default_node_value_types(self)


def default_node_value_types(schema: NodeSchema) -> Mapping[str, str]:
    """Default value-type assignment for every node in ``schema.node_order``.

    * categorical → ``schema.categorical_nodes`` (ParentOp / LeftOp / RightOp /
      Strategy by default).
    * log_scale → :data:`LOG_SCALE_NODES` (cardinalities, KB sizes, page
      counts, hash batch counts, recent tuple counters).
    * z_score → everything else (CPU %, load averages, ratios, IO counts,
      selectivity, etc.).

    Returns an immutable :class:`MappingProxyType` so the schema remains a
    value type.
    """
    mapping: dict[str, str] = {}
    for name in schema.node_order:
        if name in schema.categorical_nodes:
            mapping[name] = VALUE_TYPE_CATEGORICAL
        elif name in LOG_SCALE_NODES:
            mapping[name] = VALUE_TYPE_LOG_SCALE
        else:
            mapping[name] = VALUE_TYPE_Z_SCORE
    return MappingProxyType(mapping)


def make_node_schema(
    *,
    include_system_resource: bool = True,
    include_table_heat: bool = True,
) -> NodeSchema:
    """Build the OBG node schema with optional feature-group ablations."""
    base = NodeSchema(
        behavior_nodes=QUERY_BEHAVIOR_NODES,
        resource_nodes=SYSTEM_RESOURCE_NODES if include_system_resource else (),
        table_heat_nodes=TABLE_HEAT_NODES if include_table_heat else (),
        categorical_nodes=frozenset(OPERATOR_CATEGORICAL_FEATURES),
    )
    # Bake the default value-type mapping into the schema so callers see a
    # fully-typed object. A bare ``NodeSchema(...)`` without ``node_value_types``
    # still works because ``value_types()`` falls back to the same factory.
    value_types_mapping = default_node_value_types(base)
    return NodeSchema(
        behavior_nodes=base.behavior_nodes,
        resource_nodes=base.resource_nodes,
        table_heat_nodes=base.table_heat_nodes,
        categorical_nodes=base.categorical_nodes,
        node_value_types=tuple(value_types_mapping.items()),
    )


def default_node_schema() -> NodeSchema:
    return make_node_schema()
