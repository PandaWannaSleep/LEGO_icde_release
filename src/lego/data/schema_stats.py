"""Schema-statistics interfaces and cache-backed providers."""

from __future__ import annotations

import json
import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SchemaStatsSnapshot:
    db_name: str
    loaded_from_cache: bool = False
    config_info: dict[str, Any] = field(default_factory=dict)
    table_features: dict[str, Any] = field(default_factory=dict)
    index_features: dict[str, Any] = field(default_factory=dict)

    def process_config(self, config: dict[str, Any]) -> dict[str, Any]:
        processed: dict[str, Any] = {}
        for setting, value in config.items():
            try:
                processed[setting] = int(value)
            except (TypeError, ValueError):
                if value in {"on", "off"}:
                    processed[setting] = 1 if value == "on" else 0
                else:
                    processed[setting] = value
        return processed

    def get_column_info(self, column: str) -> dict[str, Any]:
        if "." in column:
            table_name, column_name = column.split(".", 1)
            table_name = table_name.replace('"', "")
            column_name = column_name.replace('"', "")
            table = self.table_features.get(table_name)
            if table and column_name in table.get("columns", {}):
                return {
                    "table": table_name,
                    "column": column_name,
                    **table["columns"][column_name],
                }

        longest_length = 0
        matched: dict[str, Any] | None = None
        for table_name, table in self.table_features.items():
            for column_name, info in table.get("columns", {}).items():
                if column_name == column:
                    return {"table": table_name, "column": column_name, **info}
                if column_name in column and len(column_name) > longest_length:
                    matched = {"table": table_name, "column": column_name, **info}
                    longest_length = len(column_name)
        return matched or {}


class SchemaStatsProvider:
    def load(self, db_name: str) -> SchemaStatsSnapshot:
        raise NotImplementedError


class SchemaStatsFileProvider(SchemaStatsProvider):
    """Load schema statistics from an explicit cache artifact path.

    Supports either the legacy pickle payload:

        {"config": ..., "table": ..., "index": ...}

    or a JSON file with the same three top-level keys.
    """

    def __init__(self, cache_path: str | Path):
        self.cache_path = Path(cache_path)

    def load(self, db_name: str) -> SchemaStatsSnapshot:
        if not self.cache_path.exists():
            raise FileNotFoundError(f"Schema stats cache not found: {self.cache_path}")

        if self.cache_path.suffix.lower() == ".json":
            with self.cache_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        else:
            with self.cache_path.open("rb") as handle:
                payload = pickle.load(handle)

        if not isinstance(payload, dict):
            raise ValueError(f"Schema stats cache must deserialize to a dict: {self.cache_path}")

        return SchemaStatsSnapshot(
            db_name=db_name,
            loaded_from_cache=True,
            config_info=payload.get("config", {}),
            table_features=payload.get("table", {}),
            index_features=payload.get("index", {}),
        )


class LegacyCacheSchemaStatsProvider(SchemaStatsProvider):
    """
    Load schema statistics from the legacy artifact cache.

    This preserves DB-aware feature enrichment without reintroducing the old
    `Plan_class` as the central abstraction in the new code.
    """

    def __init__(self, legacy_root: str | Path):
        self.legacy_root = Path(legacy_root)

    def load(self, db_name: str) -> SchemaStatsSnapshot:
        cache_path = (
            self.legacy_root
            / "data"
            / "temporary"
            / "schemeinfo"
            / f"scheme_{db_name}_histogram_info.pickle"
        )
        if not cache_path.exists():
            raise FileNotFoundError(f"Legacy schema cache not found: {cache_path}")

        with cache_path.open("rb") as handle:
            payload = pickle.load(handle)

        return SchemaStatsSnapshot(
            db_name=db_name,
            loaded_from_cache=True,
            config_info=payload.get("config", {}),
            table_features=payload.get("table", {}),
            index_features=payload.get("index", {}),
        )


def load_schema_stats(
    *,
    db_name: str,
    schema_cache: str | Path | None = None,
    legacy_root: str | Path | None = None,
) -> SchemaStatsSnapshot:
    if schema_cache:
        return SchemaStatsFileProvider(schema_cache).load(db_name)
    if legacy_root:
        return LegacyCacheSchemaStatsProvider(legacy_root).load(db_name)
    raise ValueError("Either schema_cache or legacy_root must be provided to load schema statistics")


class LiveSchemaStatsProvider(SchemaStatsProvider):
    """Query a live PostgreSQL instance to build a SchemaStatsSnapshot.

    Requires the ``pageinspect`` extension (``CREATE EXTENSION IF NOT EXISTS
    pageinspect;``) for B-tree index height. Non-B-tree indexes get
    ``tree_height=None`` without raising an error.

    Usage::

        import psycopg2
        conn = psycopg2.connect(...)
        provider = LiveSchemaStatsProvider(conn, schema="public")
        snapshot = provider.load("imdbload")
        provider.save(snapshot, Path("scheme_imdbload_histogram_info.pickle"))
    """

    def __init__(self, conn: Any, schema: str = "public"):
        self.conn = conn
        self.schema = schema

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self, db_name: str) -> SchemaStatsSnapshot:
        table_features = self._collect_table_features()
        index_features = self._collect_index_features()
        return SchemaStatsSnapshot(
            db_name=db_name,
            loaded_from_cache=False,
            table_features=table_features,
            index_features=index_features,
        )

    @staticmethod
    def save(snapshot: SchemaStatsSnapshot, output_path: str | Path) -> None:
        """Persist a snapshot to the legacy pickle format expected by
        ``LegacyCacheSchemaStatsProvider``."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "table": snapshot.table_features,
            "index": snapshot.index_features,
            "config": snapshot.config_info,
        }
        with output_path.open("wb") as fh:
            pickle.dump(payload, fh)
        logger.info("schema_stats saved to %s", output_path)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _execute(self, sql: str, params: tuple = ()) -> list[tuple]:
        cur = self.conn.cursor()
        try:
            cur.execute(sql, params)
            return cur.fetchall()
        finally:
            cur.close()

    def _collect_table_features(self) -> dict[str, Any]:
        """Return per-table dict with relpages, reltuples, and per-column stats."""
        # Table-level sizes
        size_rows = self._execute(
            """
            SELECT c.relname, c.relpages, c.reltuples
            FROM pg_class c
            JOIN pg_namespace n ON n.oid = c.relnamespace
            WHERE n.nspname = %s AND c.relkind = 'r'
            ORDER BY c.relname
            """,
            (self.schema,),
        )

        # Column-level stats from pg_stats (one row per table.column)
        col_rows = self._execute(
            """
            SELECT tablename, attname, n_distinct, correlation, null_frac,
                   avg_width, data_type
            FROM pg_stats
            JOIN information_schema.columns
                ON table_schema = schemaname
               AND table_name  = tablename
               AND column_name = attname
            WHERE schemaname = %s
            ORDER BY tablename, attname
            """,
            (self.schema,),
        )

        # Build column map: {table: {col: {n_distinct, correlation, ...}}}
        col_map: dict[str, dict[str, Any]] = {}
        for tablename, attname, n_distinct, correlation, null_frac, avg_width, dtype in col_rows:
            col_map.setdefault(tablename, {})[attname] = {
                "n_distinct": float(n_distinct) if n_distinct is not None else None,
                "correlation": float(correlation) if correlation is not None else None,
                "null_frac": float(null_frac) if null_frac is not None else None,
                "avg_width": int(avg_width) if avg_width is not None else None,
                "type": dtype,
                # offset: legacy field used by extractor's _infer_column_info
                "offset": 0.0,
            }

        features: dict[str, Any] = {}
        for relname, relpages, reltuples in size_rows:
            features[relname] = {
                "table_pages": int(relpages or 0),
                "tuple_num": float(reltuples or 0.0),
                "columns": col_map.get(relname, {}),
            }
        return features

    def _collect_index_features(self) -> dict[str, Any]:
        """Return per-index dict with tree_height, pages, indexCorrelation, distinctnum."""
        # All B-tree indexes on user tables in the target schema
        idx_rows = self._execute(
            """
            SELECT
                ci.relname                        AS index_name,
                ct.relname                        AS table_name,
                ci.relpages                       AS pages,
                ix.indkey,
                am.amname                         AS index_type
            FROM pg_index ix
            JOIN pg_class ci ON ci.oid = ix.indexrelid
            JOIN pg_class ct ON ct.oid = ix.indrelid
            JOIN pg_namespace n ON n.oid = ct.relnamespace
            JOIN pg_am am ON am.oid = ci.relam
            WHERE n.nspname = %s AND ci.relkind = 'i'
            ORDER BY ci.relname
            """,
            (self.schema,),
        )

        # First indexed column per index (for pg_stats lookup)
        att_rows = self._execute(
            """
            SELECT
                ci.relname   AS index_name,
                a.attname    AS column_name
            FROM pg_index ix
            JOIN pg_class ci ON ci.oid = ix.indexrelid
            JOIN pg_class ct ON ct.oid = ix.indrelid
            JOIN pg_namespace n ON n.oid = ct.relnamespace
            JOIN pg_attribute a
                ON a.attrelid = ct.oid
               AND a.attnum   = ix.indkey[1]   -- first key column
            WHERE n.nspname = %s AND ci.relkind = 'i'
            ORDER BY ci.relname
            """,
            (self.schema,),
        )
        idx_first_col: dict[str, str] = {row[0]: row[1] for row in att_rows}

        features: dict[str, Any] = {}
        for index_name, table_name, pages, _indkey, index_type in idx_rows:
            tree_height: int | None = None
            if index_type == "btree":
                try:
                    rows = self._execute(
                        "SELECT level FROM bt_metap(%s)",
                        (f"{self.schema}.{index_name}",),
                    )
                    if rows:
                        tree_height = int(rows[0][0])
                except Exception as exc:
                    # bt_metap fails on empty indexes; degrade gracefully
                    logger.debug("bt_metap(%s) failed: %s", index_name, exc)

            # Correlation and n_distinct from pg_stats for first indexed column
            col_name = idx_first_col.get(index_name)
            correlation: float | None = None
            n_distinct: float | None = None
            if col_name:
                stat_rows = self._execute(
                    """
                    SELECT correlation, n_distinct
                    FROM pg_stats
                    WHERE schemaname = %s AND tablename = %s AND attname = %s
                    """,
                    (self.schema, table_name, col_name),
                )
                if stat_rows:
                    r = stat_rows[0]
                    correlation = float(r[0]) if r[0] is not None else None
                    n_distinct = float(r[1]) if r[1] is not None else None

            features[index_name] = {
                "tree_height": tree_height,
                "pages": int(pages or 0),
                "indexCorrelation": correlation,
                "distinctnum": n_distinct,
                "table": table_name,
                "index_type": index_type,
            }

        logger.info(
            "Collected stats for %d tables, %d indexes (schema=%s)",
            0,  # populated after return in caller
            len(features),
            self.schema,
        )
        return features
