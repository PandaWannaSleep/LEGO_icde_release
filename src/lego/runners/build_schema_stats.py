"""Build schema statistics used by OperatorContextExtractor."""
from __future__ import annotations

import argparse
import json
import logging

import psycopg2

from lego.data.schema_stats import LiveSchemaStatsProvider


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build schema-stats pickle from live PostgreSQL")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--user", required=True)
    p.add_argument("--password", default="")
    p.add_argument("--dbname", required=True)
    p.add_argument("--schema", default="public", help="PostgreSQL schema name (default: public)")
    p.add_argument("--output-path", required=True, help="Path to write the pickle file")
    p.add_argument("--output-json", help="Optional JSON export of the same schema stats payload")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    args = parse_args()

    dsn = (
        f"host={args.host} port={args.port} user={args.user} "
        f"dbname={args.dbname}"
        + (f" password={args.password}" if args.password else "")
    )
    logging.info("Connecting to PostgreSQL: host=%s port=%s dbname=%s", args.host, args.port, args.dbname)
    conn = psycopg2.connect(dsn)
    conn.autocommit = True

    try:
        provider = LiveSchemaStatsProvider(conn, schema=args.schema)
        snapshot = provider.load(args.dbname)
    finally:
        conn.close()

    n_tables = len(snapshot.table_features)
    n_indexes = len(snapshot.index_features)
    n_btree = sum(
        1 for v in snapshot.index_features.values()
        if v.get("index_type") == "btree"
    )
    n_with_height = sum(
        1 for v in snapshot.index_features.values()
        if v.get("tree_height") is not None
    )

    LiveSchemaStatsProvider.save(snapshot, args.output_path)
    if args.output_json:
        from pathlib import Path
        json_output = Path(args.output_json)
        json_output.parent.mkdir(parents=True, exist_ok=True)
        with json_output.open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "config": snapshot.config_info,
                    "table": snapshot.table_features,
                    "index": snapshot.index_features,
                },
                handle,
                indent=2,
            )

    print(f"db={args.dbname}")
    print(f"schema={args.schema}")
    print(f"tables={n_tables}")
    print(f"indexes={n_indexes}  (btree={n_btree}, height_collected={n_with_height})")
    print(f"output_path={args.output_path}")


if __name__ == "__main__":
    main()
