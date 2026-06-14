from __future__ import annotations

import json
import logging
import queue
import random
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import psycopg2
import psycopg2.extras

from .workload_manifest import WorkloadEntry, load_workload_manifest

logger = logging.getLogger(__name__)


def summarize_query(sql: str, max_length: int = 160) -> str:
    collapsed = " ".join(sql.split())
    if len(collapsed) <= max_length:
        return collapsed
    return collapsed[: max_length - 3] + "..."


@dataclass(frozen=True)
class LocalPostgresConfig:
    host: str
    port: int
    user: str
    password: str
    dbname: str
    statement_timeout_ms: int = 600000


@dataclass(frozen=True)
class CollectionMetadata:
    benchmark: str
    concurrency_level: str
    repeat_id: int
    session_id: str
    background: dict[str, Any]
    worker_role: str
    worker_slot: int
    total_slots: int


@dataclass(frozen=True)
class CollectionTask:
    index: int
    entry: WorkloadEntry
    metadata: CollectionMetadata


class LocalPostgresConnector:
    def __init__(self, config: LocalPostgresConfig):
        self.config = config
        self._connection = psycopg2.connect(
            host=config.host,
            port=config.port,
            user=config.user,
            password=config.password,
            dbname=config.dbname,
        )
        self.execute(f"SET statement_timeout = {config.statement_timeout_ms}", set_env=True)

    def execute(self, query: str, set_env: bool = False):
        cursor = self._connection.cursor(cursor_factory=psycopg2.extras.DictCursor)
        cursor.execute(query)
        if not set_env:
            return cursor.fetchall()
        return None

    def execute_no_result(self, sql: str):
        with self._connection.cursor() as cursor:
            cursor.execute(sql)
        self._connection.commit()

    def explain(self, query: str, execute: bool = False, timeout: int = 600000):
        if "explain" not in query.lower():
            prefix = (
                "EXPLAIN (ANALYZE, COSTS, VERBOSE, BUFFERS, FORMAT JSON) "
                if execute
                else "EXPLAIN (COSTS, VERBOSE, FORMAT JSON) "
            )
            query = prefix + query
        self.execute(f"SET statement_timeout = {timeout}", set_env=True)
        cursor = self._connection.cursor(cursor_factory=psycopg2.extras.DictCursor)
        cursor.execute(query)
        rows = cursor.fetchall()
        return rows[0][0][0]

    def rollback(self):
        self._connection.rollback()

    def close(self):
        self._connection.close()


@dataclass
class ResultWriter:
    path: Path
    lock: threading.Lock = threading.Lock()
    append: bool = False

    def __post_init__(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.append:
            self.path.touch(exist_ok=True)
        else:
            self.path.write_text("", encoding="utf-8")

    def write_record(self, record: dict[str, Any]) -> None:
        payload = json.dumps(record, ensure_ascii=False)
        with self.lock:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(payload)
                handle.write("\n")
                handle.flush()


class ConcurrentDatasetGenerator:
    def __init__(self, db_config: LocalPostgresConfig, thread_id: int = 0, *, enable_table_heat: bool = True):
        self.db = LocalPostgresConnector(db_config)
        self.thread_id = thread_id
        self.enable_table_heat = enable_table_heat
        if self.enable_table_heat:
            self.setup_heat_tracking_table()

    def setup_heat_tracking_table(self):
        statements = [
            """
            CREATE TABLE IF NOT EXISTS table_heat_history (
                id SERIAL PRIMARY KEY,
                schema_name TEXT NOT NULL,
                table_name TEXT NOT NULL,
                seq_scan BIGINT DEFAULT 0,
                seq_tup_read BIGINT DEFAULT 0,
                idx_scan BIGINT DEFAULT 0,
                idx_tup_fetch BIGINT DEFAULT 0,
                n_tup_ins BIGINT DEFAULT 0,
                n_tup_upd BIGINT DEFAULT 0,
                n_tup_del BIGINT DEFAULT 0,
                n_tup_hot_upd BIGINT DEFAULT 0,
                n_live_tup BIGINT DEFAULT 0,
                n_dead_tup BIGINT DEFAULT 0,
                vacuum_count BIGINT DEFAULT 0,
                autovacuum_count BIGINT DEFAULT 0,
                analyze_count BIGINT DEFAULT 0,
                autoanalyze_count BIGINT DEFAULT 0,
                recorded_at TIMESTAMP DEFAULT NOW(),
                thread_id INTEGER DEFAULT 0
            )
            """,
            "CREATE INDEX IF NOT EXISTS idx_table_heat_history_time ON table_heat_history(recorded_at DESC)",
            "CREATE INDEX IF NOT EXISTS idx_table_heat_history_table ON table_heat_history(schema_name, table_name)",
            "CREATE INDEX IF NOT EXISTS idx_table_heat_history_table_time ON table_heat_history(schema_name, table_name, recorded_at DESC)",
        ]
        for sql in statements:
            self.db.execute_no_result(sql)

    def record_current_heat_stats(self):
        insert_query = f"""
        INSERT INTO table_heat_history
        (schema_name, table_name, seq_scan, seq_tup_read, idx_scan,
         idx_tup_fetch, n_tup_ins, n_tup_upd, n_tup_del, n_tup_hot_upd,
         n_live_tup, n_dead_tup, vacuum_count, autovacuum_count,
         analyze_count, autoanalyze_count, thread_id)
        SELECT
            schemaname,
            relname,
            COALESCE(seq_scan, 0),
            COALESCE(seq_tup_read, 0),
            COALESCE(idx_scan, 0),
            COALESCE(idx_tup_fetch, 0),
            COALESCE(n_tup_ins, 0),
            COALESCE(n_tup_upd, 0),
            COALESCE(n_tup_del, 0),
            COALESCE(n_tup_hot_upd, 0),
            COALESCE(n_live_tup, 0),
            COALESCE(n_dead_tup, 0),
            COALESCE(vacuum_count, 0),
            COALESCE(autovacuum_count, 0),
            COALESCE(analyze_count, 0),
            COALESCE(autoanalyze_count, 0),
            {self.thread_id}
        FROM pg_stat_user_tables
        """
        self.db.execute_no_result(insert_query)

    def get_recent_heat(self, time_window_minutes: int = 60):
        query = f"""
        WITH time_boundary AS (
            SELECT NOW() - INTERVAL '{time_window_minutes} minutes' AS start_time
        ),
        recent_records AS (
            SELECT schema_name, table_name, seq_scan, seq_tup_read, idx_scan, idx_tup_fetch,
                   n_tup_ins, n_tup_upd, n_tup_del, n_tup_hot_upd,
                   n_live_tup, n_dead_tup, vacuum_count, autovacuum_count,
                   analyze_count, autoanalyze_count, recorded_at,
                   ROW_NUMBER() OVER (PARTITION BY schema_name, table_name ORDER BY recorded_at DESC) AS latest_rn,
                   ROW_NUMBER() OVER (PARTITION BY schema_name, table_name ORDER BY recorded_at ASC) AS earliest_rn
            FROM table_heat_history, time_boundary
            WHERE recorded_at >= time_boundary.start_time
        ),
        latest_stats AS (SELECT * FROM recent_records WHERE latest_rn = 1),
        earliest_stats AS (SELECT * FROM recent_records WHERE earliest_rn = 1),
        current_stats AS (
            SELECT schemaname AS schema_name, relname AS table_name,
                   COALESCE(seq_scan, 0) AS seq_scan,
                   COALESCE(seq_tup_read, 0) AS seq_tup_read,
                   COALESCE(idx_scan, 0) AS idx_scan,
                   COALESCE(idx_tup_fetch, 0) AS idx_tup_fetch,
                   COALESCE(n_tup_ins, 0) AS n_tup_ins,
                   COALESCE(n_tup_upd, 0) AS n_tup_upd,
                   COALESCE(n_tup_del, 0) AS n_tup_del,
                   COALESCE(n_tup_hot_upd, 0) AS n_tup_hot_upd,
                   COALESCE(n_live_tup, 0) AS n_live_tup,
                   COALESCE(n_dead_tup, 0) AS n_dead_tup,
                   COALESCE(vacuum_count, 0) AS vacuum_count,
                   COALESCE(autovacuum_count, 0) AS autovacuum_count,
                   COALESCE(analyze_count, 0) AS analyze_count,
                   COALESCE(autoanalyze_count, 0) AS autoanalyze_count
            FROM pg_stat_user_tables
        )
        SELECT c.schema_name, c.table_name,
               CASE WHEN e.seq_scan IS NOT NULL THEN GREATEST(0, c.seq_scan - e.seq_scan) ELSE c.seq_scan END,
               CASE WHEN e.seq_tup_read IS NOT NULL THEN GREATEST(0, c.seq_tup_read - e.seq_tup_read) ELSE c.seq_tup_read END,
               CASE WHEN e.idx_scan IS NOT NULL THEN GREATEST(0, c.idx_scan - e.idx_scan) ELSE c.idx_scan END,
               CASE WHEN e.idx_tup_fetch IS NOT NULL THEN GREATEST(0, c.idx_tup_fetch - e.idx_tup_fetch) ELSE c.idx_tup_fetch END,
               CASE WHEN e.n_tup_ins IS NOT NULL THEN GREATEST(0, c.n_tup_ins - e.n_tup_ins) ELSE c.n_tup_ins END,
               CASE WHEN e.n_tup_upd IS NOT NULL THEN GREATEST(0, c.n_tup_upd - e.n_tup_upd) ELSE c.n_tup_upd END,
               CASE WHEN e.n_tup_del IS NOT NULL THEN GREATEST(0, c.n_tup_del - e.n_tup_del) ELSE c.n_tup_del END,
               CASE WHEN e.n_tup_hot_upd IS NOT NULL THEN GREATEST(0, c.n_tup_hot_upd - e.n_tup_hot_upd) ELSE c.n_tup_hot_upd END,
               CASE WHEN e.vacuum_count IS NOT NULL THEN GREATEST(0, c.vacuum_count - e.vacuum_count) ELSE c.vacuum_count END,
               CASE WHEN e.autovacuum_count IS NOT NULL THEN GREATEST(0, c.autovacuum_count - e.autovacuum_count) ELSE c.autovacuum_count END,
               CASE WHEN e.analyze_count IS NOT NULL THEN GREATEST(0, c.analyze_count - e.analyze_count) ELSE c.analyze_count END,
               CASE WHEN e.autoanalyze_count IS NOT NULL THEN GREATEST(0, c.autoanalyze_count - e.autoanalyze_count) ELSE c.autoanalyze_count END,
               c.n_live_tup, c.n_dead_tup,
               l.recorded_at, e.recorded_at
        FROM current_stats c
        LEFT JOIN latest_stats l ON c.schema_name = l.schema_name AND c.table_name = l.table_name
        LEFT JOIN earliest_stats e ON c.schema_name = e.schema_name AND c.table_name = e.table_name
        ORDER BY c.schema_name, c.table_name
        """
        rows = self.db.execute(query)
        timestamp = datetime.now().isoformat()
        metrics = {}
        for row in rows:
            key = f"{row[0]}.{row[1]}"
            metrics[key] = {
                "schema": row[0],
                "table_name": row[1],
                "recent_seq_scan": row[2] or 0,
                "recent_seq_tup_read": row[3] or 0,
                "recent_idx_scan": row[4] or 0,
                "recent_idx_tup_fetch": row[5] or 0,
                "recent_n_tup_ins": row[6] or 0,
                "recent_n_tup_upd": row[7] or 0,
                "recent_n_tup_del": row[8] or 0,
                "recent_n_tup_hot_upd": row[9] or 0,
                "recent_vacuum_count": row[10] or 0,
                "recent_autovacuum_count": row[11] or 0,
                "recent_analyze_count": row[12] or 0,
                "recent_autoanalyze_count": row[13] or 0,
                "current_n_live_tup": row[14] or 0,
                "current_n_dead_tup": row[15] or 0,
                "latest_record_time": row[16].isoformat() if row[16] else None,
                "earliest_record_time": row[17].isoformat() if row[17] else None,
                "timestamp": timestamp,
                "requested_time_window_minutes": time_window_minutes,
            }
        return metrics

    def run_query(
        self,
        task: CollectionTask,
        writer: ResultWriter,
        config: dict[str, str],
        *,
        heat_time_window_min: int = 5,
        collect_table_heat: bool = True,
    ) -> dict[str, Any]:
        started_at = time.time()
        query_preview = summarize_query(task.entry.sql)
        logger.info(
            "thread=%s status=query_start query_index=%s template_id=%s instance_id=%s repeat_id=%s concurrency=%s sql=%s",
            self.thread_id,
            task.index,
            task.entry.template_id,
            task.entry.instance_id,
            task.metadata.repeat_id,
            task.metadata.concurrency_level,
            query_preview,
        )
        table_heat_metrics = (
            self.get_recent_heat(time_window_minutes=heat_time_window_min)
            if collect_table_heat
            else {}
        )
        planinfo = self.db.explain(task.entry.sql, execute=True, timeout=self.db.config.statement_timeout_ms)
        record = {
            "benchmark": task.entry.benchmark,
            "template_id": task.entry.template_id,
            "instance_id": task.entry.instance_id,
            "concurrency_level": task.metadata.concurrency_level,
            "repeat_id": task.metadata.repeat_id,
            "session_id": task.metadata.session_id,
            "background": task.metadata.background,
            "worker_role": task.metadata.worker_role,
            "worker_slot": task.metadata.worker_slot,
            "total_slots": task.metadata.total_slots,
            "planinfo": planinfo,
            "query": task.entry.sql,
            "config": {
                "statement_timeout_ms": self.db.config.statement_timeout_ms,
                **config,
            },
            "table_heat_metrics": table_heat_metrics,
            "thread_id": self.thread_id,
            "timestamp": datetime.now().isoformat(),
            "source_path": task.entry.source_path,
        }
        writer.write_record(record)
        elapsed = time.time() - started_at
        logger.info(
            "thread=%s status=query_done query_index=%s elapsed_sec=%.3f template_id=%s instance_id=%s repeat_id=%s concurrency=%s sql=%s",
            self.thread_id,
            task.index,
            elapsed,
            task.entry.template_id,
            task.entry.instance_id,
            task.metadata.repeat_id,
            task.metadata.concurrency_level,
            query_preview,
        )
        return record

    def run_query_with_optional_config(
        self,
        task: CollectionTask,
        writer: ResultWriter,
        *,
        vary_config: bool = False,
        knobs: dict | None = None,
        step: int = 3,
        heat_time_window_min: int = 5,
        collect_table_heat: bool = True,
    ) -> dict[str, Any]:
        config = {}
        if vary_config and knobs:
            n_steps = np.arange(step) / step
            bool_step = [0, 1]
            for name, knob in knobs.items():
                if knob.get("type") == "bool":
                    value = random.choice(bool_step)
                else:
                    value = random.choice(n_steps)
                applied = knob["to_string"](value) if callable(knob.get("to_string")) else str(value)
                self.db.execute(f"set {name}={applied}", set_env=True)
                config[name] = applied
        return self.run_query(
            task,
            writer,
            config,
            heat_time_window_min=heat_time_window_min,
            collect_table_heat=collect_table_heat,
        )


def load_workload_queries(path: str | Path, limit: int = 0) -> list[str]:
    return [entry.sql for entry in load_workload_manifest(path, benchmark="unknown", limit=limit)]


def _build_collection_tasks(
    entries: list[WorkloadEntry],
    *,
    concurrency_level: str,
    repeats: int,
    session_id: str,
    total_slots: int,
    background_workload_label: str,
    collection_policy: str,
) -> list[CollectionTask]:
    tasks: list[CollectionTask] = []
    task_index = 1
    background = {
        "n_streams": max(total_slots - 1, 0),
        "stream_workload": background_workload_label,
        "started_at": datetime.now().isoformat(),
        "collection_policy": collection_policy,
    }
    for repeat_id in range(repeats):
        for entry in entries:
            tasks.append(
                CollectionTask(
                    index=task_index,
                    entry=entry,
                    metadata=CollectionMetadata(
                        benchmark=entry.benchmark,
                        concurrency_level=concurrency_level,
                        repeat_id=repeat_id,
                        session_id=session_id,
                        background=background,
                        worker_role="measured",
                        worker_slot=0,
                        total_slots=total_slots,
                    ),
                )
            )
            task_index += 1
    return tasks


def _partition_tasks_round_robin(tasks: list[CollectionTask], shard_count: int) -> list[list[CollectionTask]]:
    if shard_count <= 0:
        raise ValueError("shard_count must be >= 1")
    shards: list[list[CollectionTask]] = [[] for _ in range(shard_count)]
    for index, task in enumerate(tasks):
        shard_id = index % shard_count
        shards[shard_id].append(
            CollectionTask(
                index=task.index,
                entry=task.entry,
                metadata=CollectionMetadata(
                    benchmark=task.metadata.benchmark,
                    concurrency_level=task.metadata.concurrency_level,
                    repeat_id=task.metadata.repeat_id,
                    session_id=task.metadata.session_id,
                    background=task.metadata.background,
                    worker_role="measured",
                    worker_slot=shard_id,
                    total_slots=shard_count,
                ),
            )
        )
    return shards


def _make_background_task(
    *,
    entry: WorkloadEntry,
    concurrency_level: str,
    session_id: str,
    total_slots: int,
    worker_slot: int,
    collection_policy: str,
) -> CollectionTask:
    return CollectionTask(
        index=0,
        entry=entry,
        metadata=CollectionMetadata(
            benchmark=entry.benchmark,
            concurrency_level=concurrency_level,
            repeat_id=-1,
            session_id=session_id,
            background={
                "n_streams": max(total_slots - 1, 0),
                "stream_workload": "same_workload_pool",
                "started_at": datetime.now().isoformat(),
                "collection_policy": collection_policy,
            },
            worker_role="background_drain",
            worker_slot=worker_slot,
            total_slots=total_slots,
        ),
    )


def _start_heat_daemon(
    *,
    db_config: LocalPostgresConfig,
    heat_interval_min: int,
    failure_event: threading.Event,
    record_failure,
) -> threading.Thread:
    def heat_daemon() -> None:
        try:
            recorder = ConcurrentDatasetGenerator(db_config, thread_id=-1)
            while not failure_event.is_set():
                recorder.record_current_heat_stats()
                logger.info("thread=%s status=heat_snapshot_recorded", -1)
                time.sleep(heat_interval_min * 60)
        except Exception as exc:  # pragma: no cover - hard to deterministically trigger
            record_failure(f"heat_daemon_failed error={exc!r}")

    daemon = threading.Thread(target=heat_daemon, daemon=True)
    daemon.start()
    return daemon


def run_concurrent_queries(
    db_config: LocalPostgresConfig,
    queries: list[str],
    save_file: str | Path,
    num_threads: int = 1,
    vary_config: bool = False,
    knobs: dict | None = None,
    step: int = 3,
    shuffle_queries: bool = True,
    heat_interval_min: int = 1,
    heat_time_window_min: int = 5,
    restart_each_query: bool = False,
    stop_on_error: bool = True,
    max_background_records_per_worker: int = 0,
    append_output: bool = False,
    collect_table_heat: bool = True,
):
    entries = [
        WorkloadEntry(
            benchmark="unknown",
            template_id="unknown",
            instance_id=str(i),
            sql=sql,
            source_path="legacy_workload",
        )
        for i, sql in enumerate(queries, start=1)
    ]
    return run_paired_concurrency(
        db_config=db_config,
        entries=entries,
        save_file=save_file,
        background_save_file=None,
        concurrency_level="s1",
        repeats=1,
        session_id=Path(save_file).stem,
        total_slots=max(1, num_threads),
        background_entries=entries,
        collection_mode="measured_shards",
        vary_config=vary_config,
        knobs=knobs,
        step=step,
        shuffle_queries=shuffle_queries,
        heat_interval_min=heat_interval_min,
        heat_time_window_min=heat_time_window_min,
        restart_each_query=restart_each_query,
        stop_on_error=stop_on_error,
        max_background_records_per_worker=max_background_records_per_worker,
        append_output=append_output,
        collect_table_heat=collect_table_heat,
    )


def run_paired_concurrency(
    *,
    db_config: LocalPostgresConfig,
    entries: list[WorkloadEntry],
    save_file: str | Path,
    background_save_file: str | Path | None,
    concurrency_level: str,
    repeats: int,
    session_id: str,
    total_slots: int,
    background_entries: list[WorkloadEntry] | None,
    collection_mode: str = "measured_stream_with_background",
    vary_config: bool = False,
    knobs: dict | None = None,
    step: int = 3,
    shuffle_queries: bool = False,
    heat_interval_min: int = 1,
    heat_time_window_min: int = 5,
    restart_each_query: bool = False,
    stop_on_error: bool = True,
    max_background_records_per_worker: int = 0,
    continue_on_measured_failure: bool = False,
    skipped_save_file: str | Path | None = None,
    append_output: bool = False,
    collect_table_heat: bool = True,
) -> None:
    if total_slots < 1:
        raise ValueError("total_slots must be >= 1")
    if collection_mode not in {"measured_stream_with_background", "measured_shards"}:
        raise ValueError(
            "collection_mode must be 'measured_stream_with_background' or 'measured_shards'"
        )
    main_entries = entries.copy()
    if shuffle_queries:
        random.shuffle(main_entries)
    collection_policy = (
        "measured_shards_then_background_drain"
        if collection_mode == "measured_shards"
        else "single_measured_stream_with_background"
    )
    main_tasks = _build_collection_tasks(
        main_entries,
        concurrency_level=concurrency_level,
        repeats=repeats,
        session_id=session_id,
        total_slots=total_slots,
        background_workload_label="same_workload_pool" if background_entries else "none",
        collection_policy=collection_policy,
    )
    if collection_mode == "measured_shards":
        measured_task_shards = _partition_tasks_round_robin(main_tasks, shard_count=total_slots)
    else:
        measured_task_shards = [main_tasks] + [[] for _ in range(total_slots - 1)]
    writer = ResultWriter(Path(save_file), append=append_output)
    background_writer = ResultWriter(Path(background_save_file), append=append_output) if background_save_file else None
    skipped_writer = ResultWriter(Path(skipped_save_file), append=append_output) if skipped_save_file else None

    logger.info(
        "status=collection_start output_path=%s total_queries=%s repeats=%s concurrency_level=%s total_slots=%s restart_each_query=%s shuffle_queries=%s heat_interval_min=%s heat_time_window_min=%s",
        save_file,
        len(main_entries),
        repeats,
        concurrency_level,
        total_slots,
        restart_each_query,
        shuffle_queries,
        heat_interval_min,
        heat_time_window_min,
    )

    failure_event = threading.Event()
    failure_messages: list[str] = []
    failure_lock = threading.Lock()

    def record_failure(message: str) -> None:
        with failure_lock:
            failure_messages.append(message)
        failure_event.set()
        logger.error(message)

    if collect_table_heat:
        _start_heat_daemon(
            db_config=db_config,
            heat_interval_min=heat_interval_min,
            failure_event=failure_event,
            record_failure=record_failure,
        )
        time.sleep(2)

    completed_shards = {"count": 0}
    completed_lock = threading.Lock()
    measured_all_done = threading.Event()

    def mark_measured_shard_complete(worker_id: int) -> bool:
        with completed_lock:
            completed_shards["count"] += 1
            remaining = total_slots - completed_shards["count"]
            logger.info(
                "thread=%s status=measured_shard_complete completed_shards=%s total_slots=%s remaining_shards=%s",
                worker_id,
                completed_shards["count"],
                total_slots,
                max(remaining, 0),
            )
            if remaining <= 0:
                measured_all_done.set()
                return True
        return False

    def record_skipped_measured(task: CollectionTask, exc: Exception) -> None:
        if skipped_writer is None:
            return
        skipped_writer.write_record(
            {
                "benchmark": task.entry.benchmark,
                "template_id": task.entry.template_id,
                "instance_id": task.entry.instance_id,
                "concurrency_level": task.metadata.concurrency_level,
                "repeat_id": task.metadata.repeat_id,
                "session_id": task.metadata.session_id,
                "worker_role": task.metadata.worker_role,
                "worker_slot": task.metadata.worker_slot,
                "total_slots": task.metadata.total_slots,
                "query": task.entry.sql,
                "source_path": task.entry.source_path,
                "error_type": type(exc).__name__,
                "error": repr(exc),
                "timestamp": datetime.now().isoformat(),
            }
        )

    def shard_worker(worker_id: int, shard_tasks: list[CollectionTask]) -> None:
        generator = ConcurrentDatasetGenerator(db_config, thread_id=worker_id, enable_table_heat=collect_table_heat)
        rng = random.Random(worker_id)
        local_entries = background_entries.copy() if background_entries else []
        written_records = 0
        try:
            for task in shard_tasks:
                if failure_event.is_set():
                    break
                query_preview = summarize_query(task.entry.sql)
                try:
                    logger.info(
                        "thread=%s status=worker_dispatch query_index=%s shard_queries=%s total_queries=%s template_id=%s instance_id=%s repeat_id=%s concurrency=%s sql=%s",
                        generator.thread_id,
                        task.index,
                        len(shard_tasks),
                        len(main_tasks),
                        task.entry.template_id,
                        task.entry.instance_id,
                        task.metadata.repeat_id,
                        task.metadata.concurrency_level,
                        query_preview,
                    )
                    generator.run_query_with_optional_config(
                        task,
                        writer,
                        vary_config=vary_config,
                        knobs=knobs,
                        step=step,
                        heat_time_window_min=heat_time_window_min,
                        collect_table_heat=collect_table_heat,
                    )
                    logger.info(
                        "thread=%s status=worker_complete query_index=%s shard_queries=%s total_queries=%s template_id=%s instance_id=%s repeat_id=%s concurrency=%s sql=%s",
                        generator.thread_id,
                        task.index,
                        len(shard_tasks),
                        len(main_tasks),
                        task.entry.template_id,
                        task.entry.instance_id,
                        task.metadata.repeat_id,
                        task.metadata.concurrency_level,
                        query_preview,
                    )
                    if restart_each_query:
                        generator.db.close()
                        generator.db = LocalPostgresConnector(db_config)
                        if collect_table_heat:
                            generator.setup_heat_tracking_table()
                        logger.info(
                            "thread=%s status=connection_restarted query_index=%s",
                            generator.thread_id,
                            task.index,
                        )
                except Exception as exc:
                    try:
                        generator.db.rollback()
                    except Exception as rollback_exc:  # pragma: no cover - defensive
                        logger.warning(
                            "thread=%s status=rollback_failed query_index=%s error=%r",
                            generator.thread_id,
                            task.index,
                            rollback_exc,
                        )
                    logger.exception(
                        "thread=%s status=worker_failed query_index=%s total_queries=%s template_id=%s instance_id=%s repeat_id=%s",
                        generator.thread_id,
                        task.index,
                        len(main_tasks),
                        task.entry.template_id,
                        task.entry.instance_id,
                        task.metadata.repeat_id,
                    )
                    if continue_on_measured_failure:
                        logger.warning(
                            "thread=%s status=measured_failure_skipped query_index=%s template_id=%s instance_id=%s repeat_id=%s error=%r",
                            generator.thread_id,
                            task.index,
                            task.entry.template_id,
                            task.entry.instance_id,
                            task.metadata.repeat_id,
                            exc,
                        )
                        record_skipped_measured(task, exc)
                        continue
                    record_failure(
                        f"thread={generator.thread_id} query_index={task.index} total_queries={len(main_tasks)} template_id={task.entry.template_id} instance_id={task.entry.instance_id} repeat_id={task.metadata.repeat_id} error={exc!r}"
                    )
                    if stop_on_error:
                        return
                finally:
                    time.sleep(random.uniform(0.5, 1.0))

            shard_is_last = mark_measured_shard_complete(worker_id)
            if collection_mode == "measured_stream_with_background" and shard_tasks:
                return
            if shard_is_last or not local_entries:
                return

            logger.info(
                "thread=%s status=background_drain_start max_background_records_per_worker=%s",
                generator.thread_id,
                max_background_records_per_worker,
            )
            while not failure_event.is_set() and not measured_all_done.is_set():
                rng.shuffle(local_entries)
                for entry in local_entries:
                    if failure_event.is_set() or measured_all_done.is_set():
                        break
                    if max_background_records_per_worker and written_records >= max_background_records_per_worker:
                        logger.info(
                            "thread=%s status=background_cap_reached max_background_records_per_worker=%s",
                            generator.thread_id,
                            max_background_records_per_worker,
                        )
                        return
                    background_task = _make_background_task(
                        entry=entry,
                        concurrency_level=concurrency_level,
                        session_id=session_id,
                        total_slots=total_slots,
                        worker_slot=worker_id,
                        collection_policy=collection_policy,
                    )
                    try:
                        generator.run_query_with_optional_config(
                            background_task,
                            background_writer or writer,
                            vary_config=vary_config,
                            knobs=knobs,
                            step=step,
                            heat_time_window_min=heat_time_window_min,
                            collect_table_heat=collect_table_heat,
                        )
                        written_records += 1
                    except Exception as exc:
                        try:
                            generator.db.rollback()
                        except Exception:
                            pass
                        logger.exception(
                            "thread=%s status=background_worker_failed template_id=%s instance_id=%s",
                            generator.thread_id,
                            entry.template_id,
                            entry.instance_id,
                        )
                        if continue_on_measured_failure:
                            logger.warning(
                                "thread=%s status=background_failure_skipped template_id=%s instance_id=%s error=%r",
                                generator.thread_id,
                                entry.template_id,
                                entry.instance_id,
                                exc,
                            )
                            if skipped_writer is not None:
                                skipped_writer.write_record(
                                    {
                                        "benchmark": entry.benchmark,
                                        "template_id": entry.template_id,
                                        "instance_id": entry.instance_id,
                                        "concurrency_level": concurrency_level,
                                        "repeat_id": -1,
                                        "session_id": session_id,
                                        "worker_role": "background_drain",
                                        "worker_slot": worker_id,
                                        "total_slots": total_slots,
                                        "query": entry.sql,
                                        "source_path": entry.source_path,
                                        "error_type": type(exc).__name__,
                                        "error": repr(exc),
                                        "timestamp": datetime.now().isoformat(),
                                    }
                                )
                            continue
                        with completed_lock:
                            measured_finished = completed_shards["count"] >= total_slots
                        if measured_finished:
                            logger.warning(
                                "thread=%s status=background_failure_ignored_after_measured_completion template_id=%s instance_id=%s error=%r",
                                generator.thread_id,
                                entry.template_id,
                                entry.instance_id,
                                exc,
                            )
                            return
                        record_failure(
                            f"thread={generator.thread_id} background_template_id={entry.template_id} background_instance_id={entry.instance_id} error={exc!r}"
                        )
                        if stop_on_error:
                            return
                    finally:
                        time.sleep(random.uniform(0.1, 0.4))
        finally:
            generator.db.close()

    worker_ids = list(range(total_slots))
    if collection_mode == "measured_stream_with_background":
        worker_ids = list(range(1, total_slots)) + [0]
    threads = [
        threading.Thread(target=shard_worker, args=(worker_id, measured_task_shards[worker_id]), daemon=True)
        for worker_id in worker_ids
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    if failure_messages:
        raise RuntimeError("; ".join(failure_messages))

    logger.info(
        "status=collection_complete output_path=%s total_queries=%s repeats=%s concurrency_level=%s",
        save_file,
        len(main_entries),
        repeats,
        concurrency_level,
    )
