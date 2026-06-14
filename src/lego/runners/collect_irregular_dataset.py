from __future__ import annotations

import argparse
import logging
import random
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from lego.data.concurrent_dataset_generator import (
    CollectionMetadata,
    CollectionTask,
    ConcurrentDatasetGenerator,
    LocalPostgresConfig,
    ResultWriter,
)
from lego.data.workload_manifest import (
    WorkloadEntry,
    load_job_directory_manifest,
    load_workload_manifest,
)


logger = logging.getLogger(__name__)


@dataclass
class SharedCounter:
    next_index: int = 1
    completed: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)

    def claim(self, total: int) -> int | None:
        with self.lock:
            if self.completed >= total:
                return None
            index = self.next_index
            self.next_index += 1
            self.completed += 1
            return index

    def claim_unbounded(self) -> int:
        with self.lock:
            index = self.next_index
            self.next_index += 1
            self.completed += 1
            return index


@dataclass
class SharedEntryQueue:
    entries: list[WorkloadEntry]
    next_index: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)

    def claim(self) -> tuple[int, WorkloadEntry] | None:
        with self.lock:
            if self.next_index >= len(self.entries):
                return None
            index = self.next_index + 1
            entry = self.entries[self.next_index]
            self.next_index += 1
            return index, entry


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect an irregular LEGO dataset with random per-connection query streams"
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--user", required=True)
    parser.add_argument("--password", default="")
    parser.add_argument("--dbname", required=True)
    parser.add_argument("--statement-timeout-ms", type=int, default=600000)
    parser.add_argument("--workload-file", help="Tab-separated workload manifest or one-query-per-line text file")
    parser.add_argument("--workload-dir", help="Directory containing per-file JOB SQL assets")
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--output-path", help="Defaults to data/<benchmark>/raw/<benchmark>_irregular_<mmdd>.jsonl")
    parser.add_argument("--skipped-output-path", help="Optional JSONL path for failed queries when --continue-on-error is enabled")
    parser.add_argument("--concurrency", type=int, required=True, help="Number of concurrent PostgreSQL connections")
    parser.add_argument(
        "--total-queries",
        type=int,
        default=0,
        help="Total measured records across all connections. Default: number of workload entries.",
    )
    parser.add_argument(
        "--queries-per-connection",
        type=int,
        default=0,
        help="Optional fixed records per connection; overrides --total-queries when positive.",
    )
    parser.add_argument(
        "--workload-repeats",
        type=int,
        default=1,
        help="Coverage mode only: collect every workload query this many times.",
    )
    parser.add_argument(
        "--sample-with-replacement",
        action="store_true",
        help=(
            "Randomly sample workload entries with replacement. By default, "
            "the runner shuffles the workload and collects every query once."
        ),
    )
    parser.add_argument("--limit-queries", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--session-id")
    parser.add_argument("--delay-probability", type=float, default=0.2)
    parser.add_argument("--delay-seconds", type=float, default=10.0)
    parser.add_argument("--max-retries", type=int, default=3, help="Retry each failed query this many times before failing or skipping")
    parser.add_argument("--heat-time-window-min", type=int, default=5)
    parser.add_argument("--disable-table-heat", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--append-output", action="store_true")
    return parser.parse_args()


def _load_entries(args: argparse.Namespace) -> list[WorkloadEntry]:
    if args.workload_dir:
        return load_job_directory_manifest(
            args.workload_dir,
            benchmark=args.benchmark,
            limit=args.limit_queries,
        )
    if args.workload_file:
        return load_workload_manifest(
            args.workload_file,
            benchmark=args.benchmark,
            limit=args.limit_queries,
        )
    raise ValueError("One of --workload-file or --workload-dir is required")


def _default_output_path(benchmark: str, now: datetime | None = None) -> Path:
    stamp = (now or datetime.now()).strftime("%m%d")
    return Path("data") / benchmark / "raw" / f"{benchmark}_irregular_{stamp}.jsonl"


def _validate_args(args: argparse.Namespace) -> None:
    if args.concurrency < 1:
        raise ValueError("--concurrency must be >= 1")
    if args.total_queries < 0:
        raise ValueError("--total-queries must be >= 0")
    if args.queries_per_connection < 0:
        raise ValueError("--queries-per-connection must be >= 0")
    if args.workload_repeats < 1:
        raise ValueError("--workload-repeats must be >= 1")
    if not 0.0 <= args.delay_probability <= 1.0:
        raise ValueError("--delay-probability must be between 0 and 1")
    if args.delay_seconds < 0:
        raise ValueError("--delay-seconds must be >= 0")
    if args.max_retries < 0:
        raise ValueError("--max-retries must be >= 0")


def _default_skipped_output_path(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.stem}_skipped{output_path.suffix}")


def _make_task(
    *,
    entry: WorkloadEntry,
    index: int,
    benchmark: str,
    concurrency_level: str,
    session_id: str,
    worker_slot: int,
    total_slots: int,
    delay_probability: float,
    delay_seconds: float,
    sampling_mode: str,
    workload_repeats: int,
    max_retries: int,
) -> CollectionTask:
    return CollectionTask(
        index=index,
        entry=entry,
        metadata=CollectionMetadata(
            benchmark=benchmark,
            concurrency_level=concurrency_level,
            repeat_id=0,
            session_id=session_id,
            background={
                "n_streams": total_slots,
                "stream_workload": "irregular_same_workload_pool",
                "started_at": datetime.now().isoformat(),
                "collection_policy": f"irregular_random_streams_{sampling_mode}",
                "sampling_mode": sampling_mode,
                "workload_repeats": workload_repeats,
                "delay_probability": delay_probability,
                "delay_seconds": delay_seconds,
                "max_retries": max_retries,
            },
            worker_role="irregular",
            worker_slot=worker_slot,
            total_slots=total_slots,
        ),
    )


def collect_irregular_dataset(
    *,
    db_config: LocalPostgresConfig,
    entries: list[WorkloadEntry],
    output_path: str | Path,
    skipped_output_path: str | Path | None,
    benchmark: str,
    concurrency: int,
    total_queries: int,
    queries_per_connection: int,
    sample_with_replacement: bool,
    workload_repeats: int,
    session_id: str,
    seed: int,
    delay_probability: float,
    delay_seconds: float,
    heat_time_window_min: int,
    max_retries: int,
    collect_table_heat: bool,
    continue_on_error: bool,
    append_output: bool,
) -> None:
    if not entries:
        raise ValueError("Workload is empty")
    writer = ResultWriter(Path(output_path), append=append_output)
    skipped_writer = (
        ResultWriter(Path(skipped_output_path), append=append_output)
        if skipped_output_path is not None
        else None
    )
    counter = SharedCounter()
    scheduled_queue: SharedEntryQueue | None = None
    sampling_mode = "with_replacement" if sample_with_replacement else "without_replacement"
    if not sample_with_replacement:
        scheduled_entries: list[WorkloadEntry] = []
        for _ in range(workload_repeats):
            scheduled_entries.extend(entries)
        schedule_rng = random.Random(seed)
        schedule_rng.shuffle(scheduled_entries)
        scheduled_queue = SharedEntryQueue(scheduled_entries)
        total_queries = len(scheduled_entries)
    failure_event = threading.Event()
    failures: list[str] = []
    failures_lock = threading.Lock()
    concurrency_level = f"s{concurrency}"

    def record_failure(message: str) -> None:
        with failures_lock:
            failures.append(message)
        failure_event.set()
        logger.error(message)

    def write_skipped_record(
        *,
        task: CollectionTask,
        exc: Exception,
        attempts: int,
    ) -> None:
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
                "query_index": task.index,
                "query": task.entry.sql,
                "source_path": task.entry.source_path,
                "attempts": attempts,
                "max_retries": max_retries,
                "error_type": type(exc).__name__,
                "error": repr(exc),
                "timestamp": datetime.now().isoformat(),
                "background": task.metadata.background,
            }
        )

    def worker(worker_id: int) -> None:
        rng = random.Random(seed + worker_id)
        generator = ConcurrentDatasetGenerator(
            db_config,
            thread_id=worker_id,
            enable_table_heat=collect_table_heat,
        )
        local_count = 0
        try:
            while not failure_event.is_set():
                if scheduled_queue is not None:
                    claim = scheduled_queue.claim()
                    if claim is None:
                        return
                    index, entry = claim
                elif queries_per_connection > 0:
                    if local_count >= queries_per_connection:
                        return
                    index = counter.claim_unbounded()
                    entry = rng.choice(entries)
                else:
                    index = counter.claim(total_queries)
                    if index is None:
                        return
                    entry = rng.choice(entries)
                if local_count > 0 and rng.random() < delay_probability:
                    logger.info(
                        "thread=%s status=irregular_delay delay_seconds=%.3f probability=%.3f",
                        worker_id,
                        delay_seconds,
                        delay_probability,
                    )
                    time.sleep(delay_seconds)

                task = _make_task(
                    entry=entry,
                    index=index,
                    benchmark=benchmark,
                    concurrency_level=concurrency_level,
                    session_id=session_id,
                    worker_slot=worker_id,
                    total_slots=concurrency,
                    delay_probability=delay_probability,
                    delay_seconds=delay_seconds,
                    sampling_mode=sampling_mode,
                    workload_repeats=workload_repeats,
                    max_retries=max_retries,
                )
                attempts = 0
                while not failure_event.is_set():
                    try:
                        generator.run_query_with_optional_config(
                            task,
                            writer,
                            vary_config=False,
                            knobs=None,
                            heat_time_window_min=heat_time_window_min,
                            collect_table_heat=collect_table_heat,
                        )
                        local_count += 1
                        break
                    except Exception as exc:
                        attempts += 1
                        try:
                            generator.db.rollback()
                        except Exception:
                            pass
                        logger.exception(
                            "thread=%s status=irregular_query_failed query_index=%s template_id=%s instance_id=%s attempt=%s max_retries=%s",
                            worker_id,
                            index,
                            entry.template_id,
                            entry.instance_id,
                            attempts,
                            max_retries,
                        )
                        if attempts <= max_retries:
                            logger.info(
                                "thread=%s status=irregular_query_retry query_index=%s next_attempt=%s",
                                worker_id,
                                index,
                                attempts + 1,
                            )
                            continue
                        if continue_on_error:
                            write_skipped_record(task=task, exc=exc, attempts=attempts)
                            local_count += 1
                            break
                        record_failure(
                            f"thread={worker_id} query_index={index} template_id={entry.template_id} "
                            f"instance_id={entry.instance_id} attempts={attempts} error={exc!r}"
                        )
                        return
        finally:
            generator.db.close()

    logger.info(
        "status=irregular_collection_start output_path=%s workload_size=%s total_queries=%s queries_per_connection=%s concurrency=%s sampling_mode=%s workload_repeats=%s max_retries=%s continue_on_error=%s delay_probability=%.3f delay_seconds=%.3f",
        output_path,
        len(entries),
        total_queries,
        queries_per_connection,
        concurrency,
        sampling_mode,
        workload_repeats,
        max_retries,
        continue_on_error,
        delay_probability,
        delay_seconds,
    )
    threads = [
        threading.Thread(target=worker, args=(worker_id,), daemon=True)
        for worker_id in range(concurrency)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if failures:
        raise RuntimeError("; ".join(failures))
    logger.info(
        "status=irregular_collection_complete output_path=%s total_queries=%s concurrency=%s sampling_mode=%s",
        output_path,
        total_queries,
        concurrency,
        sampling_mode,
    )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    args = parse_args()
    _validate_args(args)
    entries = _load_entries(args)
    sample_with_replacement = (
        args.sample_with_replacement
        or args.queries_per_connection > 0
        or args.total_queries > 0
    )
    if sample_with_replacement:
        total_queries = (
            args.queries_per_connection * args.concurrency
            if args.queries_per_connection > 0
            else args.total_queries
        )
        if total_queries <= 0:
            total_queries = len(entries)
    else:
        total_queries = len(entries) * args.workload_repeats
    output_path = Path(args.output_path) if args.output_path else _default_output_path(args.benchmark)
    skipped_output_path = (
        Path(args.skipped_output_path)
        if args.skipped_output_path
        else (_default_skipped_output_path(output_path) if args.continue_on_error else None)
    )
    session_id = args.session_id or output_path.stem
    config = LocalPostgresConfig(
        host=args.host,
        port=args.port,
        user=args.user,
        password=args.password,
        dbname=args.dbname,
        statement_timeout_ms=args.statement_timeout_ms,
    )
    collect_irregular_dataset(
        db_config=config,
        entries=entries,
        output_path=output_path,
        skipped_output_path=skipped_output_path,
        benchmark=args.benchmark,
        concurrency=args.concurrency,
        total_queries=total_queries,
        queries_per_connection=args.queries_per_connection,
        sample_with_replacement=sample_with_replacement,
        workload_repeats=args.workload_repeats,
        session_id=session_id,
        seed=args.seed,
        delay_probability=args.delay_probability,
        delay_seconds=args.delay_seconds,
        heat_time_window_min=args.heat_time_window_min,
        max_retries=args.max_retries,
        collect_table_heat=not args.disable_table_heat,
        continue_on_error=args.continue_on_error,
        append_output=args.append_output,
    )
    print(f"collected_queries={total_queries}")
    print(f"output_path={output_path}")
    if skipped_output_path is not None:
        print(f"skipped_output_path={skipped_output_path}")
    print(f"session_id={session_id}")


if __name__ == "__main__":
    main()
