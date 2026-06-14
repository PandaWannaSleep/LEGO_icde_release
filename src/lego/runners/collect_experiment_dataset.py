from __future__ import annotations

import argparse
import logging
from pathlib import Path
import re

from lego.data.concurrent_dataset_generator import (
    LocalPostgresConfig,
    load_workload_queries,
    run_concurrent_queries,
    run_paired_concurrency,
)
from lego.data.workload_manifest import load_job_directory_manifest, load_workload_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect a LEGO training dataset from a local PostgreSQL instance")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--user", required=True)
    parser.add_argument("--password", default="")
    parser.add_argument("--dbname", required=True)
    parser.add_argument("--statement-timeout-ms", type=int, default=600000, help="Per-query statement timeout in milliseconds")
    parser.add_argument("--workload-file", help="Tab-separated workload manifest or one-query-per-line text file")
    parser.add_argument("--workload-dir", help="Directory containing per-file JOB SQL assets")
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--background-output-path", help="Optional background audit JSONL path")
    parser.add_argument("--skipped-output-path", help="Optional JSONL path for skipped measured queries when continue_on_measured_failure is enabled")
    parser.add_argument("--num-threads", type=int, default=1)
    parser.add_argument("--limit-queries", type=int, default=0)
    parser.add_argument("--shuffle-queries", action="store_true")
    parser.add_argument("--heat-interval-min", type=int, default=1)
    parser.add_argument("--heat-time-window-min", type=int, default=5)
    parser.add_argument("--disable-table-heat", action="store_true", help="Skip table_heat_history snapshots and write empty table_heat_metrics")
    parser.add_argument("--restart-each-query", action="store_true", help="Restart the database session after each query; only takes effect for measured stream")
    parser.add_argument("--benchmark", default="job")
    parser.add_argument(
        "--collection-mode",
        choices=("measured_stream_with_background", "measured_shards"),
        default="measured_stream_with_background",
        help=(
            "measured_stream_with_background runs one measured stream plus "
            "concurrency-1 background streams. measured_shards preserves the "
            "older mode where measured queries are sharded across all slots."
        ),
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        help="Total concurrent worker slots. For example, --concurrency 8 records concurrency_level=s8.",
    )
    parser.add_argument(
        "--concurrency-level",
        default=None,
        help="Legacy concurrency label such as s1, s4, or s8. Any sN label is accepted.",
    )
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--background-streams", type=int, help="Deprecated alias retained for older invocations; ignored by shard mode unless --measured-shards is omitted and concurrency_level is s1")
    parser.add_argument("--measured-shards", type=int, help="Override the number of measured shards / worker groups; defaults to s1=1, s4=4, s8=8")
    parser.add_argument("--background-workload-file", help="Optional workload manifest for background workers")
    parser.add_argument("--session-id", help="Explicit session id; defaults to output filename stem")
    parser.add_argument("--max-background-records-per-worker", type=int, default=0, help="Optional cap for drained-worker background audit records per worker; 0 means unbounded")
    parser.add_argument("--continue-on-measured-failure", action="store_true", help="Record measured query failures to skipped-output-path and continue instead of aborting the collection")
    parser.add_argument("--append-output", action="store_true", help="Append to existing output/background/skipped JSONL files instead of truncating them")
    return parser.parse_args()


def _load_entries(args: argparse.Namespace):
    if args.workload_dir:
        return load_job_directory_manifest(args.workload_dir, benchmark=args.benchmark, limit=args.limit_queries)
    if args.workload_file:
        return load_workload_manifest(args.workload_file, benchmark=args.benchmark, limit=args.limit_queries)
    raise ValueError("One of --workload-file or --workload-dir is required")


def _resolve_total_slots(args: argparse.Namespace) -> int:
    if getattr(args, "concurrency", None) is not None:
        if args.concurrency < 1:
            raise ValueError("--concurrency must be >= 1")
        return args.concurrency
    if args.measured_shards is not None:
        if args.measured_shards < 1:
            raise ValueError("--measured-shards must be >= 1")
        return args.measured_shards
    concurrency_level = args.concurrency_level or "s1"
    if concurrency_level == "s1" and args.background_streams is not None:
        return max(1, args.background_streams + 1)
    match = re.fullmatch(r"s(\d+)", concurrency_level)
    if not match:
        raise ValueError(
            "--concurrency-level must look like sN, for example s1, s4, s8, or s16"
        )
    total_slots = int(match.group(1))
    if total_slots < 1:
        raise ValueError("--concurrency-level must be >= s1")
    return total_slots


def _resolve_concurrency_label(args: argparse.Namespace, total_slots: int) -> str:
    if getattr(args, "concurrency", None) is not None:
        return f"s{total_slots}"
    return args.concurrency_level or f"s{total_slots}"


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    args = parse_args()
    config = LocalPostgresConfig(
        host=args.host,
        port=args.port,
        user=args.user,
        password=args.password,
        dbname=args.dbname,
        statement_timeout_ms=args.statement_timeout_ms,
    )
    entries = _load_entries(args)
    total_slots = _resolve_total_slots(args)
    concurrency_label = _resolve_concurrency_label(args, total_slots)
    session_id = args.session_id or Path(args.output_path).stem
    background_entries = None
    background_output_path = args.background_output_path
    skipped_output_path = args.skipped_output_path
    if args.continue_on_measured_failure and skipped_output_path is None:
        output_path = Path(args.output_path)
        skipped_output_path = str(output_path.with_name(f"{output_path.stem}_skipped{output_path.suffix}"))
    if total_slots > 1 and background_output_path is None:
        output_path = Path(args.output_path)
        background_output_path = str(output_path.with_name(f"{output_path.stem}_background{output_path.suffix}"))
    if total_slots > 1:
        if args.background_workload_file:
            background_entries = load_workload_manifest(
                args.background_workload_file,
                benchmark=args.benchmark,
                limit=0,
            )
        else:
            background_entries = entries

    logging.info(
        "status=runner_start workload=%s output_path=%s total_queries=%s concurrency_level=%s repeats=%s total_slots=%s restart_each_query=%s continue_on_measured_failure=%s",
        args.workload_dir or args.workload_file,
        args.output_path,
        len(entries),
        concurrency_label,
        args.repeats,
        total_slots,
        args.restart_each_query,
        args.continue_on_measured_failure,
    )

    if (
        concurrency_label == "s1"
        and args.repeats == 1
        and total_slots == max(1, args.num_threads)
        and args.benchmark == "unknown"
    ):
        queries = [entry.sql for entry in entries]
        run_concurrent_queries(
            db_config=config,
            queries=queries,
            save_file=args.output_path,
            num_threads=args.num_threads,
            vary_config=False,
            knobs=None,
            shuffle_queries=args.shuffle_queries,
            heat_interval_min=args.heat_interval_min,
            heat_time_window_min=args.heat_time_window_min,
            restart_each_query=args.restart_each_query,
            max_background_records_per_worker=args.max_background_records_per_worker,
            append_output=args.append_output,
            collect_table_heat=not args.disable_table_heat,
        )
    else:
        run_paired_concurrency(
            db_config=config,
            entries=entries,
            save_file=args.output_path,
            background_save_file=background_output_path,
            concurrency_level=concurrency_label,
            repeats=args.repeats,
            session_id=session_id,
            total_slots=total_slots,
            background_entries=background_entries,
            collection_mode=args.collection_mode,
            vary_config=False,
            knobs=None,
            shuffle_queries=args.shuffle_queries,
            heat_interval_min=args.heat_interval_min,
            heat_time_window_min=args.heat_time_window_min,
            restart_each_query=args.restart_each_query,
            max_background_records_per_worker=args.max_background_records_per_worker,
            continue_on_measured_failure=args.continue_on_measured_failure,
            skipped_save_file=skipped_output_path,
            append_output=args.append_output,
            collect_table_heat=not args.disable_table_heat,
        )
    print(f"collected_queries={len(entries) * args.repeats}")
    print(f"output_path={args.output_path}")
    if skipped_output_path:
        print(f"skipped_output_path={skipped_output_path}")
    print(f"session_id={session_id}")


if __name__ == "__main__":
    main()
