from __future__ import annotations

import argparse
from pathlib import Path

from lego.data.workload_manifest import dump_workload_manifest, load_job_directory_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a normalized JOB workload manifest from per-file SQL assets")
    parser.add_argument("--job-dir", required=True, help="Directory containing JOB *.sql files")
    parser.add_argument("--benchmark", default="job", help="Benchmark label written into the manifest")
    parser.add_argument("--output-path", required=True, help="Tab-separated workload manifest path")
    parser.add_argument("--limit", type=int, default=0, help="Maximum number of entries to emit; 0 means all")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    entries = load_job_directory_manifest(args.job_dir, benchmark=args.benchmark, limit=args.limit)
    dump_workload_manifest(entries, args.output_path)
    print(f"job_dir={Path(args.job_dir)}")
    print(f"entries={len(entries)}")
    print(f"output_path={args.output_path}")


if __name__ == "__main__":
    main()
