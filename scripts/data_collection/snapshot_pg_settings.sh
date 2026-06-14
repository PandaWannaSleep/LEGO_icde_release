#!/usr/bin/env bash
set -euo pipefail

HOST="127.0.0.1"
PORT="5188"
USER_NAME="${USER:-postgres}"
DB_NAME="imdb"
OUTPUT_PATH=""
PSQL_BIN="${PSQL_BIN:-psql}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --host) HOST="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --user) USER_NAME="$2"; shift 2 ;;
    --dbname) DB_NAME="$2"; shift 2 ;;
    --output-path) OUTPUT_PATH="$2"; shift 2 ;;
    *) echo "Unknown arg: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "$OUTPUT_PATH" ]]; then
  echo "--output-path is required" >&2
  exit 2
fi

mkdir -p "$(dirname "$OUTPUT_PATH")"

PSQL=("$PSQL_BIN" -h "$HOST" -p "$PORT" -U "$USER_NAME" -d "$DB_NAME" -At)
SETTINGS_JSON="$(${PSQL[@]} -c "SELECT json_object_agg(name, setting) FROM pg_settings WHERE name IN ('track_activities','track_counts','track_io_timing','max_connections','shared_buffers','effective_cache_size','work_mem','maintenance_work_mem','max_wal_size','min_wal_size','checkpoint_timeout','checkpoint_completion_target','wal_compression','max_worker_processes','max_parallel_workers','max_parallel_workers_per_gather','autovacuum_max_workers','autovacuum_vacuum_scale_factor');")"
VERSION_STR="$(${PSQL[@]} -c "SELECT version();")"
UNAME_STR="$(uname -a)"

python3 - <<'PY' "$SETTINGS_JSON" "$VERSION_STR" "$UNAME_STR" "$DB_NAME" "$OUTPUT_PATH"
import json
import sys
settings = json.loads(sys.argv[1]) if sys.argv[1] else {}
version = sys.argv[2]
uname = sys.argv[3]
db_name = sys.argv[4]
out = sys.argv[5]
payload = {
    'db_name': db_name,
    'postgres_version': version,
    'uname': uname,
    'settings': settings,
}
with open(out, 'w', encoding='utf-8') as handle:
    json.dump(payload, handle, indent=2)
print(out)
PY
