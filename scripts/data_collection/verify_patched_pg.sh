#!/usr/bin/env bash
set -euo pipefail

HOST="127.0.0.1"
PORT="5188"
USER_NAME="${USER:-postgres}"
DB_NAME="imdb"
PSQL_BIN="${PSQL_BIN:-psql}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --host) HOST="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --user) USER_NAME="$2"; shift 2 ;;
    --dbname) DB_NAME="$2"; shift 2 ;;
    *) echo "Unknown arg: $1" >&2; exit 2 ;;
  esac
done

PSQL=("$PSQL_BIN" -h "$HOST" -p "$PORT" -U "$USER_NAME" -d "$DB_NAME" -At)
JSON_OUTPUT="$(${PSQL[@]} -c "EXPLAIN (ANALYZE, FORMAT JSON) SELECT 1;")"

python3 - <<'PY' "$JSON_OUTPUT"
import json
import sys
payload = json.loads(sys.argv[1])
if not isinstance(payload, list) or not payload:
    raise SystemExit("Patched PG verification failed: EXPLAIN JSON root is not a non-empty list")
root = payload[0]
plan = root.get('Plan', {})
plan_metrics = root.get('System Metrics at Plan Time')
node_metrics = plan.get('Pre-execution Metrics')
if not isinstance(plan_metrics, dict):
    raise SystemExit('Patched PG verification failed: System Metrics at Plan Time missing or not a JSON object')
if not isinstance(node_metrics, dict):
    raise SystemExit('Patched PG verification failed: Pre-execution Metrics missing or not a JSON object')
if not isinstance(plan_metrics.get('System Metrics'), dict):
    raise SystemExit('Patched PG verification failed: plan-level System Metrics wrapper missing')
if not isinstance(node_metrics.get('System Metrics'), dict):
    raise SystemExit('Patched PG verification failed: node-level System Metrics wrapper missing')
print('patched_pg_ok=1')
print('plan_level_wrapper=dict')
print('node_level_wrapper=dict')
PY
