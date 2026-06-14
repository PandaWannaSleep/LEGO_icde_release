#!/usr/bin/env bash
set -euo pipefail

HOST="127.0.0.1"
PORT="5188"
USER_NAME="${USER:-postgres}"
DB_NAME="imdb"
SCHEMA_NAME="public"
PSQL_BIN="${PSQL_BIN:-psql}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --host) HOST="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --user) USER_NAME="$2"; shift 2 ;;
    --dbname) DB_NAME="$2"; shift 2 ;;
    --schema) SCHEMA_NAME="$2"; shift 2 ;;
    *) echo "Unknown arg: $1" >&2; exit 2 ;;
  esac
done

PSQL=("$PSQL_BIN" -h "$HOST" -p "$PORT" -U "$USER_NAME" -d "$DB_NAME" -v ON_ERROR_STOP=1 -At)

"${PSQL[@]}" -c "CREATE EXTENSION IF NOT EXISTS pageinspect;"
"${PSQL[@]}" -c "CREATE EXTENSION IF NOT EXISTS pg_prewarm;"
TABLES="$(${PSQL[@]} -c "SELECT quote_ident(schemaname) || '.' || quote_ident(relname) FROM pg_stat_user_tables WHERE schemaname = '$SCHEMA_NAME' ORDER BY relname;")"
while IFS= read -r table_name; do
  [[ -z "$table_name" ]] && continue
  "${PSQL[@]}" -c "SELECT pg_prewarm('$table_name', 'buffer');" >/dev/null
  echo "prewarmed=$table_name"
done <<< "$TABLES"
"${PSQL[@]}" -c "CHECKPOINT;" >/dev/null
"${PSQL[@]}" -c "SELECT pg_stat_reset();" >/dev/null
"${PSQL[@]}" -c "DROP TABLE IF EXISTS table_heat_history;" >/dev/null

echo "prewarm_complete=1"
