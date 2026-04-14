#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PID_FILE="${LOB_COLLECTION_PID_FILE:-/tmp/epic_lob_collection.pid}"
LOG_FILE="${LOB_COLLECTION_LOG_FILE:-$ROOT_DIR/logs/lob_collection_service.log}"
INTERVAL_SECONDS="${LOB_COLLECTION_INTERVAL_SECONDS:-60}"
DEPTH_LIMIT="${LOB_COLLECTION_DEPTH_LIMIT:-20}"
AGG_LIMIT="${LOB_COLLECTION_AGG_LIMIT:-200}"
SYMBOLS_RAW="${LOB_COLLECTION_SYMBOLS:-BTCUSDT,BNBUSDT}"
STARTUP_WAIT_SECONDS="${LOB_COLLECTION_STARTUP_WAIT_SECONDS:-5}"

mkdir -p "$(dirname "$LOG_FILE")"

is_running() {
  if [[ -f "$PID_FILE" ]]; then
    local pid
    pid="$(cat "$PID_FILE")"
    if kill -0 "$pid" >/dev/null 2>&1; then
      return 0
    fi
  fi
  return 1
}

wait_for_start() {
  local pid="$1"
  local attempts="${2:-5}"
  local i
  for ((i = 0; i < attempts; i++)); do
    if kill -0 "$pid" >/dev/null 2>&1; then
      sleep 1
      continue
    fi
    return 1
  done
  return 0
}

case "${1:-}" in
  start)
    if is_running; then
      echo "lob collector already running (pid $(cat "$PID_FILE"))"
      exit 0
    fi
    IFS=',' read -r -a symbols <<< "$SYMBOLS_RAW"
    : >"$LOG_FILE"
    nohup "$ROOT_DIR/.venv/bin/python" -u "$ROOT_DIR/scripts/lob_market_data.py" loop \
      --symbols "${symbols[@]}" \
      --depth-limit "$DEPTH_LIMIT" \
      --agg-limit "$AGG_LIMIT" \
      --interval-seconds "$INTERVAL_SECONDS" \
      >"$LOG_FILE" 2>&1 < /dev/null &
    pid="$!"
    echo "$pid" >"$PID_FILE"
    if ! wait_for_start "$pid" "$STARTUP_WAIT_SECONDS"; then
      echo "lob collector failed to stay running"
      tail -n 40 "$LOG_FILE" || true
      rm -f "$PID_FILE"
      exit 1
    fi
    echo "lob collector started (pid $pid)"
    ;;
  stop)
    if ! is_running; then
      echo "lob collector is not running"
      rm -f "$PID_FILE"
      exit 0
    fi
    kill "$(cat "$PID_FILE")"
    rm -f "$PID_FILE"
    echo "lob collector stopped"
    ;;
  status)
    if is_running; then
      echo "lob collector running (pid $(cat "$PID_FILE"))"
      exit 0
    fi
    echo "lob collector stopped"
    exit 1
    ;;
  *)
    echo "usage: $0 {start|stop|status}"
    exit 2
    ;;
esac
