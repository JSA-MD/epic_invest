#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PID_FILE="${PAIRWISE_SHADOW_PID_FILE:-/tmp/epic_pairwise_shadow.pid}"
LOG_FILE="${PAIRWISE_SHADOW_LOG_FILE:-$ROOT_DIR/logs/pairwise_shadow_service.log}"
STATE_PATH="${PAIRWISE_SHADOW_STATE_PATH:-$ROOT_DIR/models/pairwise_regime_shadow_state.json}"
DECISION_LOG_PATH="${PAIRWISE_SHADOW_DECISION_LOG_PATH:-$ROOT_DIR/logs/pairwise_regime_shadow_decisions.jsonl}"
POLL_SECONDS="${PAIRWISE_SHADOW_POLL_SECONDS:-300}"

mkdir -p "$(dirname "$LOG_FILE")" "$(dirname "$STATE_PATH")" "$(dirname "$DECISION_LOG_PATH")"

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

case "${1:-}" in
  start)
    if is_running; then
      echo "pairwise shadow already running (pid $(cat "$PID_FILE"))"
      exit 0
    fi
    nohup "$ROOT_DIR/.venv/bin/python" -u "$ROOT_DIR/scripts/pairwise_regime_live.py" shadow-loop \
      --poll-seconds "$POLL_SECONDS" \
      --state-path "$STATE_PATH" \
      --decision-log-path "$DECISION_LOG_PATH" \
      >>"$LOG_FILE" 2>&1 &
    echo $! >"$PID_FILE"
    echo "pairwise shadow started (pid $!)"
    ;;
  stop)
    if ! is_running; then
      echo "pairwise shadow is not running"
      rm -f "$PID_FILE"
      exit 0
    fi
    kill "$(cat "$PID_FILE")"
    rm -f "$PID_FILE"
    echo "pairwise shadow stopped"
    ;;
  status)
    if is_running; then
      echo "pairwise shadow running (pid $(cat "$PID_FILE"))"
      exit 0
    fi
    echo "pairwise shadow stopped"
    exit 1
    ;;
  *)
    echo "usage: $0 {start|stop|status}"
    exit 2
    ;;
esac
