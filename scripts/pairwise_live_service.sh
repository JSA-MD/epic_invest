#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PID_FILE="${PAIRWISE_LIVE_PID_FILE:-/tmp/epic_pairwise_live.pid}"
LOG_FILE="${PAIRWISE_LIVE_LOG_FILE:-$ROOT_DIR/logs/pairwise_live_service.log}"
STATE_PATH="${PAIRWISE_LIVE_STATE_PATH:-$ROOT_DIR/models/pairwise_regime_live_state.json}"
DECISION_LOG_PATH="${PAIRWISE_LIVE_DECISION_LOG_PATH:-$ROOT_DIR/logs/pairwise_regime_decisions.jsonl}"
POLL_SECONDS="${PAIRWISE_LIVE_POLL_SECONDS:-300}"
MODE="${PAIRWISE_LIVE_MODE:-demo}"
FORCE_EXECUTE="${PAIRWISE_FORCE_EXECUTE:-0}"
FORCE_NOTE="${PAIRWISE_FORCE_NOTE:-manual_primary_switch}"
PROMOTION_REPORT_PATH="${PAIRWISE_LIVE_PROMOTION_REPORT_PATH:-$ROOT_DIR/models/gp_regime_mixture_btc_bnb_pairwise_market_os_pipeline_report.json}"

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
      echo "pairwise live already running (pid $(cat "$PID_FILE"))"
      exit 0
    fi
    extra_args=()
    case "$(printf '%s' "$FORCE_EXECUTE" | tr '[:upper:]' '[:lower:]')" in
      1|true|yes|on)
        extra_args+=(--force-execute --force-note "$FORCE_NOTE")
        ;;
    esac
    if (( ${#extra_args[@]} > 0 )); then
      nohup "$ROOT_DIR/.venv/bin/python" -u "$ROOT_DIR/scripts/pairwise_regime_live.py" loop \
        --execute \
        --mode "$MODE" \
        --poll-seconds "$POLL_SECONDS" \
        --promotion-report "$PROMOTION_REPORT_PATH" \
        --state-path "$STATE_PATH" \
        --decision-log-path "$DECISION_LOG_PATH" \
        "${extra_args[@]}" \
        >>"$LOG_FILE" 2>&1 &
    else
      nohup "$ROOT_DIR/.venv/bin/python" -u "$ROOT_DIR/scripts/pairwise_regime_live.py" loop \
        --execute \
        --mode "$MODE" \
        --poll-seconds "$POLL_SECONDS" \
        --promotion-report "$PROMOTION_REPORT_PATH" \
        --state-path "$STATE_PATH" \
        --decision-log-path "$DECISION_LOG_PATH" \
        >>"$LOG_FILE" 2>&1 &
    fi
    echo $! >"$PID_FILE"
    echo "pairwise live started (pid $!)"
    ;;
  stop)
    if ! is_running; then
      echo "pairwise live is not running"
      rm -f "$PID_FILE"
      exit 0
    fi
    kill "$(cat "$PID_FILE")"
    rm -f "$PID_FILE"
    echo "pairwise live stopped"
    ;;
  status)
    if is_running; then
      echo "pairwise live running (pid $(cat "$PID_FILE"))"
      exit 0
    fi
    echo "pairwise live stopped"
    exit 1
    ;;
  *)
    echo "usage: $0 {start|stop|status}"
    exit 2
    ;;
esac
