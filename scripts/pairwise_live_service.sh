#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT_DIR/scripts/trader_service.sh"
PID_FILE="${PAIRWISE_LIVE_PID_FILE:-/tmp/epic_pairwise_live.pid}"
LOG_FILE="${PAIRWISE_LIVE_LOG_FILE:-$ROOT_DIR/logs/pairwise_live_service.log}"
STATE_PATH="${PAIRWISE_LIVE_STATE_PATH:-$ROOT_DIR/models/pairwise_regime_live_state.json}"
DECISION_LOG_PATH="${PAIRWISE_LIVE_DECISION_LOG_PATH:-$ROOT_DIR/logs/pairwise_regime_decisions.jsonl}"
POLL_SECONDS="${PAIRWISE_LIVE_POLL_SECONDS:-300}"
MODE="${PAIRWISE_LIVE_MODE:-demo}"
FORCE_EXECUTE="${PAIRWISE_FORCE_EXECUTE:-0}"
FORCE_NOTE="${PAIRWISE_FORCE_NOTE:-manual_primary_switch}"
PROMOTION_REPORT_PATH="${PAIRWISE_LIVE_PROMOTION_REPORT_PATH:-$ROOT_DIR/models/gp_regime_mixture_btc_bnb_pairwise_market_os_pipeline_report.json}"
PAIRWISE_LAUNCHD_LABEL="${PAIRWISE_LAUNCHD_LABEL:-com.epicinvest.pairwise-trader}"
PAIRWISE_LAUNCHD_DOMAIN="${PAIRWISE_LAUNCHD_DOMAIN:-gui/$(id -u)}"
PAIRWISE_LAUNCHD_PLIST_PATH="${PAIRWISE_LAUNCHD_PLIST_PATH:-$ROOT_DIR/scripts/com.epicinvest.pairwise-trader.plist}"
PAIRWISE_LAUNCHD_ENTRY_PATH="${PAIRWISE_LAUNCHD_ENTRY_PATH:-$ROOT_DIR/scripts/pairwise_live_launchd_entry.sh}"
PAIRWISE_LAUNCHD_ENV_PATH="${PAIRWISE_LAUNCHD_ENV_PATH:-$ROOT_DIR/models/pairwise_live_launchd_env.sh}"

mkdir -p \
  "$(dirname "$LOG_FILE")" \
  "$(dirname "$STATE_PATH")" \
  "$(dirname "$DECISION_LOG_PATH")" \
  "$(dirname "$PAIRWISE_LAUNCHD_ENV_PATH")"

maybe_send_lifecycle_message() {
  local text="$1"
  case "$(printf '%s' "${PAIRWISE_SUPPRESS_LIFECYCLE_MESSAGE:-0}" | tr '[:upper:]' '[:lower:]')" in
    1|true|yes|on) return 0 ;;
  esac
  send_telegram_lifecycle_message "$text"
}

pairwise_launchd_pid() {
  launchctl print "$PAIRWISE_LAUNCHD_DOMAIN/$PAIRWISE_LAUNCHD_LABEL" 2>/dev/null \
    | awk '/^[[:space:]]*pid = / {print $3; exit}'
}

bootout_pairwise_launchd() {
  launchctl bootout "$PAIRWISE_LAUNCHD_DOMAIN/$PAIRWISE_LAUNCHD_LABEL" >/dev/null 2>&1 || true
}

refresh_pid_file_from_launchd() {
  local pid
  pid="$(pairwise_launchd_pid || true)"
  if [[ -n "${pid:-}" ]] && kill -0 "$pid" >/dev/null 2>&1; then
    printf '%s\n' "$pid" >"$PID_FILE"
    return 0
  fi
  if [[ -f "$PID_FILE" ]]; then
    local file_pid
    file_pid="$(cat "$PID_FILE" 2>/dev/null || true)"
    if [[ -n "${file_pid:-}" ]] && kill -0 "$file_pid" >/dev/null 2>&1; then
      return 0
    fi
  fi
  rm -f "$PID_FILE"
  return 1
}

is_running() {
  refresh_pid_file_from_launchd
}

ensure_launchd_prerequisites() {
  if [[ ! -x "$ROOT_DIR/.venv/bin/python" ]]; then
    echo "pairwise live python not found: $ROOT_DIR/.venv/bin/python" >&2
    exit 1
  fi
  if [[ ! -f "$PAIRWISE_LAUNCHD_PLIST_PATH" ]]; then
    echo "pairwise launchd plist not found: $PAIRWISE_LAUNCHD_PLIST_PATH" >&2
    exit 1
  fi
  if [[ ! -x "$PAIRWISE_LAUNCHD_ENTRY_PATH" ]]; then
    echo "pairwise launchd entry not executable: $PAIRWISE_LAUNCHD_ENTRY_PATH" >&2
    exit 1
  fi
}

write_launchd_env_file() {
  {
    printf 'export PAIRWISE_LIVE_PID_FILE=%q\n' "$PID_FILE"
    printf 'export PAIRWISE_LIVE_LOG_FILE=%q\n' "$LOG_FILE"
    printf 'export PAIRWISE_LIVE_STATE_PATH=%q\n' "$STATE_PATH"
    printf 'export PAIRWISE_LIVE_DECISION_LOG_PATH=%q\n' "$DECISION_LOG_PATH"
    printf 'export PAIRWISE_LIVE_POLL_SECONDS=%q\n' "$POLL_SECONDS"
    printf 'export PAIRWISE_LIVE_MODE=%q\n' "$MODE"
    printf 'export PAIRWISE_FORCE_EXECUTE=%q\n' "$FORCE_EXECUTE"
    printf 'export PAIRWISE_FORCE_NOTE=%q\n' "$FORCE_NOTE"
    printf 'export PAIRWISE_LIVE_PROMOTION_REPORT_PATH=%q\n' "$PROMOTION_REPORT_PATH"
    # Drift-fix overlays — pinned safe defaults so watchdog-triggered restarts
    # never silently revert to unbounded sizing or stale-price exposure.
    printf 'export PAIRWISE_GROSS_CAP=%q\n' "${PAIRWISE_GROSS_CAP:-0.01}"
    printf 'export PAIRWISE_NO_TRADE_BAND_PCT=%q\n' "${PAIRWISE_NO_TRADE_BAND_PCT:-40}"
    printf 'export PAIRWISE_MAX_HOLD_BARS=%q\n' "${PAIRWISE_MAX_HOLD_BARS:-288}"
    printf 'export PAIRWISE_CVAR_CUT=%q\n' "${PAIRWISE_CVAR_CUT:-1}"
    printf 'export PAIRWISE_CVAR_CUT_HOLD_HOURS=%q\n' "${PAIRWISE_CVAR_CUT_HOLD_HOURS:-24}"
    printf 'export EPIC_MARKET_DATA_SOURCE=%q\n' "${EPIC_MARKET_DATA_SOURCE:-postgres}"
    for key in \
      MARKET_DATA_SOURCE \
      EPIC_POSTGRES_CONTAINER \
      EPIC_POSTGRES_USER \
      EPIC_POSTGRES_DB \
      EPIC_POSTGRES_SCHEMA \
      EPIC_POSTGRES_CANDLES_TABLE \
      EPIC_POSTGRES_FUNDING_TABLE \
      EPIC_DOCKER_BIN \
      PAIRWISE_DIRECTIONAL_GA_OVERLAY \
      PAIRWISE_DIRECTIONAL_GA_MODE \
      PAIRWISE_DIRECTIONAL_GA_PATH
    do
      if [[ -n "${!key:-}" ]]; then
        printf 'export %s=%q\n' "$key" "${!key}"
      fi
    done
  } >"$PAIRWISE_LAUNCHD_ENV_PATH"
}

case "${1:-}" in
  start)
    ensure_launchd_prerequisites
    if is_running; then
      echo "pairwise live already running (pid $(cat "$PID_FILE"))"
      exit 0
    fi
    write_launchd_env_file
    : >"$LOG_FILE"
    bootout_pairwise_launchd
    launchctl bootstrap "$PAIRWISE_LAUNCHD_DOMAIN" "$PAIRWISE_LAUNCHD_PLIST_PATH"
    launchctl enable "$PAIRWISE_LAUNCHD_DOMAIN/$PAIRWISE_LAUNCHD_LABEL" >/dev/null 2>&1 || true
    launchctl kickstart -k "$PAIRWISE_LAUNCHD_DOMAIN/$PAIRWISE_LAUNCHD_LABEL"

    pid=""
    for _attempt in $(seq 1 10); do
      pid="$(pairwise_launchd_pid || true)"
      if [[ -n "${pid:-}" ]] && kill -0 "$pid" >/dev/null 2>&1; then
        printf '%s\n' "$pid" >"$PID_FILE"
        break
      fi
      pid=""
      sleep 1
    done
    if [[ -z "${pid:-}" ]]; then
      echo "pairwise live failed to stay running; recent log:" >&2
      tail -n 40 "$LOG_FILE" >&2 || true
      rm -f "$PID_FILE"
      exit 1
    fi
    echo "pairwise live started (pid $pid)"
    write_active_runtime_profile "pairwise" "$MODE" "$FORCE_EXECUTE"
    maybe_send_lifecycle_message $'pairwise live service 시작\n- 트레이더: 실행 중\n- 시작 경로: scripts/pairwise_live_service.sh\n- 제어 명령: /status /plan /positions'
    ;;
  stop)
    running_pid="$(cat "$PID_FILE" 2>/dev/null || true)"
    bootout_pairwise_launchd
    if [[ -n "${running_pid:-}" ]] && kill -0 "$running_pid" >/dev/null 2>&1; then
      kill "$running_pid" >/dev/null 2>&1 || true
    fi
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
