#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PID_FILE="${LOB_COLLECTION_PID_FILE:-/tmp/epic_lob_collection.pid}"
LOG_FILE="${LOB_COLLECTION_LOG_FILE:-$ROOT_DIR/logs/lob_collection_service.log}"
INTERVAL_SECONDS="${LOB_COLLECTION_INTERVAL_SECONDS:-60}"
DEPTH_LIMIT="${LOB_COLLECTION_DEPTH_LIMIT:-20}"
AGG_LIMIT="${LOB_COLLECTION_AGG_LIMIT:-200}"
SYMBOLS_RAW="${LOB_COLLECTION_SYMBOLS:-BTCUSDT,BNBUSDT}"
LOB_LAUNCHD_LABEL="${LOB_LAUNCHD_LABEL:-com.epicinvest.lob-collector}"
LOB_LAUNCHD_DOMAIN="${LOB_LAUNCHD_DOMAIN:-gui/$(id -u)}"
LOB_LAUNCHD_PLIST_PATH="${LOB_LAUNCHD_PLIST_PATH:-$ROOT_DIR/scripts/com.epicinvest.lob-collector.plist}"
LOB_LAUNCHD_ENTRY_PATH="${LOB_LAUNCHD_ENTRY_PATH:-$ROOT_DIR/scripts/lob_collection_launchd_entry.sh}"
LOB_LAUNCHD_ENV_PATH="${LOB_LAUNCHD_ENV_PATH:-$ROOT_DIR/models/lob_collection_launchd_env.sh}"

mkdir -p \
  "$(dirname "$LOG_FILE")" \
  "$(dirname "$LOB_LAUNCHD_ENV_PATH")"

lob_launchd_pid() {
  launchctl print "$LOB_LAUNCHD_DOMAIN/$LOB_LAUNCHD_LABEL" 2>/dev/null \
    | awk '/^[[:space:]]*pid = / {print $3; exit}'
}

bootout_lob_launchd() {
  launchctl bootout "$LOB_LAUNCHD_DOMAIN/$LOB_LAUNCHD_LABEL" >/dev/null 2>&1 || true
}

refresh_pid_file_from_launchd() {
  local pid
  pid="$(lob_launchd_pid || true)"
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
    echo "lob collector python not found: $ROOT_DIR/.venv/bin/python" >&2
    exit 1
  fi
  if [[ ! -f "$LOB_LAUNCHD_PLIST_PATH" ]]; then
    echo "lob collector launchd plist not found: $LOB_LAUNCHD_PLIST_PATH" >&2
    exit 1
  fi
  if [[ ! -x "$LOB_LAUNCHD_ENTRY_PATH" ]]; then
    echo "lob collector launchd entry not executable: $LOB_LAUNCHD_ENTRY_PATH" >&2
    exit 1
  fi
}

write_launchd_env_file() {
  {
    printf 'export LOB_COLLECTION_PID_FILE=%q\n' "$PID_FILE"
    printf 'export LOB_COLLECTION_LOG_FILE=%q\n' "$LOG_FILE"
    printf 'export LOB_COLLECTION_INTERVAL_SECONDS=%q\n' "$INTERVAL_SECONDS"
    printf 'export LOB_COLLECTION_DEPTH_LIMIT=%q\n' "$DEPTH_LIMIT"
    printf 'export LOB_COLLECTION_AGG_LIMIT=%q\n' "$AGG_LIMIT"
    printf 'export LOB_COLLECTION_SYMBOLS=%q\n' "$SYMBOLS_RAW"
  } >"$LOB_LAUNCHD_ENV_PATH"
}

case "${1:-}" in
  start)
    ensure_launchd_prerequisites
    if is_running; then
      echo "lob collector already running (pid $(cat "$PID_FILE"))"
      exit 0
    fi
    write_launchd_env_file
    : >"$LOG_FILE"
    bootout_lob_launchd
    launchctl bootstrap "$LOB_LAUNCHD_DOMAIN" "$LOB_LAUNCHD_PLIST_PATH"
    launchctl enable "$LOB_LAUNCHD_DOMAIN/$LOB_LAUNCHD_LABEL" >/dev/null 2>&1 || true
    launchctl kickstart -k "$LOB_LAUNCHD_DOMAIN/$LOB_LAUNCHD_LABEL"

    pid=""
    for _attempt in $(seq 1 10); do
      pid="$(lob_launchd_pid || true)"
      if [[ -n "${pid:-}" ]] && kill -0 "$pid" >/dev/null 2>&1; then
        printf '%s\n' "$pid" >"$PID_FILE"
        break
      fi
      pid=""
      sleep 1
    done
    if [[ -z "${pid:-}" ]]; then
      echo "lob collector failed to stay running; recent log:" >&2
      tail -n 40 "$LOG_FILE" >&2 || true
      rm -f "$PID_FILE"
      exit 1
    fi
    echo "lob collector started (pid $pid)"
    ;;
  stop)
    running_pid="$(cat "$PID_FILE" 2>/dev/null || true)"
    bootout_lob_launchd
    if [[ -n "${running_pid:-}" ]] && kill -0 "$running_pid" >/dev/null 2>&1; then
      kill "$running_pid" >/dev/null 2>&1 || true
    fi
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
