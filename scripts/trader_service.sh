#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
ENTRY_SCRIPT="$ROOT_DIR/scripts/rotation_target_050_live.py"
LEGACY_SCRIPT="$ROOT_DIR/scripts/live_trader.py"
TRADER_PLIST_PATH="$ROOT_DIR/scripts/com.epicinvest.trader.plist"
TRADER_LAUNCHD_ENTRY="$ROOT_DIR/scripts/trader_launchd_entry.sh"
TRADER_DOMAIN="gui/$(id -u)"
TRADER_LABEL="com.epicinvest.trader"
LOG_FILE="${TRADER_LOG_FILE:-/tmp/epic-invest-trader.log}"
PID_FILE="${TRADER_PID_FILE:-/tmp/epic-invest-trader.pid}"
POLL_SECONDS="${TRADER_POLL_SECONDS:-60}"
STARTUP_WAIT_SECONDS="${TRADER_STARTUP_WAIT_SECONDS:-20}"
REBALANCE_NOTIONAL_BAND_USD="${REBALANCE_NOTIONAL_BAND_USD:-25}"

print_line() {
  printf '%s\n' "$1"
}

print_info() {
  printf '  %s\n' "$1"
}

read_env_value() {
  local key="$1"
  if [[ ! -f "$ROOT_DIR/.env" ]]; then
    return 0
  fi
  awk -F= -v k="$key" '$1 == k {print substr($0, index($0, "=") + 1)}' "$ROOT_DIR/.env" \
    | tail -n 1 \
    | tr -d '"' \
    | tr -d "'"
}

telegram_enabled() {
  local flag
  flag="$(read_env_value "TELEGRAM_NOTIFICATIONS_ENABLED")"
  flag="$(printf '%s' "$flag" | tr '[:upper:]' '[:lower:]')"
  case "$flag" in
    0|false|no|off) return 1 ;;
  esac
  [[ -n "$(read_env_value "TELEGRAM_BOT_TOKEN")" ]]
}

send_telegram_lifecycle_message() {
  local text="$1"
  if ! telegram_enabled; then
    return 0
  fi

  local token chat_raw
  token="$(read_env_value "TELEGRAM_BOT_TOKEN")"
  chat_raw="$(read_env_value "TELEGRAM_ALLOWED_CHAT_IDS")"
  if [[ -z "${chat_raw:-}" ]]; then
    chat_raw="$(read_env_value "TELEGRAM_CHAT_ID")"
  fi
  if [[ -z "${token:-}" || -z "${chat_raw:-}" ]]; then
    return 0
  fi

  local IFS=','
  local chat_id
  for chat_id in $chat_raw; do
    chat_id="${chat_id//[[:space:]]/}"
    [[ -n "${chat_id:-}" ]] || continue
    curl -sS -X POST "https://api.telegram.org/bot${token}/sendMessage" \
      -d "chat_id=${chat_id}" \
      --data-urlencode "text=${text}" >/dev/null 2>&1 || true
  done
}

collect_process_rows() {
  ps -ax -o pid=,ppid=,stat=,etime=,command= | awk \
    -v entry="$ENTRY_SCRIPT" \
    -v entry_rel="scripts/rotation_target_050_live.py" \
    -v legacy="$LEGACY_SCRIPT" \
    -v legacy_rel="scripts/live_trader.py" '
    index($0, entry) > 0 || index($0, entry_rel) > 0 || index($0, legacy) > 0 || index($0, legacy_rel) > 0 { print }
  '
}

collect_pids() {
  collect_process_rows | awk '{ print $1 }' | awk 'NF' | sort -u
}

collect_zombie_rows() {
  collect_process_rows | awk '$3 ~ /^Z/ { print }'
}

launchd_trader_pid() {
  launchctl print "$TRADER_DOMAIN/$TRADER_LABEL" 2>/dev/null | awk '/^[[:space:]]*pid = / {print $3; exit}'
}

bootout_trader_launchd() {
  launchctl bootout "$TRADER_DOMAIN/$TRADER_LABEL" >/dev/null 2>&1 || true
}

read_pid_file() {
  if [[ -f "$PID_FILE" ]]; then
    cat "$PID_FILE"
  fi
}

is_pid_running() {
  local pid="$1"
  kill -0 "$pid" 2>/dev/null
}

cleanup_stale_pid_file() {
  if [[ ! -f "$PID_FILE" ]]; then
    return
  fi
  local pid
  pid="$(read_pid_file || true)"
  if [[ -z "${pid:-}" ]]; then
    rm -f "$PID_FILE"
    return
  fi
  if ! is_pid_running "$pid"; then
    rm -f "$PID_FILE"
  fi
}

ensure_prerequisites() {
  if [[ ! -x "$PYTHON_BIN" ]]; then
    print_line "❌ Python 가상환경을 찾을 수 없습니다: $PYTHON_BIN"
    exit 1
  fi
  if [[ ! -f "$ENTRY_SCRIPT" ]]; then
    print_line "❌ 실행 스크립트를 찾을 수 없습니다: $ENTRY_SCRIPT"
    exit 1
  fi
  if [[ ! -f "$TRADER_PLIST_PATH" ]]; then
    print_line "❌ launchd plist를 찾을 수 없습니다: $TRADER_PLIST_PATH"
    exit 1
  fi
  if [[ ! -f "$TRADER_LAUNCHD_ENTRY" ]]; then
    print_line "❌ launchd 실행 스크립트를 찾을 수 없습니다: $TRADER_LAUNCHD_ENTRY"
    exit 1
  fi
  if [[ ! -f "$ROOT_DIR/.env" ]]; then
    print_line "❌ .env 파일이 없습니다: $ROOT_DIR/.env"
    exit 1
  fi

  local env_keys
  env_keys="$(awk -F= '/^[A-Za-z_][A-Za-z0-9_]*=/{print $1}' "$ROOT_DIR/.env" | tr '\n' ' ')"
  if [[ "$env_keys" != *"BINANCE_API_KEY"* && "$env_keys" != *"BINANCE_DEMO_API_KEY"* ]]; then
    print_line "❌ Binance API 키가 .env에 없습니다."
    exit 1
  fi
  if [[ "$env_keys" != *"BINANCE_SECRET"* && "$env_keys" != *"BINANCE_SECRET_KEY"* && "$env_keys" != *"BINANCE_DEMO_API_SECRET"* ]]; then
    print_line "❌ Binance API 시크릿이 .env에 없습니다."
    exit 1
  fi
}

wait_for_exit() {
  local pid="$1"
  local attempts="${2:-20}"
  local sleep_s="${3:-0.5}"
  local i
  for ((i = 0; i < attempts; i++)); do
    if ! is_pid_running "$pid"; then
      return 0
    fi
    sleep "$sleep_s"
  done
  return 1
}

install_shutdown_protection() {
  if [[ ! -x "$PYTHON_BIN" || ! -f "$ENTRY_SCRIPT" ]]; then
    return 0
  fi
  print_line "🛡️ 종료 보호주문 설치 중..."
  if "$PYTHON_BIN" -u "$ENTRY_SCRIPT" shutdown-protect --execute >> "$LOG_FILE" 2>&1; then
    print_info "보호주문 설치 완료"
  else
    print_info "보호주문 설치 실패 - 로그 확인: $LOG_FILE"
  fi
}

stop_pid() {
  local pid="$1"
  if ! is_pid_running "$pid"; then
    return 0
  fi

  kill -TERM "$pid" 2>/dev/null || true
  if wait_for_exit "$pid" 20 0.5; then
    return 0
  fi

  kill -KILL "$pid" 2>/dev/null || true
  wait_for_exit "$pid" 10 0.5 || true
}

stop_trader() {
  print_line "🛑 Epic Invest 트레이더 종료"
  cleanup_stale_pid_file
  bootout_trader_launchd

  local pids=()
  local pid_line
  while IFS= read -r pid_line; do
    [[ -n "${pid_line:-}" ]] || continue
    pids+=("$pid_line")
  done < <(
    {
      read_pid_file || true
      collect_pids || true
    } | awk 'NF' | sort -u
  )

  if (( ${#pids[@]} == 0 )); then
    print_info "실행 중인 트레이더 프로세스 없음"
  else
    if (( ${#pids[@]} > 1 )); then
      print_info "다중 프로세스 감지: ${#pids[@]}개"
    fi
    local pid
    for pid in "${pids[@]}"; do
      stop_pid "$pid"
      print_info "프로세스 종료 완료 (PID: $pid)"
    done
  fi

  local zombie_rows
  zombie_rows="$(collect_zombie_rows || true)"
  if [[ -n "${zombie_rows:-}" ]]; then
    print_info "좀비 프로세스 흔적 감지"
    printf '%s\n' "$zombie_rows"
  fi

  install_shutdown_protection
  rm -f "$PID_FILE"
  print_line "🏁 트레이더 종료 완료"
}

start_trader() {
  print_line "🚀 Epic Invest 트레이더 시작"
  ensure_prerequisites

  print_line "🧹 기존 프로세스 정리 중..."
  cleanup_stale_pid_file
  local existing_pids=()
  local pid_line
  while IFS= read -r pid_line; do
    [[ -n "${pid_line:-}" ]] || continue
    existing_pids+=("$pid_line")
  done < <(
    {
      read_pid_file || true
      collect_pids || true
    } | awk 'NF' | sort -u
  )
  if (( ${#existing_pids[@]} > 0 )); then
    local old_pid
    for old_pid in "${existing_pids[@]}"; do
      stop_pid "$old_pid"
      print_info "기존 프로세스 정리 (PID: $old_pid)"
    done
  fi
  rm -f "$PID_FILE"
  print_info "프로세스 정리 완료"

  print_line "🔐 환경 확인 중..."
  print_info "실행기: $ENTRY_SCRIPT"
  print_info "로그: $LOG_FILE"
  print_info "PID 파일: $PID_FILE"
  print_info "폴링 주기: ${POLL_SECONDS}s"
  print_info "미세 리밸런싱 band: \$${REBALANCE_NOTIONAL_BAND_USD}"

  : > "$LOG_FILE"
  bootout_trader_launchd
  launchctl bootstrap "$TRADER_DOMAIN" "$TRADER_PLIST_PATH"
  launchctl enable "$TRADER_DOMAIN/$TRADER_LABEL" >/dev/null 2>&1 || true
  launchctl kickstart -k "$TRADER_DOMAIN/$TRADER_LABEL"

  local pid=""
  local launch_attempt
  for ((launch_attempt = 0; launch_attempt < 10; launch_attempt++)); do
    pid="$(launchd_trader_pid || true)"
    if [[ -n "${pid:-}" ]] && is_pid_running "$pid"; then
      break
    fi
    pid=""
    sleep 1
  done
  if [[ -z "${pid:-}" ]]; then
    print_line "❌ launchd 트레이더 PID를 확인하지 못했습니다."
    exit 1
  fi

  echo "$pid" > "$PID_FILE"
  print_info "트레이더 시작 (PID: $pid, 로그: $LOG_FILE)"
  print_info "준비 대기 중..."

  local ready=0
  local i
  for ((i = 0; i < STARTUP_WAIT_SECONDS; i++)); do
    local current_pid
    current_pid="$(launchd_trader_pid || true)"
    if [[ -n "${current_pid:-}" ]] && is_pid_running "$current_pid"; then
      pid="$current_pid"
      echo "$pid" > "$PID_FILE"
    fi
    if rg -q 'Rotation Target 0.5% Live Plan|Action:' "$LOG_FILE" 2>/dev/null; then
      ready=1
      break
    fi
    sleep 1
  done

  if ! is_pid_running "$pid"; then
    print_line "❌ 트레이더가 시작 직후 종료되었습니다."
    print_info "최근 로그:"
    tail -n 40 "$LOG_FILE" || true
    rm -f "$PID_FILE"
    exit 1
  fi

  if (( ready == 1 )); then
    print_info "트레이더 준비 완료"
  else
    print_info "트레이더는 실행 중이지만 준비 로그는 아직 확인되지 않았습니다"
  fi

  print_line ""
  print_line "════════════════════════════════════════"
  print_line "  Epic Invest 트레이더 시작 완료"
  print_line "  실행 스크립트: $ENTRY_SCRIPT"
  print_line "  로그:          $LOG_FILE"
  print_line "  PID 파일:      $PID_FILE"
  print_line "  종료:          ./stop.sh"
  print_line "════════════════════════════════════════"
}

restart_trader() {
  stop_trader
  start_trader
}
