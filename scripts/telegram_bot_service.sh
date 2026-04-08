#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
ENTRY_SCRIPT="$ROOT_DIR/scripts/telegram_bot.py"
LOG_FILE="${TELEGRAM_BOT_LOG_FILE:-/tmp/epic-invest-telegram-bot.log}"
PID_FILE="${TELEGRAM_BOT_PID_FILE:-/tmp/epic-invest-telegram-bot.pid}"
STARTUP_WAIT_SECONDS="${TELEGRAM_BOT_STARTUP_WAIT_SECONDS:-15}"

print_line() {
  printf '%s\n' "$1"
}

print_info() {
  printf '  %s\n' "$1"
}

collect_process_rows() {
  ps -ax -o pid=,ppid=,stat=,etime=,command= | awk -v entry="$ENTRY_SCRIPT" '
    index($0, entry) > 0 { print }
  '
}

collect_pids() {
  collect_process_rows | awk '{ print $1 }' | awk 'NF' | sort -u
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
  if [[ ! -f "$ROOT_DIR/.env" ]]; then
    print_line "❌ .env 파일이 없습니다: $ROOT_DIR/.env"
    exit 1
  fi
  if ! awk -F= '/^TELEGRAM_BOT_TOKEN=/{found=1} END{exit(found?0:1)}' "$ROOT_DIR/.env"; then
    print_line "❌ TELEGRAM_BOT_TOKEN 이 .env에 없습니다."
    exit 1
  fi
  if ! awk -F= '/^TELEGRAM_ALLOWED_CHAT_IDS=/{found=1} END{exit(found?0:1)}' "$ROOT_DIR/.env"; then
    print_line "❌ TELEGRAM_ALLOWED_CHAT_IDS 가 .env에 없습니다."
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

stop_bot() {
  print_line "🛑 Epic Invest 텔레그램 봇 종료"
  cleanup_stale_pid_file

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
    print_info "실행 중인 텔레그램 봇 없음"
  else
    local pid
    for pid in "${pids[@]}"; do
      kill -TERM "$pid" 2>/dev/null || true
      if ! wait_for_exit "$pid" 20 0.5; then
        kill -KILL "$pid" 2>/dev/null || true
        wait_for_exit "$pid" 10 0.5 || true
      fi
      print_info "프로세스 종료 완료 (PID: $pid)"
    done
  fi

  rm -f "$PID_FILE"
  print_line "🏁 텔레그램 봇 종료 완료"
}

start_bot() {
  print_line "🚀 Epic Invest 텔레그램 봇 시작"
  ensure_prerequisites
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
      kill -TERM "$old_pid" 2>/dev/null || true
      wait_for_exit "$old_pid" 20 0.5 || true
      print_info "기존 프로세스 정리 (PID: $old_pid)"
    done
  fi
  rm -f "$PID_FILE"

  : > "$LOG_FILE"
  nohup "$PYTHON_BIN" -u "$ENTRY_SCRIPT" loop > "$LOG_FILE" 2>&1 < /dev/null &
  local pid="$!"
  echo "$pid" > "$PID_FILE"
  print_info "텔레그램 봇 시작 (PID: $pid, 로그: $LOG_FILE)"
  print_info "준비 대기 중..."

  local ready=0
  local i
  for ((i = 0; i < STARTUP_WAIT_SECONDS; i++)); do
    if ! is_pid_running "$pid"; then
      print_line "❌ 텔레그램 봇이 시작 직후 종료되었습니다."
      tail -n 40 "$LOG_FILE" || true
      rm -f "$PID_FILE"
      exit 1
    fi
    if rg -q 'Telegram bot ready' "$LOG_FILE" 2>/dev/null; then
      ready=1
      break
    fi
    sleep 1
  done

  if (( ready == 1 )); then
    print_info "텔레그램 봇 준비 완료"
  else
    print_info "텔레그램 봇은 실행 중이지만 준비 로그는 아직 확인되지 않았습니다"
  fi

  print_line ""
  print_line "════════════════════════════════════════"
  print_line "  Epic Invest 텔레그램 봇 시작 완료"
  print_line "  실행 스크립트: $ENTRY_SCRIPT"
  print_line "  로그:          $LOG_FILE"
  print_line "  PID 파일:      $PID_FILE"
  print_line "  종료:          ./telegram_stop.sh"
  print_line "════════════════════════════════════════"
}

restart_bot() {
  stop_bot
  start_bot
}
