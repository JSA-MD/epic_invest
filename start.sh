#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$ROOT_DIR/scripts/trader_service.sh"

start_trader
"$ROOT_DIR/telegram_launchd_load.sh"
"$ROOT_DIR/watchdog_launchd_load.sh"
send_telegram_lifecycle_message $'트레이더 시작 완료\n- 상태: 실행 중\n- 텔레그램 봇: 활성\n- 운영 감시: 활성\n- 제어 명령: /status /plan /positions'
