#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$ROOT_DIR/scripts/trader_service.sh"

stop_pairwise_processes
start_trader
write_active_runtime_profile "core" "${BINANCE_MODE:-demo}" "0"
"$ROOT_DIR/telegram_launchd_load.sh" || true
"$ROOT_DIR/watchdog_launchd_load.sh" || true
send_telegram_lifecycle_message $'코어 트레이더 시작 완료\n- 상태: 실행 중\n- 텔레그램 봇: 활성\n- 운영 감시: 활성\n- core 주문: 활성\n- pairwise 주문: 비활성\n- pairwise shadow: 비활성\n- 제어 명령: /status /plan /positions'
