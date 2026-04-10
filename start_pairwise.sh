#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$ROOT_DIR/scripts/trader_service.sh"

stop_core_processes
stop_pairwise_processes
PAIRWISE_FORCE_EXECUTE="${PAIRWISE_FORCE_EXECUTE:-1}" \
PAIRWISE_FORCE_NOTE="${PAIRWISE_FORCE_NOTE:-manual_primary_switch}" \
"$ROOT_DIR/scripts/pairwise_live_service.sh" start
write_active_runtime_profile "pairwise" "${PAIRWISE_LIVE_MODE:-${BINANCE_MODE:-demo}}" "${PAIRWISE_FORCE_EXECUTE:-1}"
"$ROOT_DIR/telegram_launchd_load.sh" || true
"$ROOT_DIR/watchdog_launchd_load.sh" || true
send_telegram_lifecycle_message $'pairwise 트레이더 시작 완료\n- 상태: 실행 중\n- 텔레그램 봇: 활성\n- 운영 감시: 활성\n- pairwise 주문: 활성\n- core 주문: 비활성\n- pairwise shadow: 비활성\n- 제어 명령: /status /plan /positions'
