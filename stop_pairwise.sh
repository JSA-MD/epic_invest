#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$ROOT_DIR/scripts/trader_service.sh"

"$ROOT_DIR/watchdog_launchd_unload.sh" || true
"$ROOT_DIR/pairwise_shadow_launchd_unload.sh" || true
stop_pairwise_processes
stop_core_processes
clear_active_runtime_profile
"$ROOT_DIR/telegram_launchd_unload.sh" || true
send_telegram_lifecycle_message $'pairwise 트레이더 종료 완료\n- 상태: 중지됨\n- 텔레그램 봇: 비활성\n- 운영 감시: 비활성\n- pairwise 주문: 비활성\n- core 주문: 비활성\n- pairwise shadow: 비활성\n- 종료 보호주문: 설치 시도 완료'
