#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$ROOT_DIR/scripts/trader_service.sh"

stop_trader
"$ROOT_DIR/telegram_launchd_unload.sh"
send_telegram_lifecycle_message $'트레이더 종료 완료\n- 상태: 중지됨\n- 텔레그램 봇: 비활성\n- 종료 보호주문: 설치 시도 완료'
