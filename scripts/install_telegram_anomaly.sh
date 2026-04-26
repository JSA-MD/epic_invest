#!/usr/bin/env bash
# 이상 감지 알람 launchd 에이전트 설치 / 제거 스크립트
# Usage: ./install_telegram_anomaly.sh [load|unload|status]

set -euo pipefail

PLIST_SRC="$(cd "$(dirname "$0")" && pwd)/com.epicinvest.telegram-anomaly.plist"
PLIST_DST="${HOME}/Library/LaunchAgents/com.epicinvest.telegram-anomaly.plist"
LABEL="com.epicinvest.telegram-anomaly"
LOG_DIR="$(cd "$(dirname "$0")/.." && pwd)/logs"

action="${1:-load}"

mkdir -p "$LOG_DIR"

case "$action" in
    load)
        echo "이상 감지 알람 에이전트 설치 중..."
        cp "$PLIST_SRC" "$PLIST_DST"
        launchctl load -w "$PLIST_DST"
        echo "설치 완료: $LABEL"
        echo "스케줄: 매일 09:30 KST (00:30 UTC)"
        ;;
    unload)
        echo "이상 감지 알람 에이전트 제거 중..."
        launchctl unload -w "$PLIST_DST" 2>/dev/null || true
        rm -f "$PLIST_DST"
        echo "제거 완료: $LABEL"
        ;;
    status)
        echo "=== launchctl 상태 ==="
        launchctl list | grep "$LABEL" || echo "실행 중이 아님"
        echo ""
        echo "=== 마지막 로그 ==="
        tail -20 "$LOG_DIR/telegram-anomaly.log" 2>/dev/null || echo "(로그 없음)"
        ;;
    *)
        echo "사용법: $0 [load|unload|status]"
        exit 1
        ;;
esac
