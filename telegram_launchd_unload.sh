#!/usr/bin/env bash
set -euo pipefail

DOMAIN="gui/$(id -u)"
LABEL="com.epicinvest.telegram-bot"

echo "🛑 launchd 텔레그램 봇 언로드"
launchctl bootout "$DOMAIN/$LABEL" >/dev/null 2>&1 || true
echo "  Label: $LABEL"
