#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
PLIST_PATH="$ROOT_DIR/scripts/com.epicinvest.telegram-bot.plist"
DOMAIN="gui/$(id -u)"
LABEL="com.epicinvest.telegram-bot"

echo "🚀 launchd 텔레그램 봇 로드"

launchctl bootout "$DOMAIN/$LABEL" >/dev/null 2>&1 || true
launchctl bootstrap "$DOMAIN" "$PLIST_PATH"
launchctl enable "$DOMAIN/$LABEL" >/dev/null 2>&1 || true
launchctl kickstart -k "$DOMAIN/$LABEL"

echo "  Label: $LABEL"
echo "  Domain: $DOMAIN"
echo "  Plist: $PLIST_PATH"
echo "  Log: /tmp/epic-invest-telegram-bot-launchd.log"

