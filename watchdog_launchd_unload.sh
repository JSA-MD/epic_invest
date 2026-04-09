#!/usr/bin/env bash
set -euo pipefail

DOMAIN="gui/$(id -u)"
LABEL="com.epicinvest.watchdog"

echo "🛑 launchd 운영 감시 언로드"
launchctl bootout "$DOMAIN/$LABEL" >/dev/null 2>&1 || true
echo "  Label: $LABEL"
