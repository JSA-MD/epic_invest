#!/usr/bin/env bash
set -euo pipefail

DOMAIN="gui/$(id -u)"
LABEL="com.epicinvest.pairwise-trader"

echo "🛑 launchd pairwise 트레이더 언로드"
launchctl bootout "$DOMAIN/$LABEL" >/dev/null 2>&1 || true
echo "  Label: $LABEL"
