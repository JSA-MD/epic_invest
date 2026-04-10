#!/usr/bin/env bash
set -euo pipefail

DOMAIN="gui/$(id -u)"
LABEL="com.epicinvest.pairwise-shadow-trader"

echo "🛑 launchd pairwise shadow 언로드"
launchctl bootout "$DOMAIN/$LABEL" >/dev/null 2>&1 || true
echo "  Label: $LABEL"
