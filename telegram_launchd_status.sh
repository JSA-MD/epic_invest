#!/usr/bin/env bash
set -euo pipefail

DOMAIN="gui/$(id -u)"
LABEL="com.epicinvest.telegram-bot"

launchctl print "$DOMAIN/$LABEL"
