#!/usr/bin/env bash
set -euo pipefail

DOMAIN="gui/$(id -u)"
LABEL="com.epicinvest.pairwise-trader"

launchctl print "$DOMAIN/$LABEL"
