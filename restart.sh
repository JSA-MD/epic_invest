#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$ROOT_DIR/scripts/trader_service.sh"

restart_trader
