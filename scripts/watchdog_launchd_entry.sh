#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
ENTRY_SCRIPT="$ROOT_DIR/scripts/operation_watchdog.py"
INTERVAL_SECONDS="${WATCHDOG_INTERVAL_SECONDS:-60}"

exec /bin/bash -lc "cd \"$ROOT_DIR\" && \"$PYTHON_BIN\" -u \"$ENTRY_SCRIPT\" loop --interval-seconds \"$INTERVAL_SECONDS\" --recover"
