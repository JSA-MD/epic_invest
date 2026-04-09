#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
ENTRY_SCRIPT="$ROOT_DIR/scripts/rotation_target_050_live.py"
POLL_SECONDS="${TRADER_POLL_SECONDS:-60}"

exec /bin/bash -lc "cd \"$ROOT_DIR\" && \"$PYTHON_BIN\" -u \"$ENTRY_SCRIPT\" loop --execute --poll-seconds \"$POLL_SECONDS\""
