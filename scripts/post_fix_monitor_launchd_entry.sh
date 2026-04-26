#!/usr/bin/env bash
# post_fix_monitor_launchd_entry.sh
# launchd entry point for the post-fix evolution monitor.
#
# launchd fires at :15 every hour. This script self-gates to KST 09:xx with
# catch-up support (runs even if system was asleep at 09:15 KST).
# Dedup is handled by monitor_post_fix_evolution.py via /tmp state file.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

_KST_HOUR=$(TZ=Asia/Seoul date +%H)

# Force base-10 (avoid octal misparse of 08/09)
if (( 10#$_KST_HOUR < 9 )); then
    exit 0
fi

PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
ENTRY_SCRIPT="$ROOT_DIR/scripts/monitor_post_fix_evolution.py"
ENV_FILE="$ROOT_DIR/.env"
RUNTIME_ENV_PATH="${PAIRWISE_LAUNCHD_ENV_PATH:-$ROOT_DIR/models/pairwise_live_launchd_env.sh}"

if [[ -f "$RUNTIME_ENV_PATH" ]]; then
    # shellcheck disable=SC1090
    source "$RUNTIME_ENV_PATH"
fi

if [[ -f "$ENV_FILE" ]]; then
    # shellcheck disable=SC1090
    set -a
    source "$ENV_FILE"
    set +a
fi

cd "$ROOT_DIR"
exec "$PYTHON_BIN" -u "$ENTRY_SCRIPT"
