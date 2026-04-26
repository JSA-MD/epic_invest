#!/usr/bin/env bash
# Install / uninstall the weekly validation launchd job.
# Usage:
#   ./scripts/install_weekly_validation.sh load    # install and enable
#   ./scripts/install_weekly_validation.sh unload  # disable and remove
#   ./scripts/install_weekly_validation.sh run     # run once right now (foreground)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PLIST_SRC="$REPO_ROOT/scripts/com.epicinvest.weekly-validation.plist"
PLIST_LABEL="com.epicinvest.weekly-validation"
PLIST_DEST="$HOME/Library/LaunchAgents/$PLIST_LABEL.plist"
LOG_DIR="$REPO_ROOT/logs"
PYTHON="$REPO_ROOT/.venv/bin/python"
SCRIPT="$REPO_ROOT/scripts/weekly_validation.py"

cmd="${1:-help}"

case "$cmd" in
load)
    echo "[install_weekly_validation] Creating log directory ..."
    mkdir -p "$LOG_DIR"

    echo "[install_weekly_validation] Copying plist to ~/Library/LaunchAgents/ ..."
    cp "$PLIST_SRC" "$PLIST_DEST"

    # Unload first in case it was previously loaded (ignore errors)
    launchctl unload "$PLIST_DEST" 2>/dev/null || true

    echo "[install_weekly_validation] Loading launchd job ..."
    launchctl load "$PLIST_DEST"

    echo "[install_weekly_validation] Done. Job will run every Sunday at 00:00 UTC (09:00 KST)."
    echo "  Log: $LOG_DIR/weekly_validation.log"
    echo "  Check status: launchctl list | grep $PLIST_LABEL"
    ;;

unload)
    echo "[install_weekly_validation] Unloading launchd job ..."
    launchctl unload "$PLIST_DEST" 2>/dev/null || true
    rm -f "$PLIST_DEST"
    echo "[install_weekly_validation] Job removed."
    ;;

run)
    echo "[install_weekly_validation] Running weekly validation now (dry-run) ..."
    cd "$REPO_ROOT"
    "$PYTHON" "$SCRIPT" --dry-run
    ;;

run-live)
    echo "[install_weekly_validation] Running weekly validation now (LIVE - will send Telegram) ..."
    cd "$REPO_ROOT"
    "$PYTHON" "$SCRIPT"
    ;;

help|*)
    echo "Usage: $0 {load|unload|run|run-live}"
    echo ""
    echo "  load      Copy plist to ~/Library/LaunchAgents and enable the job"
    echo "  unload    Disable and remove the job"
    echo "  run       Run once right now in --dry-run mode (no Telegram)"
    echo "  run-live  Run once right now and send Telegram message"
    ;;
esac
