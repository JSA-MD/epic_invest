#!/usr/bin/env bash
# Install / uninstall the position reconciliation launchd job (runs every 5 min).
# Usage:
#   ./scripts/install_position_reconciliation.sh load      # install and enable
#   ./scripts/install_position_reconciliation.sh unload    # disable and remove
#   ./scripts/install_position_reconciliation.sh run       # run once now (dry-run)
#   ./scripts/install_position_reconciliation.sh run-live  # run once now (live alerts)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PLIST_SRC="$REPO_ROOT/scripts/com.epicinvest.position-reconciliation.plist"
PLIST_LABEL="com.epicinvest.position-reconciliation"
PLIST_DEST="$HOME/Library/LaunchAgents/$PLIST_LABEL.plist"
PYTHON="$REPO_ROOT/.venv/bin/python"
SCRIPT="$REPO_ROOT/scripts/position_reconciliation.py"

cmd="${1:-help}"

case "$cmd" in
load)
    echo "[install_position_reconciliation] Copying plist to ~/Library/LaunchAgents/ ..."
    cp "$PLIST_SRC" "$PLIST_DEST"

    # Unload first in case it was previously loaded (ignore errors)
    launchctl unload "$PLIST_DEST" 2>/dev/null || true

    echo "[install_position_reconciliation] Loading launchd job ..."
    launchctl load "$PLIST_DEST"

    echo "[install_position_reconciliation] Done. Job runs every 300s."
    echo "  Log: /tmp/epic-invest-recon.log"
    echo "  Check status: launchctl list | grep $PLIST_LABEL"
    ;;

unload)
    echo "[install_position_reconciliation] Unloading launchd job ..."
    launchctl unload "$PLIST_DEST" 2>/dev/null || true
    rm -f "$PLIST_DEST"
    echo "[install_position_reconciliation] Job removed."
    ;;

run)
    echo "[install_position_reconciliation] Running reconciliation now (dry-run) ..."
    cd "$REPO_ROOT"
    "$PYTHON" "$SCRIPT" --dry-run
    ;;

run-live)
    echo "[install_position_reconciliation] Running reconciliation now (LIVE - will send Telegram) ..."
    cd "$REPO_ROOT"
    "$PYTHON" "$SCRIPT"
    ;;

help|*)
    echo "Usage: $0 {load|unload|run|run-live}"
    echo ""
    echo "  load      Copy plist to ~/Library/LaunchAgents and enable the job"
    echo "  unload    Disable and remove the job"
    echo "  run       Run once now in --dry-run mode (no Telegram)"
    echo "  run-live  Run once now and send Telegram alerts if mismatches found"
    ;;
esac
