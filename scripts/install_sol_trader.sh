#!/usr/bin/env bash
# Install / uninstall the SOL pairwise trader launchd job.
# Usage:
#   ./scripts/install_sol_trader.sh load    # install and enable (DO NOT run until summary candidate is validated)
#   ./scripts/install_sol_trader.sh unload  # disable and remove

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PLIST_SRC="$REPO_ROOT/scripts/com.epicinvest.sol-trader.plist"
PLIST_LABEL="com.epicinvest.sol-trader"
PLIST_DEST="$HOME/Library/LaunchAgents/$PLIST_LABEL.plist"
LOG_DIR="$REPO_ROOT/logs"

cmd="${1:-help}"

case "$cmd" in
load)
    echo "[install_sol_trader] Creating log directory ..."
    mkdir -p "$LOG_DIR"

    echo "[install_sol_trader] Copying plist to ~/Library/LaunchAgents/ ..."
    cp "$PLIST_SRC" "$PLIST_DEST"

    # Unload first in case it was previously loaded (ignore errors)
    launchctl unload "$PLIST_DEST" 2>/dev/null || true

    echo "[install_sol_trader] Loading launchd job ..."
    launchctl load "$PLIST_DEST"

    echo "[install_sol_trader] Done. SOL trader is running."
    echo "  Log: $LOG_DIR/sol_live_service.log"
    echo "  Decisions: $LOG_DIR/sol_decisions.jsonl"
    echo "  Check status: launchctl list | grep $PLIST_LABEL"
    ;;

unload)
    echo "[install_sol_trader] Unloading launchd job ..."
    launchctl unload "$PLIST_DEST" 2>/dev/null || true
    rm -f "$PLIST_DEST"
    echo "[install_sol_trader] Job removed."
    ;;

help|*)
    echo "Usage: $0 {load|unload}"
    echo ""
    echo "  load    Copy plist to ~/Library/LaunchAgents and enable the SOL trader"
    echo "  unload  Disable and remove the SOL trader job"
    echo ""
    echo "BLOCKER: Do not load until models/sol_pairwise_candidate_summary.json"
    echo "         has been generated and validated via search_pair_subset_regime_mixture."
    ;;
esac
