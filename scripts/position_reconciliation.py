#!/usr/bin/env python3
"""Reconcile actual exchange positions against live system state.

Runs every 5 minutes via launchd. Alerts via Telegram on mismatch.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

import ccxt

try:
    from telegram_format import AlertLevel as _TF_AlertLevel, format_alert as _tf_format_alert, should_send as _tf_should_send
    _TF_AVAILABLE = True
except ImportError:
    _TF_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger("position_reconciliation")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PAIRS = ("BTCUSDT", "BNBUSDT")
PAIR_TO_MARKET = {
    "BTCUSDT": "BTC/USDT:USDT",
    "BNBUSDT": "BNB/USDT:USDT",
}
MARKET_TO_PAIR = {v: k for k, v in PAIR_TO_MARKET.items()}
EPSILON = 1e-9

DEFAULT_STATE_PATH = ROOT / "models" / "pairwise_regime_live_state.json"
RECON_OUTPUT_DIR = ROOT / "models"

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def utc_now() -> datetime:
    return datetime.now(UTC)


def iso_now() -> str:
    return utc_now().isoformat()


def send_telegram(message: str, dry_run: bool = False) -> None:
    """Send a Telegram message using BOT_TOKEN / CHAT_ID from env."""
    if dry_run:
        log.info("[DRY-RUN] Telegram: %s", message)
        return
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        log.warning("TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not set; skipping alert")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = urlencode({"chat_id": TELEGRAM_CHAT_ID, "text": message}).encode()
    req = Request(url, data=payload, method="POST")
    try:
        with urlopen(req, timeout=15) as resp:
            body = resp.read()
            log.info("Telegram sent (status=%s)", resp.status)
            return
    except URLError as exc:
        log.error("Telegram send failed: %s", exc)
    except Exception as exc:  # noqa: BLE001
        log.error("Telegram unexpected error: %s", exc)


def get_demo_credentials() -> tuple[str, str]:
    api_key = (
        os.getenv("BINANCE_DEMO_API_KEY", "")
        or os.getenv("BINANCE_TESTNET_API_KEY", "")
        or os.getenv("BINANCE_API_KEY", "")
    )
    secret = (
        os.getenv("BINANCE_DEMO_API_SECRET", "")
        or os.getenv("BINANCE_TESTNET_API_SECRET", "")
        or os.getenv("BINANCE_TESTNET_SECRET", "")
        or os.getenv("BINANCE_SECRET", "")
        or os.getenv("BINANCE_SECRET_KEY", "")
    )
    return api_key, secret


def get_live_credentials() -> tuple[str, str]:
    api_key = os.getenv("BINANCE_LIVE_API_KEY", "") or os.getenv("BINANCE_API_KEY", "")
    secret = (
        os.getenv("BINANCE_LIVE_API_SECRET", "")
        or os.getenv("BINANCE_LIVE_SECRET", "")
        or os.getenv("BINANCE_SECRET", "")
        or os.getenv("BINANCE_SECRET_KEY", "")
    )
    return api_key, secret


def get_exchange(mode: str) -> ccxt.binanceusdm:
    api_key, secret = get_demo_credentials() if mode == "demo" else get_live_credentials()
    exchange = ccxt.binanceusdm(
        {
            "apiKey": api_key,
            "secret": secret,
            "enableRateLimit": True,
            "options": {"defaultType": "swap"},
        }
    )
    if mode == "demo":
        exchange.enable_demo_trading(True)
    return exchange


def fetch_exchange_qty_map(exchange: ccxt.binanceusdm) -> dict[str, float]:
    """Return {pair: signed_qty} for all tracked pairs. Returns {} on error."""
    try:
        positions = exchange.fetch_positions(list(PAIR_TO_MARKET.values()))
    except Exception as exc:  # noqa: BLE001
        log.error("fetch_positions failed: %s", exc)
        return {}

    qty_map: dict[str, float] = {}
    for pos in positions:
        symbol = pos.get("symbol")
        pair = MARKET_TO_PAIR.get(symbol)
        if pair is None:
            continue
        info = pos.get("info", {}) if isinstance(pos.get("info"), dict) else {}
        raw_qty = info.get("positionAmt")
        if raw_qty is None:
            raw_qty = pos.get("contracts", 0.0)
        position_side = info.get("positionSide", "BOTH")
        try:
            qty_float = float(raw_qty)
        except (TypeError, ValueError):
            qty_float = 0.0
        # BOTH-side hedge: positionAmt is already signed; one-way: same.
        # SHORT hedge accounts have negative positionAmt; respect that.
        if position_side == "SHORT":
            qty_float = -abs(qty_float)
        qty_map[pair] = qty_float
    return qty_map


def load_state_qty_map(state_path: Path) -> dict[str, float]:
    """Return {pair: signed_qty} from the live state's latest_live_sync.positions."""
    try:
        with state_path.open() as fh:
            state: dict[str, Any] = json.load(fh)
    except Exception as exc:  # noqa: BLE001
        log.error("Failed to load state file %s: %s", state_path, exc)
        return {}

    positions: dict[str, Any] = state.get("latest_live_sync", {}).get("positions", {})
    qty_map: dict[str, float] = {}
    for pair in PAIRS:
        pos = positions.get(pair)
        if pos is None:
            qty_map[pair] = 0.0
        else:
            try:
                qty_map[pair] = float(pos.get("qty", 0.0))
            except (TypeError, ValueError):
                qty_map[pair] = 0.0
    return qty_map


def write_report(data: dict[str, Any]) -> Path:
    ts = utc_now().strftime("%Y%m%dT%H%M%SZ")
    out_path = RECON_OUTPUT_DIR / f"position_reconciliation_{ts}.json"
    RECON_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.rename(out_path)
    log.info("Wrote reconciliation report: %s", out_path)
    return out_path


# ---------------------------------------------------------------------------
# Core reconciliation
# ---------------------------------------------------------------------------
def reconcile(
    mode: str,
    state_path: Path,
    tolerance: float,
    dry_run: bool,
) -> dict[str, Any]:
    log.info("Starting reconciliation (mode=%s, tolerance=%s, dry_run=%s)", mode, tolerance, dry_run)

    # 1. Load state positions
    state_qty = load_state_qty_map(state_path)
    log.info("State positions: %s", state_qty)

    # 2. Fetch exchange positions
    try:
        exchange = get_exchange(mode)
    except Exception as exc:  # noqa: BLE001
        msg = f"[RECON ERROR] Could not create exchange (mode={mode}): {exc}"
        log.error(msg)
        send_telegram(msg, dry_run=dry_run)
        return {"error": str(exc), "timestamp": iso_now()}

    exchange_qty = fetch_exchange_qty_map(exchange)
    if not exchange_qty and not state_qty:
        log.warning("Both exchange and state returned empty position maps")

    # Fill in zeros for pairs not returned by exchange
    for pair in PAIRS:
        exchange_qty.setdefault(pair, 0.0)

    log.info("Exchange positions: %s", exchange_qty)

    # 3. Compare
    mismatches: list[dict[str, Any]] = []
    pair_results: dict[str, Any] = {}
    for pair in PAIRS:
        exch_q = exchange_qty.get(pair, 0.0)
        state_q = state_qty.get(pair, 0.0)
        delta = exch_q - state_q
        matched = abs(delta) <= tolerance
        pair_results[pair] = {
            "exchange_qty": exch_q,
            "state_qty": state_q,
            "delta": delta,
            "matched": matched,
        }
        if not matched:
            mismatches.append({"pair": pair, "exchange_qty": exch_q, "state_qty": state_q, "delta": delta})
            log.warning(
                "MISMATCH %s: exchange=%.6f state=%.6f delta=%.6f",
                pair,
                exch_q,
                state_q,
                delta,
            )

    # 4. Alert on mismatch
    for mm in mismatches:
        alert = (
            f"[RECON ALERT] {mm['pair']}: "
            f"exchange={mm['exchange_qty']:.6f}, "
            f"state={mm['state_qty']:.6f}, "
            f"delta={mm['delta']:+.6f}"
        )
        log.warning(alert)
        if _TF_AVAILABLE and _tf_should_send(_TF_AlertLevel.HIGH, f"recon-{mm['pair']}"):
            _payload = _tf_format_alert(
                _TF_AlertLevel.HIGH,
                title="포지션 불일치",
                body=f"{mm['pair']}: 거래소={mm['exchange_qty']:.6f}, 상태={mm['state_qty']:.6f}, 차이={mm['delta']:+.6f}",
                context={
                    "📌 종목": mm['pair'],
                    "🏦 거래소": f"{mm['exchange_qty']:.6f}",
                    "💾 상태": f"{mm['state_qty']:.6f}",
                    "△ 차이": f"{mm['delta']:+.6f}",
                },
            )
            send_telegram(_payload["text"], dry_run=dry_run)
        else:
            send_telegram(alert, dry_run=dry_run)

    if not mismatches:
        log.info("All positions matched within tolerance %.6f", tolerance)

    report = {
        "timestamp": iso_now(),
        "mode": mode,
        "tolerance": tolerance,
        "dry_run": dry_run,
        "pairs": pair_results,
        "mismatches": mismatches,
        "all_matched": len(mismatches) == 0,
    }

    # 5. Write report
    try:
        write_report(report)
    except Exception as exc:  # noqa: BLE001
        log.error("Failed to write reconciliation report: %s", exc)

    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Reconcile exchange positions against live state")
    p.add_argument(
        "--mode",
        choices=["demo", "live"],
        default=os.getenv("BINANCE_MODE", os.getenv("PAIRWISE_LIVE_MODE", "demo")),
        help="Exchange mode (default: demo)",
    )
    p.add_argument(
        "--state-path",
        type=Path,
        default=Path(os.getenv("PAIRWISE_LIVE_STATE_PATH", str(DEFAULT_STATE_PATH))),
        help="Path to pairwise_regime_live_state.json",
    )
    p.add_argument(
        "--tolerance",
        type=float,
        default=float(os.getenv("RECON_TOLERANCE", "0.001")),
        help="Qty mismatch tolerance (default: 0.001)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Log alerts but do not send Telegram messages",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()
    report = reconcile(
        mode=args.mode,
        state_path=args.state_path,
        tolerance=args.tolerance,
        dry_run=args.dry_run,
    )
    if not report.get("all_matched", True):
        sys.exit(1)


if __name__ == "__main__":
    main()
