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


def fetch_exchange_qty_map(exchange: ccxt.binanceusdm) -> dict[str, float] | None:
    """Return {pair: signed_qty} for all tracked pairs. Returns None on error.

    A None return means the exchange could not be queried; callers must NOT
    treat that as "exchange is flat" because the phantom-mismatch filter would
    then mistake fetch outages for genuine flat positions and silently swallow
    real divergences.
    """
    try:
        positions = exchange.fetch_positions(list(PAIR_TO_MARKET.values()))
    except Exception as exc:  # noqa: BLE001
        log.error("fetch_positions failed: %s", exc)
        return None

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


def force_close_position(
    exchange: ccxt.binanceusdm,
    pair: str,
    current_qty: float,
    *,
    dry_run: bool = False,
) -> dict[str, Any] | None:
    """Submit a reduce-only market order to flatten an unwanted exchange position.

    Returns the order dict on success, None on no-op (qty below epsilon),
    or {"error": ...} on failure. Never raises — callers may continue to
    next pair.
    """
    if abs(current_qty) < EPSILON:
        return None
    market = PAIR_TO_MARKET.get(pair)
    if market is None:
        return {"error": f"unknown_pair_{pair}"}
    side = "sell" if current_qty > 0 else "buy"
    qty = abs(float(current_qty))
    if dry_run:
        log.info("[DRY-RUN] force-close %s %s qty=%.6f (reduceOnly)", pair, side, qty)
        return {"dry_run": True, "pair": pair, "side": side, "qty": qty}
    try:
        order = exchange.create_order(
            market,
            type="market",
            side=side,
            amount=qty,
            params={"reduceOnly": True},
        )
        log.warning("force-closed %s %s qty=%.6f order_id=%s", pair, side, qty, order.get("id"))
        return order
    except Exception as exc:  # noqa: BLE001
        log.error("force-close failed for %s: %s", pair, exc)
        return {"error": str(exc), "pair": pair, "side": side, "qty": qty}


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
    *,
    force_close_on_mismatch: bool = False,
) -> dict[str, Any]:
    log.info(
        "Starting reconciliation (mode=%s, tolerance=%s, dry_run=%s, force_close=%s)",
        mode,
        tolerance,
        dry_run,
        force_close_on_mismatch,
    )

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

    raw_exchange_qty = fetch_exchange_qty_map(exchange)
    exchange_fetch_failed = raw_exchange_qty is None
    if exchange_fetch_failed:
        # Without exchange ground truth we cannot trust comparison results.
        # Emit a high-severity alert immediately so the outage is visible, then
        # disable the phantom filter below so any state-side mismatch still
        # surfaces as a normal alert (defence-in-depth).
        err_body = f"mode={mode}: 거래소 포지션 조회 실패. 비교 결과 신뢰 불가."
        log.error("[RECON ERROR] %s", err_body)
        if _TF_AVAILABLE and _tf_should_send(_TF_AlertLevel.HIGH, "recon-fetch-failed"):
            _payload = _tf_format_alert(
                _TF_AlertLevel.HIGH,
                title="Recon 거래소 조회 실패",
                body=err_body,
            )
            send_telegram(_payload["text"], dry_run=dry_run)
        else:
            send_telegram(f"[RECON ERROR] {err_body}", dry_run=dry_run)
        exchange_qty: dict[str, float] = {}
    else:
        exchange_qty = raw_exchange_qty
    if not exchange_qty and not state_qty and not exchange_fetch_failed:
        log.warning("Both exchange and state returned empty position maps")

    # Fill in zeros for pairs not returned by exchange (only meaningful when
    # the fetch succeeded; on fetch failure these zeros are placeholders that
    # the phantom filter must NOT treat as authoritative flat positions).
    for pair in PAIRS:
        exchange_qty.setdefault(pair, 0.0)

    log.info(
        "Exchange positions: %s%s",
        exchange_qty,
        " (FETCH FAILED — values are placeholders)" if exchange_fetch_failed else "",
    )

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
    # Identify phantom mismatches: demo mode, exchange is flat, state cache claims a
    # position.  This happens when recon lands in the window between pairwise_regime_live.py
    # writing the pre-trade snapshot (line 1776) and the post-trade zeroing (line 2174).
    # We suppress the Telegram alert for phantoms to stop the every-5-min spam, but we
    # still record them in both `mismatches` and `phantom_mismatches` for audit.
    phantom_mismatches: list[dict[str, Any]] = []
    phantom_pairs: set[str] = set()
    # Skip phantom filtering entirely on fetch failure — placeholder zeros
    # would otherwise be misread as authoritative flat positions and silence
    # alerts for genuine state-side divergences during exchange outages.
    if not exchange_fetch_failed:
        for mm in mismatches:
            if (
                mode == "demo"
                and abs(mm["exchange_qty"]) < EPSILON
                and abs(mm["state_qty"]) >= EPSILON
            ):
                phantom_mismatches.append(mm)
                phantom_pairs.add(mm["pair"])
                log.warning(
                    "Suppressed demo phantom state-only mismatch %s: exchange=%.6f state=%.6f"
                    " (pre-trade snapshot race in pairwise_regime_live.py:1776)",
                    mm["pair"],
                    mm["exchange_qty"],
                    mm["state_qty"],
                )

    for mm in mismatches:
        if mm["pair"] in phantom_pairs:
            # Already logged above; do not send Telegram for phantoms.
            continue
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

    # 4b. Force-close on mismatch (Stage 0 EOD reconcile per López de Prado/Chan)
    force_close_results: list[dict[str, Any]] = []
    if mismatches and force_close_on_mismatch:
        for mm in mismatches:
            if mm["pair"] in phantom_pairs:
                # Phantom: exchange is already flat, nothing to close.
                force_close_results.append(
                    {"pair": mm["pair"], "skipped": "phantom_demo_state_only"}
                )
                continue
            exch_q = float(mm.get("exchange_qty", 0.0) or 0.0)
            if abs(exch_q) < EPSILON:
                # state thinks we have a position but exchange is flat — nothing to close
                force_close_results.append(
                    {"pair": mm["pair"], "skipped": "exchange_already_flat"}
                )
                continue
            result = force_close_position(exchange, mm["pair"], exch_q, dry_run=dry_run)
            force_close_results.append({"pair": mm["pair"], "result": result})
            if _TF_AVAILABLE and _tf_should_send(_TF_AlertLevel.HIGH, f"recon-close-{mm['pair']}"):
                _payload = _tf_format_alert(
                    _TF_AlertLevel.HIGH,
                    title="강제 청산 실행",
                    body=f"{mm['pair']}: 거래소 {exch_q:+.6f} 강제 reduceOnly 청산",
                    context={"📌 종목": mm["pair"], "🏦 거래소": f"{exch_q:+.6f}"},
                )
                send_telegram(_payload["text"], dry_run=dry_run)

    # all_matched MUST be False when the exchange fetch failed, even if
    # mismatches happens to be empty (it would be empty when state is also
    # flat, since exchange_qty was filled with placeholder zeros). Otherwise
    # downstream consumers reading the report could mistake an outage for a
    # clean reconciliation and mute alerts that are still warranted.
    report = {
        "timestamp": iso_now(),
        "mode": mode,
        "tolerance": tolerance,
        "dry_run": dry_run,
        "force_close_on_mismatch": force_close_on_mismatch,
        "exchange_fetch_failed": exchange_fetch_failed,
        "pairs": pair_results,
        "mismatches": mismatches,
        "phantom_mismatches": phantom_mismatches,
        "force_close_results": force_close_results,
        "all_matched": (len(mismatches) == 0) and not exchange_fetch_failed,
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
    p.add_argument(
        "--force-close-on-mismatch",
        action="store_true",
        default=os.getenv("RECON_FORCE_CLOSE_ON_MISMATCH", "0").strip().lower()
        in {"1", "true", "yes", "on"},
        help=(
            "Submit reduce-only market orders to flatten any exchange position "
            "that diverges from state by more than --tolerance. Honours --dry-run."
        ),
    )
    return p


def main() -> None:
    args = build_parser().parse_args()
    report = reconcile(
        mode=args.mode,
        state_path=args.state_path,
        tolerance=args.tolerance,
        dry_run=args.dry_run,
        force_close_on_mismatch=args.force_close_on_mismatch,
    )
    if not report.get("all_matched", True):
        sys.exit(1)


if __name__ == "__main__":
    main()
