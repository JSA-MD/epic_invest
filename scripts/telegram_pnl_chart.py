#!/usr/bin/env python3
"""Component 1: PnL chart generator and Telegram photo sender.

Usage:
    python telegram_pnl_chart.py --window 30            # save only
    python telegram_pnl_chart.py --window 30 --send     # save + send
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from io import BytesIO
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Paths & env
# ---------------------------------------------------------------------------

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parents[1]
PNL_JSON_PATH = ROOT_DIR / "models" / "live_actual_pnl_30d.json"

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()

KST = timezone(timedelta(hours=9))


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_daily_pnl(window_days: int = 30, path: Path = PNL_JSON_PATH) -> list[dict]:
    """Return aggregated daily total PnL rows within *window_days*, oldest first."""
    raw = json.loads(path.read_text())
    rows: list[dict] = raw.get("daily_pnl_live", [])

    # Aggregate across pairs per date
    by_date: dict[str, float] = {}
    for row in rows:
        d = row["date"]
        by_date[d] = by_date.get(d, 0.0) + float(row.get("total", 0.0))

    cutoff_date = (datetime.now(tz=timezone.utc) - timedelta(days=window_days)).date()
    filtered = {
        d: v for d, v in sorted(by_date.items())
        if datetime.fromisoformat(d).date() >= cutoff_date
    }
    return [{"date": d, "total": v} for d, v in filtered.items()]


# ---------------------------------------------------------------------------
# Chart generation
# ---------------------------------------------------------------------------


def generate_pnl_chart_png(
    window_days: int = 30,
    out_path: Path = Path("/tmp/epic-pnl.png"),
) -> Path:
    """Generate a cumulative PnL chart PNG and return its path.

    - 누적 PnL 라인: 양수 구간 초록, 음수 구간 빨강
    - 7일 이동평균 파선
    - KST 기준 x축 레이블
    - 격자선 포함
    """
    rows = load_daily_pnl(window_days=window_days)

    if not rows:
        # 데이터 없음 — 빈 차트 저장
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes, fontsize=14)
        fig.tight_layout()
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(out_path), dpi=120, bbox_inches="tight")
        plt.close(fig)
        return out_path

    dates = [datetime.fromisoformat(r["date"]).replace(tzinfo=timezone.utc).astimezone(KST) for r in rows]
    daily_vals = np.array([r["total"] for r in rows], dtype=float)
    cumulative = np.cumsum(daily_vals)

    # 7-day rolling average of cumulative PnL
    window_ma = min(7, len(cumulative))
    rolling_avg = np.convolve(cumulative, np.ones(window_ma) / window_ma, mode="full")[: len(cumulative)]

    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#0f1117")

    # Positive / negative coloring under the line
    pos_mask = cumulative >= 0
    neg_mask = ~pos_mask

    # Draw as a single line with segment coloring
    for i in range(1, len(dates)):
        color = "#00c853" if cumulative[i] >= 0 else "#d50000"
        ax.plot(dates[i - 1 : i + 1], cumulative[i - 1 : i + 1], color=color, linewidth=2.0)

    # Fill under curve
    ax.fill_between(
        dates, cumulative, 0,
        where=pos_mask, alpha=0.15, color="#00c853", interpolate=True
    )
    ax.fill_between(
        dates, cumulative, 0,
        where=neg_mask, alpha=0.15, color="#d50000", interpolate=True
    )

    # 7-day rolling average
    ax.plot(dates, rolling_avg, color="#ffeb3b", linewidth=1.2, linestyle="--", label="7d MA", alpha=0.85)

    # Zero line
    ax.axhline(0, color="#888888", linewidth=0.8, linestyle="-")

    # Axes styling
    ax.set_title(f"Epic Invest — Cumulative PnL (last {window_days}d)", color="white", fontsize=13, pad=10)
    ax.set_ylabel("Cumulative PnL (USDT)", color="#cccccc", fontsize=10)
    ax.tick_params(colors="#aaaaaa")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333333")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d", tz=KST))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0, tz=KST))
    fig.autofmt_xdate(rotation=30, ha="right")
    ax.yaxis.get_major_formatter().set_useOffset(False)
    ax.grid(True, color="#222222", linewidth=0.5, linestyle="--", alpha=0.7)
    ax.legend(facecolor="#1a1a2e", edgecolor="#444444", labelcolor="white", fontsize=9)

    # Final PnL annotation
    final_pnl = cumulative[-1]
    color_final = "#00c853" if final_pnl >= 0 else "#d50000"
    ax.annotate(
        f"{final_pnl:+.2f} USDT",
        xy=(dates[-1], final_pnl),
        xytext=(-60, 12),
        textcoords="offset points",
        color=color_final,
        fontsize=10,
        fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=color_final, lw=1.0),
    )

    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Telegram sendPhoto
# ---------------------------------------------------------------------------


def send_chart_to_telegram(
    chat_id: int | str,
    png_path: Path,
    caption: str = "",
    token: str | None = None,
) -> dict:
    """Send a PNG file via Telegram sendPhoto multipart/form-data.

    Returns the Telegram API response dict.
    """
    token = token or TELEGRAM_BOT_TOKEN
    if not token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN이 설정되지 않았습니다.")

    png_path = Path(png_path)
    if not png_path.exists():
        raise FileNotFoundError(f"PNG 파일을 찾을 수 없습니다: {png_path}")

    # Build multipart body manually (no external deps)
    boundary = "----EpicInvestBoundary"
    body_parts: list[bytes] = []

    def field(name: str, value: str) -> bytes:
        return (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="{name}"\r\n\r\n'
            f"{value}\r\n"
        ).encode("utf-8")

    body_parts.append(field("chat_id", str(chat_id)))
    if caption:
        body_parts.append(field("caption", caption[:1024]))
        body_parts.append(field("parse_mode", "Markdown"))

    # File part
    img_bytes = png_path.read_bytes()
    file_part = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="photo"; filename="{png_path.name}"\r\n'
        f"Content-Type: image/png\r\n\r\n"
    ).encode("utf-8") + img_bytes + b"\r\n"
    body_parts.append(file_part)
    body_parts.append(f"--{boundary}--\r\n".encode("utf-8"))

    body = b"".join(body_parts)
    url = f"https://api.telegram.org/bot{token}/sendPhoto"
    req = Request(
        url,
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    try:
        with urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read().decode("utf-8"))
    except (HTTPError, URLError) as exc:
        raise RuntimeError(f"Telegram sendPhoto 오류: {exc}") from exc
    if not result.get("ok"):
        raise RuntimeError(f"Telegram API 실패: {result}")
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_caption(window_days: int, rows: list[dict]) -> str:
    """Build a Korean caption string for the digest photo."""
    from datetime import datetime as dt
    now_kst = datetime.now(tz=KST)
    if not rows:
        return f"📊 누적 PnL 차트 (최근 {window_days}일)\n⏰ {now_kst.strftime('%m/%d %H:%M')} KST"
    daily_vals = [r["total"] for r in rows]
    cumulative_total = sum(daily_vals)
    today_pnl = daily_vals[-1] if daily_vals else 0.0
    color = "🟢" if cumulative_total >= 0 else "🔴"
    return (
        f"{color} *Epic Invest 누적 PnL ({window_days}일)*\n"
        f"오늘: {today_pnl:+.2f} USDT | 누적: {cumulative_total:+.2f} USDT\n"
        f"⏰ {now_kst.strftime('%m/%d %H:%M')} KST"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Epic Invest PnL 차트 생성기")
    parser.add_argument("--window", type=int, default=30, help="표시할 일 수 (기본: 30)")
    parser.add_argument("--out", type=Path, default=Path("/tmp/epic-pnl.png"), help="저장 경로")
    parser.add_argument("--send", action="store_true", help="Telegram으로 전송")
    parser.add_argument("--chat-id", type=str, default="", help="Telegram chat ID (기본: 환경변수)")
    args = parser.parse_args()

    print(f"차트 생성 중 (최근 {args.window}일)...")
    rows = load_daily_pnl(window_days=args.window)
    png_path = generate_pnl_chart_png(window_days=args.window, out_path=args.out)
    size_kb = png_path.stat().st_size / 1024
    print(f"저장 완료: {png_path} ({size_kb:.1f} KB)")

    if args.send:
        chat_id = args.chat_id or TELEGRAM_CHAT_ID
        if not chat_id:
            print("오류: TELEGRAM_CHAT_ID가 설정되지 않았습니다.", file=sys.stderr)
            sys.exit(1)
        caption = _build_caption(args.window, rows)
        print(f"Telegram 전송 중 (chat_id={chat_id})...")
        send_chart_to_telegram(int(chat_id), png_path, caption=caption)
        print("전송 완료.")


if __name__ == "__main__":
    main()
