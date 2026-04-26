#!/usr/bin/env python3
"""Component 2: Chart-on-top attachment layer for the daily digest.

This module bridges telegram_pnl_chart.py with the daily digest flow.
When T2's telegram_daily_digest.py exists, it can import from here.
Until then, this script runs standalone to attach a chart to any digest text.

Usage:
    python telegram_chart_attach.py --caption "오늘의 요약" --send
    python telegram_chart_attach.py --window 30 --send
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from dotenv import load_dotenv

from telegram_pnl_chart import (
    generate_pnl_chart_png,
    load_daily_pnl,
    send_chart_to_telegram,
    _build_caption,
)
from telegram_smart_indicators import compute_smart_indicators

load_dotenv()

KST = timezone(timedelta(hours=9))
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()


def attach_chart_to_digest(
    chat_id: int | str,
    window_days: int = 30,
    extra_caption: str = "",
    out_path: Path = Path("/tmp/epic-pnl.png"),
) -> Path:
    """Generate PnL chart and send it with an enriched caption.

    This is the integration point for T2's telegram_daily_digest.py:

        from telegram_chart_attach import attach_chart_to_digest
        attach_chart_to_digest(chat_id, window_days=30, extra_caption=digest_text)

    Returns:
        Path to the saved PNG file.
    """
    rows = load_daily_pnl(window_days=window_days)
    png_path = generate_pnl_chart_png(window_days=window_days, out_path=out_path)

    # Build base caption
    base = _build_caption(window_days, rows)

    # Append smart indicators
    indicators = compute_smart_indicators(rows)
    color = indicators.get("color_indicator", "")
    streak = indicators.get("day_streak", 0)
    vs_7d = indicators.get("today_vs_7d_avg_pct", 0.0)
    vs_30d = indicators.get("today_vs_30d_avg_pct", 0.0)

    streak_label = f"{'상승' if streak > 0 else '하락'} {abs(streak)}일 연속"
    smart_block = (
        f"{color} 7일 평균 대비: {vs_7d:+.1f}%  |  30일 평균 대비: {vs_30d:+.1f}%\n"
        f"연속 흐름: {streak_label}"
    )

    # Merge: base + smart + extra
    parts = [base, smart_block]
    if extra_caption.strip():
        parts.append(extra_caption.strip())
    caption = "\n".join(parts)

    send_chart_to_telegram(int(chat_id), png_path, caption=caption)
    return png_path


def main() -> None:
    parser = argparse.ArgumentParser(description="일일 다이제스트에 PnL 차트 첨부")
    parser.add_argument("--window", type=int, default=30, help="표시 기간 (일)")
    parser.add_argument("--caption", type=str, default="", help="추가 캡션 텍스트")
    parser.add_argument("--out", type=Path, default=Path("/tmp/epic-pnl.png"), help="PNG 저장 경로")
    parser.add_argument("--send", action="store_true", help="Telegram으로 전송")
    parser.add_argument("--chat-id", type=str, default="", help="Telegram chat ID")
    args = parser.parse_args()

    rows = load_daily_pnl(window_days=args.window)
    png_path = generate_pnl_chart_png(window_days=args.window, out_path=args.out)
    size_kb = png_path.stat().st_size / 1024
    print(f"차트 저장: {png_path} ({size_kb:.1f} KB)")

    if args.send:
        chat_id = args.chat_id or TELEGRAM_CHAT_ID
        if not chat_id:
            print("오류: TELEGRAM_CHAT_ID가 설정되지 않았습니다.", file=sys.stderr)
            sys.exit(1)
        attach_chart_to_digest(int(chat_id), args.window, args.caption, args.out)
        print("차트 전송 완료.")
    else:
        caption = _build_caption(args.window, rows)
        print(f"캡션 미리보기:\n{caption}")


if __name__ == "__main__":
    main()
