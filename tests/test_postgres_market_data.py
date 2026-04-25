import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import gp_crypto_evolution as gp
import postgres_market_data as pg_market_data


class CompletedProcessStub:
    def __init__(self, stdout: str = "", stderr: str = "", returncode: int = 0) -> None:
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode


class PostgresMarketDataTests(unittest.TestCase):
    def test_load_ohlcv_uses_docker_psql_csv_contract(self) -> None:
        csv_payload = (
            "open_time,open,high,low,close,volume,taker_base,taker_quote\n"
            "2026-04-25 09:15:00+00,100,105,99,104,10,5,520\n"
            "2026-04-25 09:20:00+00,104,106,103,105,8,4,420\n"
        )
        config = pg_market_data.PostgresMarketDataConfig(
            user="epic",
            dbname="epic_trading",
            container="epic_trading_db",
        )

        with patch.object(
            pg_market_data.subprocess,
            "run",
            return_value=CompletedProcessStub(stdout=csv_payload),
        ) as run:
            frame = pg_market_data.load_ohlcv("btcusdt", "5m", "2026-04-25", None, config=config)

        self.assertEqual(list(frame.columns), list(pg_market_data.RAW_CACHE_COLUMNS))
        self.assertEqual(len(frame), 2)
        self.assertEqual(str(frame.index.tz), "UTC")
        self.assertAlmostEqual(float(frame.iloc[-1]["close"]), 105.0)
        command = run.call_args.args[0]
        self.assertEqual(Path(command[0]).name, "docker")
        self.assertEqual(command[1:4], ["exec", "-i", "epic_trading_db"])
        self.assertIn("COPY", command[-1])
        self.assertIn("FROM public.candles", command[-1])

    def test_load_funding_returns_empty_when_primary_table_is_empty(self) -> None:
        config = pg_market_data.PostgresMarketDataConfig(container="epic_trading_db")

        def fake_run(command, **kwargs):
            sql = command[-1]
            if "funding_rates" in sql:
                return CompletedProcessStub(stdout='fundingTime,fundingRate\n')
            return CompletedProcessStub(stderr='ERROR: relation "public.funding_rate_snapshots" does not exist', returncode=1)

        with patch.object(pg_market_data.subprocess, "run", side_effect=fake_run):
            frame = pg_market_data.load_funding_rates("BNBUSDT", "2026-01-01", "2026-01-02", config=config)

        self.assertTrue(frame.empty)
        self.assertEqual(list(frame.columns), ["fundingTime", "fundingRate"])

    def test_load_ohlcv_filters_misaligned_and_extreme_price_rows(self) -> None:
        csv_payload = (
            "open_time,open,high,low,close,volume,taker_base,taker_quote\n"
            "2026-03-30 00:00:00+00,66000,66100,65900,66050,10,5,330250\n"
            "2026-03-30 00:05:00+00,66050,66100,66000,66080,10,5,330400\n"
            "2026-03-30 00:10:00+00,66080,66120,66020,66060,10,5,330300\n"
            "2026-03-30 00:15:43+00,100,101,99,100.5,10,5,502\n"
            "2026-03-30 00:20:00+00,614,615,613,614.11,10,5,3070\n"
        )
        filler = "".join(
            f"2026-03-30 {hour:02d}:{minute:02d}:00+00,66000,66100,65900,66000,10,5,330000\n"
            for hour in range(1, 4)
            for minute in range(0, 60, 5)
        )
        config = pg_market_data.PostgresMarketDataConfig(container="epic_trading_db")

        with patch.object(
            pg_market_data.subprocess,
            "run",
            return_value=CompletedProcessStub(stdout=csv_payload + filler),
        ):
            frame = pg_market_data.load_ohlcv("BTCUSDT", "5m", "2026-03-30", None, config=config)

        self.assertNotIn(pd.Timestamp("2026-03-30T00:15:43Z"), frame.index)
        self.assertNotIn(pd.Timestamp("2026-03-30T00:20:00Z"), frame.index)
        self.assertGreater(len(frame), 20)

    def test_gp_load_pair_can_select_postgres_source(self) -> None:
        raw = pd.DataFrame(
            {
                "open": [100.0] * 40,
                "high": [101.0] * 40,
                "low": [99.0] * 40,
                "close": [100.0 + idx for idx in range(40)],
                "volume": [10.0] * 40,
                "taker_base": [5.0] * 40,
                "taker_quote": [500.0] * 40,
            },
            index=pd.date_range("2026-04-24", periods=40, freq="5min", tz="UTC"),
        )

        with (
            patch.dict(pg_market_data.os.environ, {"EPIC_MARKET_DATA_SOURCE": "postgres"}, clear=False),
            patch.object(gp.pg_market_data, "load_ohlcv", return_value=raw),
        ):
            frame = gp.load_pair("BTCUSDT", start="2026-04-24", end=None)

        self.assertFalse(frame.empty)
        self.assertIn("BTCUSDT_close", frame.columns)
        self.assertIn("BTCUSDT_buy_volume_share", frame.columns)


if __name__ == "__main__":
    unittest.main()
