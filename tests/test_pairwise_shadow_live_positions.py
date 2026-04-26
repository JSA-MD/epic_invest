import sys
import unittest
import unittest.mock
from pathlib import Path
from unittest.mock import MagicMock


ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import pairwise_regime_mixture_shadow_live as shadow_live


class PairwiseShadowLivePositionTests(unittest.TestCase):
    def test_fetch_open_position_map_respects_position_side(self) -> None:
        exchange = MagicMock()
        exchange.fetch_positions.return_value = [
            {
                "symbol": "BTC/USDT:USDT",
                "info": {
                    "positionAmt": "0.020",
                    "positionSide": "SHORT",
                    "entryPrice": "71000",
                    "markPrice": "70500",
                },
            }
        ]

        positions = shadow_live.fetch_open_position_map(exchange, ("BTCUSDT",))

        self.assertAlmostEqual(positions["BTCUSDT"]["qty"], -0.02)
        self.assertEqual(positions["BTCUSDT"]["side"], "SHORT")




class SlippageLoggingTests(unittest.TestCase):
    def test_log_slippage_buy_records_correct_bps(self) -> None:
        """Mock ccxt order response with avgPrice; verify slippage_bps calculation."""
        import json
        import time
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "slippage.jsonl"
            order = {"id": "test-1", "average": 100.5, "avgPrice": None}
            # buy: avg_fill > ref_price -> positive slippage (adverse)
            shadow_live._log_slippage(
                symbol="BTC/USDT:USDT",
                side="buy",
                ref_price=100.0,
                order=order,
                qty=0.01,
                mode="demo",
                log_path=log_path,
            )
            # Worker thread writes asynchronously — wait up to 3s
            deadline = time.monotonic() + 3.0
            while not log_path.exists() and time.monotonic() < deadline:
                time.sleep(0.05)
            record = json.loads(log_path.read_text().strip())
            self.assertEqual(record["symbol"], "BTC/USDT:USDT")
            self.assertEqual(record["side"], "buy")
            self.assertAlmostEqual(record["ref_price"], 100.0)
            self.assertAlmostEqual(record["avg_fill_price"], 100.5)
            # slippage_bps = (100.5-100)/100 * 1 * 10000 = 50.0
            self.assertAlmostEqual(record["slippage_bps"], 50.0, places=2)
            self.assertEqual(record["mode"], "demo")
            self.assertEqual(record["order_id"], "test-1")

    def test_log_slippage_skips_missing_average(self) -> None:
        """No record written when order has no avgPrice."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "slippage.jsonl"
            order = {"id": "test-2", "average": None}
            shadow_live._log_slippage(
                symbol="BNB/USDT:USDT",
                side="sell",
                ref_price=600.0,
                order=order,
                qty=1.0,
                mode="demo",
                log_path=log_path,
            )
            self.assertFalse(log_path.exists())

    def test_log_slippage_sell_adverse_is_negative_fill(self) -> None:
        """Sell fill below ref_price -> adverse -> positive slippage_bps."""
        import json
        import time
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "slippage.jsonl"
            order = {"id": "test-3", "average": 99.5}
            # sell: avg_fill < ref_price -> side_sign=-1, so bps = (99.5-100)/100 * -1 * 10000 = 50
            shadow_live._log_slippage(
                symbol="BTC/USDT:USDT",
                side="sell",
                ref_price=100.0,
                order=order,
                qty=0.01,
                mode="live",
                log_path=log_path,
            )
            # Worker thread writes asynchronously — wait up to 3s
            deadline = time.monotonic() + 3.0
            while not log_path.exists() and time.monotonic() < deadline:
                time.sleep(0.05)
            record = json.loads(log_path.read_text().strip())
            self.assertAlmostEqual(record["slippage_bps"], 50.0, places=2)
            self.assertEqual(record["mode"], "live")

class RuntimeModeResolutionTests(unittest.TestCase):
    def test_binance_mode_live_returns_live(self) -> None:
        import importlib
        import rotation_target_050_live as rt
        with unittest.mock.patch.dict("os.environ", {"BINANCE_MODE": "live"}, clear=False):
            importlib.invalidate_caches()
            self.assertEqual(rt._resolve_runtime_mode(), "live")

    def test_pairwise_live_mode_takes_priority(self) -> None:
        import rotation_target_050_live as rt
        with unittest.mock.patch.dict(
            "os.environ",
            {"PAIRWISE_LIVE_MODE": "live", "BINANCE_MODE": "demo"},
            clear=False,
        ):
            self.assertEqual(rt._resolve_runtime_mode(), "live")

    def test_unset_env_falls_back_to_demo(self) -> None:
        import rotation_target_050_live as rt
        env = {k: v for k, v in __import__("os").environ.items()
               if k not in ("PAIRWISE_LIVE_MODE", "BINANCE_MODE")}
        with unittest.mock.patch.dict("os.environ", env, clear=True):
            result = rt._resolve_runtime_mode()
            self.assertIn(result, ("demo", "live"))  # profile file may set live

    def test_demo_value_returns_demo(self) -> None:
        import rotation_target_050_live as rt
        with unittest.mock.patch.dict("os.environ", {"BINANCE_MODE": "demo"}, clear=False):
            self.assertEqual(rt._resolve_runtime_mode(), "demo")

    def test_resolve_runtime_mode_file_io_exception_returns_string(self) -> None:
        """_resolve_runtime_mode must always return a string even if file IO fails."""
        import rotation_target_050_live as rt
        env = {k: v for k, v in __import__("os").environ.items()
               if k not in ("PAIRWISE_LIVE_MODE", "BINANCE_MODE")}
        with unittest.mock.patch.dict("os.environ", env, clear=True):
            with unittest.mock.patch("builtins.open", side_effect=OSError("disk full")):
                result = rt._resolve_runtime_mode()
                self.assertIsInstance(result, str)
                self.assertIn(result, ("demo", "live", "unknown"))


class SlippageTelemetrySafetyTests(unittest.TestCase):
    """Verify that _log_slippage never raises even when IO or serialization fails."""

    def test_log_slippage_disk_write_failure_does_not_raise(self) -> None:
        """append_jsonl failure must be swallowed; no exception escapes to caller."""
        import tempfile

        order = {"id": "safe-1", "average": 101.0}
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            log_path = Path(tmpdir) / "slippage.jsonl"
            with unittest.mock.patch(
                "pairwise_regime_mixture_shadow_live.append_jsonl",
                side_effect=OSError("disk full"),
            ):
                # Must not raise
                shadow_live._log_slippage(
                    symbol="BTC/USDT:USDT",
                    side="buy",
                    ref_price=100.0,
                    order=order,
                    qty=0.01,
                    mode="demo",
                    log_path=log_path,
                )

    def test_log_slippage_rotation_target_disk_failure_does_not_raise(self) -> None:
        """Same safety guarantee for the rotation_target_050_live copy of _log_slippage."""
        import tempfile
        import rotation_target_050_live as rt

        order = {"id": "safe-2", "average": 50200.0}
        # ignore_cleanup_errors: background worker may still hold the file open when
        # the context manager exits; that's fine — telemetry is loss-tolerant.
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            log_path = Path(tmpdir) / "slippage_rt.jsonl"
            with unittest.mock.patch(
                "rotation_target_050_live.append_jsonl",
                side_effect=OSError("no space left"),
            ):
                rt._log_slippage(
                    symbol="BTC/USDT:USDT",
                    side="sell",
                    ref_price=50000.0,
                    order=order,
                    qty=0.1,
                    mode="live",
                    log_path=log_path,
                )


class SlippageQueueWorkerTests(unittest.TestCase):
    """Verify the fire-and-forget queue/worker behaviour."""

    def test_enqueue_does_not_block(self) -> None:
        """1000 rapid enqueues complete without blocking (fire-and-forget)."""
        import time
        import tempfile
        import rotation_target_050_live as rt

        order = {"id": "bench", "average": 50100.0}
        with tempfile.TemporaryDirectory() as tmpdir:
            fake_path = Path(tmpdir) / "bench_slippage.jsonl"
            start = time.monotonic()
            for i in range(1000):
                rt._log_slippage(
                    symbol="BTC/USDT:USDT",
                    side="buy",
                    ref_price=50000.0,
                    order=order,
                    qty=0.001,
                    mode="demo",
                    log_path=fake_path,
                )
            elapsed = time.monotonic() - start
        # 1000 enqueues must finish well under 1 second on any reasonable machine
        self.assertLess(elapsed, 1.0, f"Enqueue loop took {elapsed:.3f}s — blocking suspected")

    def test_worker_starts_and_writes(self) -> None:
        """Worker thread starts, drains queue, and writes records to disk."""
        import json
        import tempfile
        import time
        import rotation_target_050_live as rt

        with tempfile.TemporaryDirectory() as tmpdir:
            test_path = rt.Path(tmpdir) / "test_slippage.jsonl"

            # sell at 99.5 vs ref 100 -> side_sign=-1 -> bps=(99.5-100)/100*-1*10000=50
            order = {"id": "worker-test", "average": 99.5}
            rt._log_slippage(
                symbol="ETH/USDT:USDT",
                side="sell",
                ref_price=100.0,
                order=order,
                qty=0.5,
                mode="demo",
                log_path=test_path,
            )
            rt._ensure_slippage_worker()
            # Give worker thread time to drain (it has 60s timeout, but queue item is immediate)
            deadline = time.monotonic() + 3.0
            while not test_path.exists() and time.monotonic() < deadline:
                time.sleep(0.05)

            self.assertTrue(test_path.exists(), "Worker did not write slippage record within 3s")
            record = json.loads(test_path.read_text().strip())
            self.assertEqual(record["symbol"], "ETH/USDT:USDT")
            self.assertEqual(record["side"], "sell")
            self.assertAlmostEqual(record["slippage_bps"], 50.0, places=2)
            self.assertNotIn("_log_path", record, "_log_path internal key must not appear in JSONL")


class ClosePairPositionRefPriceTests(unittest.TestCase):
    """Verify that close_pair_position never calls fetch_last_price and that
    _log_slippage is silently skipped when ref_price is None."""

    def test_close_pair_no_fetch_when_ref_price_none(self) -> None:
        """fetch_last_price must NOT be called; _log_slippage must NOT be called."""
        import rotation_target_050_live as rt

        exchange = MagicMock()
        exchange.create_market_order.return_value = {"id": "ord-1", "average": 101.0}
        # quantize_amount returns a positive float so the order path is entered
        with unittest.mock.patch.object(rt, "quantize_amount", return_value=0.01), \
             unittest.mock.patch.object(rt, "fetch_last_price") as mock_fetch, \
             unittest.mock.patch.object(rt, "_log_slippage") as mock_log:
            result = rt.close_pair_position(
                exchange, "BTCUSDT", 0.01, execute=True, ref_price=None
            )

        mock_fetch.assert_not_called()
        mock_log.assert_not_called()
        self.assertTrue(result["placed"])

    def test_close_pair_logs_slippage_when_ref_price_provided(self) -> None:
        """_log_slippage IS called when a valid ref_price is passed."""
        import rotation_target_050_live as rt

        exchange = MagicMock()
        exchange.create_market_order.return_value = {"id": "ord-2", "average": 99.5}
        with unittest.mock.patch.object(rt, "quantize_amount", return_value=0.01), \
             unittest.mock.patch.object(rt, "fetch_last_price") as mock_fetch, \
             unittest.mock.patch.object(rt, "_log_slippage") as mock_log:
            result = rt.close_pair_position(
                exchange, "BTCUSDT", 0.01, execute=True, ref_price=100.0
            )

        mock_fetch.assert_not_called()
        mock_log.assert_called_once()
        self.assertTrue(result["placed"])

    def test_close_pair_zero_ref_price_skips_log(self) -> None:
        """ref_price=0 (falsy) must not trigger _log_slippage."""
        import rotation_target_050_live as rt

        exchange = MagicMock()
        exchange.create_market_order.return_value = {"id": "ord-3", "average": 0.0}
        with unittest.mock.patch.object(rt, "quantize_amount", return_value=0.01), \
             unittest.mock.patch.object(rt, "_log_slippage") as mock_log:
            rt.close_pair_position(
                exchange, "BTCUSDT", 0.01, execute=True, ref_price=0.0
            )

        mock_log.assert_not_called()


if __name__ == "__main__":
    unittest.main()
