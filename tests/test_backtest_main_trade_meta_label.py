import sys
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import backtest_main_trade_meta_label as trade_meta


class BacktestMainTradeMetaLabelTests(unittest.TestCase):
    def test_replay_pair_prefers_btc_online_blend_dispatch_for_btc(self) -> None:
        candidate = {
            "btc_online_blend": {
                "pair": "BTCUSDT",
                "alpha_cap": 0.1,
                "eta": 0.1,
                "decay": 0.98,
                "activation_mode": "disagree_only",
            },
            "btc_convex_blend": {
                "pair": "BTCUSDT",
                "alpha": 0.4,
                "mode": "always",
                "specialist_pair_config": {"mapping_indices": [1] * 12, "route_breadth_threshold": 0.5},
            },
            "pair_configs": {
                "BTCUSDT": {"mapping_indices": [0] * 12, "route_breadth_threshold": 0.5},
            },
        }
        fake_result = {"total_return": 2.34, "daily_win_rate": 0.56}

        with patch("btc_online_blend.replay_btc_online_blend_candidate", return_value=fake_result) as replay_dispatch:
            result = trade_meta.replay_pair(
                candidate=candidate,
                pair="BTCUSDT",
                context={"bucket_codes": {0.5: []}},
                library_lookup={},
                return_trace=False,
            )

        self.assertEqual(result, fake_result)
        replay_dispatch.assert_called_once()

    def test_replay_pair_returns_trace_for_btc_online_blend(self) -> None:
        candidate = {
            "btc_online_blend": {
                "pair": "BTCUSDT",
                "alpha_cap": 0.1,
                "eta": 0.1,
                "decay": 0.98,
                "activation_mode": "disagree_only",
            },
            "btc_convex_blend": {
                "pair": "BTCUSDT",
                "alpha": 0.4,
                "mode": "always",
                "specialist_pair_config": {"mapping_indices": [1] * 12, "route_breadth_threshold": 0.5},
            },
            "pair_configs": {
                "BTCUSDT": {"mapping_indices": [0] * 12, "route_breadth_threshold": 0.5},
            },
        }
        fake_result = {"total_return": 2.34, "daily_win_rate": 0.56, "trace": {"target_weight": [0.0]}}

        with patch("btc_online_blend.replay_btc_online_blend_candidate", return_value=fake_result) as replay_dispatch:
            result = trade_meta.replay_pair(
                candidate=candidate,
                pair="BTCUSDT",
                context={"bucket_codes": {0.5: []}},
                library_lookup={},
                return_trace=True,
            )

        self.assertEqual(result, fake_result)
        replay_dispatch.assert_called_once()

    def test_replay_pair_uses_btc_convex_blend_dispatch_for_btc(self) -> None:
        candidate = {
            "btc_convex_blend": {
                "pair": "BTCUSDT",
                "alpha": 0.4,
                "mode": "always",
                "specialist_pair_config": {"mapping_indices": [1] * 12, "route_breadth_threshold": 0.5},
            },
            "pair_configs": {
                "BTCUSDT": {"mapping_indices": [0] * 12, "route_breadth_threshold": 0.5},
            },
        }
        fake_result = {"total_return": 1.23, "daily_win_rate": 0.55}

        with patch.object(trade_meta, "replay_btc_convex_blend_candidate", return_value=fake_result) as replay_blend:
            result = trade_meta.replay_pair(
                candidate=candidate,
                pair="BTCUSDT",
                context={"bucket_codes": {0.5: []}},
                library_lookup={},
                return_trace=False,
            )

        self.assertEqual(result, fake_result)
        replay_blend.assert_called_once()


if __name__ == "__main__":
    unittest.main()
