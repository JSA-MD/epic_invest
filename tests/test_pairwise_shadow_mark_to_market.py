import sys
import tempfile
import unittest
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from pairwise_regime_mixture_shadow_live import apply_shadow_mark_to_market, load_shadow_state


class PairwiseShadowMarkToMarketTests(unittest.TestCase):
    def test_apply_shadow_mark_to_market_advances_observations_and_pnl(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            state = load_shadow_state(tmp / "missing_state.json", tmp / "decisions.jsonl")

        first_plan = {
            "pairs": ["BTCUSDT", "BNBUSDT"],
            "pair_plans": [
                {"pair": "BTCUSDT", "price": 100.0, "cooldown_bars_left_after": 3},
                {"pair": "BNBUSDT", "price": 50.0, "cooldown_bars_left_after": 1},
            ],
            "target_weights": {"BTCUSDT": 0.5, "BNBUSDT": -0.5},
            "signal_timestamp": "2026-04-11T00:00:00+00:00",
        }
        first_update = apply_shadow_mark_to_market(state, first_plan)

        self.assertEqual(first_update["observations"], 1)
        self.assertAlmostEqual(state["shadow_paper"]["equity"], 99940.0)
        self.assertEqual(state["shadow_paper"]["cooldown_bars_left"]["BTCUSDT"], 3)
        self.assertEqual(state["shadow_paper"]["cooldown_bars_left"]["BNBUSDT"], 1)

        second_plan = {
            "pairs": ["BTCUSDT", "BNBUSDT"],
            "pair_plans": [
                {"pair": "BTCUSDT", "price": 110.0, "cooldown_bars_left_after": 0},
                {"pair": "BNBUSDT", "price": 45.0, "cooldown_bars_left_after": 0},
            ],
            "target_weights": {"BTCUSDT": 0.5, "BNBUSDT": -0.5},
            "signal_timestamp": "2026-04-11T00:05:00+00:00",
        }
        second_update = apply_shadow_mark_to_market(state, second_plan)

        self.assertEqual(second_update["observations"], 2)
        self.assertAlmostEqual(state["shadow_paper"]["equity"], 109934.0)
        self.assertAlmostEqual(state["shadow_paper"]["return_pct"], 9.934)
        self.assertEqual(state["shadow_paper"]["cooldown_bars_left"]["BTCUSDT"], 0)
        self.assertEqual(state["shadow_paper"]["cooldown_bars_left"]["BNBUSDT"], 0)


if __name__ == "__main__":
    unittest.main()
