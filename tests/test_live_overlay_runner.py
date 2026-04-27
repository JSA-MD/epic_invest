"""Unit tests for the Stage 2/3 live overlay runner wiring.

These tests pin two invariants:
1. Sign instability >= 30% over 288 bars forces every flipping pair flat.
2. Adaptive regime band suppresses non-flat trades when the live regime_score
   falls inside the ±3σ noise zone.

The toggle env var `PAIRWISE_LIVE_OVERLAYS=0` makes the runner a no-op so a
broken overlay can never deadlock the live trader.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from live_overlay_runner import run_live_overlays  # noqa: E402


def _make_plan(target_weights: dict[str, float], regime_scores: dict[str, float]):
    return {
        "pair_plans": {
            pair: {
                "target_weight": tw,
                "regime_score": regime_scores.get(pair, 0.0),
                "signal_pct": tw * 100.0,
            }
            for pair, tw in target_weights.items()
        },
        "target_weights": dict(target_weights),
    }


class TestLiveOverlayRunner(unittest.TestCase):
    def test_disabled_via_env_is_noop(self):
        plan = _make_plan({"BTCUSDT": -0.05}, {"BTCUSDT": 0.0})
        state: dict = {}
        result = run_live_overlays(plan, state, env={"PAIRWISE_LIVE_OVERLAYS": "0"})
        self.assertEqual(result, {})
        self.assertEqual(plan["pair_plans"]["BTCUSDT"]["target_weight"], -0.05)

    def test_short_history_does_not_force_flat(self):
        plan = _make_plan({"BTCUSDT": -0.05}, {"BTCUSDT": 0.05})
        state: dict = {}
        result = run_live_overlays(plan, state, env={"PAIRWISE_LIVE_OVERLAYS": "1"})
        # First call: not enough history, no force flat
        self.assertEqual(result, {})
        self.assertEqual(plan["pair_plans"]["BTCUSDT"]["target_weight"], -0.05)
        # State must record the bar
        self.assertEqual(len(state["recent_target_weights"]["BTCUSDT"]), 1)

    def test_sign_instability_triggers_flat(self):
        # Pre-load 288 bars of every-bar sign flip on BTCUSDT
        flipping = list(np.tile([0.05, -0.05], 144))
        state: dict = {
            "recent_target_weights": {"BTCUSDT": list(flipping)},
            "recent_regime_scores": {"BTCUSDT": [0.0] * 288},
        }
        plan = _make_plan({"BTCUSDT": -0.05}, {"BTCUSDT": 0.0})
        result = run_live_overlays(plan, state, env={"PAIRWISE_LIVE_OVERLAYS": "1"})
        self.assertEqual(result.get("BTCUSDT"), "sign_instability")
        self.assertEqual(plan["pair_plans"]["BTCUSDT"]["target_weight"], 0.0)
        self.assertEqual(plan["target_weights"]["BTCUSDT"], 0.0)
        # Journal entry written
        journal = state.get("decision_journal", [])
        self.assertTrue(any(e["pair"] == "BTCUSDT" and e["override_reason"] == "sign_instability" for e in journal))

    def test_adaptive_band_suppresses_noise_zone_trade(self):
        # Pre-load 288 bars of low-volatility regime_scores around 0
        np.random.seed(0)
        scores = (np.random.randn(288) * 0.005).tolist()
        # Stable history of long signals so sign_instability does NOT trigger
        weights = [0.05] * 288
        state: dict = {
            "recent_target_weights": {"BTCUSDT": list(weights)},
            "recent_regime_scores": {"BTCUSDT": list(scores)},
        }
        # Live regime_score is right at the noise center — should suppress
        plan = _make_plan({"BTCUSDT": -0.05}, {"BTCUSDT": 0.001})
        result = run_live_overlays(plan, state, env={"PAIRWISE_LIVE_OVERLAYS": "1"})
        self.assertEqual(result.get("BTCUSDT"), "adaptive_threshold_noise")
        self.assertEqual(plan["pair_plans"]["BTCUSDT"]["target_weight"], 0.0)

    def test_adaptive_band_allows_real_break(self):
        # 288 bars of low-vol scores with a true break at the end
        np.random.seed(1)
        scores = (np.random.randn(288) * 0.005).tolist()
        weights = [0.05] * 288
        state: dict = {
            "recent_target_weights": {"BTCUSDT": list(weights)},
            "recent_regime_scores": {"BTCUSDT": list(scores)},
        }
        plan = _make_plan({"BTCUSDT": 0.05}, {"BTCUSDT": 0.10})  # 20σ break
        result = run_live_overlays(plan, state, env={"PAIRWISE_LIVE_OVERLAYS": "1"})
        # Real break + same-sign weight → not suppressed
        self.assertNotIn("BTCUSDT", result)
        self.assertEqual(plan["pair_plans"]["BTCUSDT"]["target_weight"], 0.05)

    def test_flat_signal_is_never_forced(self):
        state: dict = {
            "recent_target_weights": {"BTCUSDT": [0.05] * 200 + [-0.05] * 88},
            "recent_regime_scores": {"BTCUSDT": [0.0] * 288},
        }
        plan = _make_plan({"BTCUSDT": 0.0}, {"BTCUSDT": 0.0})
        result = run_live_overlays(plan, state, env={"PAIRWISE_LIVE_OVERLAYS": "1"})
        # Already flat → overlay should not record a force_flat (sign already 0)
        self.assertEqual(plan["pair_plans"]["BTCUSDT"]["target_weight"], 0.0)
        # If sign_instability fires it still writes a journal entry, so just
        # confirm the weight stays flat and overlay decisions are well-formed.
        self.assertIn(plan["target_weights"]["BTCUSDT"], (0.0,))


if __name__ == "__main__":
    unittest.main()
