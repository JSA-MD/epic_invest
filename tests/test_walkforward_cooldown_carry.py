"""Tests for inter-window cooldown carry in walkforward_pairwise.py.

Uses synthetic data and monkey-patched replay so no model files are required.
"""
from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


# ---------------------------------------------------------------------------
# Minimal stubs so walkforward_pairwise can be imported without heavy deps.
# ---------------------------------------------------------------------------

def _stub_gp_module() -> types.ModuleType:
    """Return a minimal gp_crypto_evolution stub."""
    mod = types.ModuleType("gp_crypto_evolution")
    mod.MODELS_DIR = Path("/tmp")  # type: ignore[attr-defined]
    mod.get_feature_arrays = lambda df, pair: (np.ones(len(df)),)  # type: ignore[attr-defined]

    class _Toolbox:
        def compile(self, expr: Any) -> Any:
            return lambda *args: np.zeros(len(args[0]))

    mod.toolbox = _Toolbox()  # type: ignore[attr-defined]
    return mod


def _ensure_stubs() -> None:
    """Inject minimal stubs for heavy dependencies."""
    for name in [
        "gp_crypto_evolution",
        "backtest_pairwise_equity_corr_risk_compare",
        "replay_regime_mixture_realistic",
        "search_gp_drawdown_overlay",
        "search_pair_subset_regime_mixture",
        "numba",
    ]:
        if name not in sys.modules:
            stub = types.ModuleType(name)
            sys.modules[name] = stub

    sys.modules["gp_crypto_evolution"] = _stub_gp_module()

    # backtest stub
    bt = sys.modules["backtest_pairwise_equity_corr_risk_compare"]
    bt.filter_funding_window = lambda df, s, e: df  # type: ignore[attr-defined]
    bt.load_funding_cache = lambda pair: pd.DataFrame(  # type: ignore[attr-defined]
        columns=["fundingTime", "fundingRate"]
    )

    # replay_regime_mixture_realistic stub
    rr = sys.modules["replay_regime_mixture_realistic"]
    rr.load_model = lambda path: (None, None)  # type: ignore[attr-defined]

    # search_gp_drawdown_overlay stub
    so = sys.modules["search_gp_drawdown_overlay"]
    so.iter_params = lambda: []  # type: ignore[attr-defined]

    # search_pair_subset_regime_mixture stub — exports used at import time.
    srm = sys.modules["search_pair_subset_regime_mixture"]
    srm.build_overlay_inputs = lambda df, pairs, regime_pair=None: {}  # type: ignore[attr-defined]
    srm.realistic_overlay_replay = lambda *a, **kw: {}  # type: ignore[attr-defined]


_ensure_stubs()

import walkforward_pairwise as wf  # noqa: E402  (after stubs)


# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------

def _make_df(n_bars: int = 500) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n_bars, freq="5min", tz="UTC")
    rng = np.random.default_rng(42)
    close = np.cumprod(1 + rng.normal(0, 0.001, n_bars)) * 100
    return pd.DataFrame(
        {
            "open": close * 0.999,
            "high": close * 1.001,
            "low": close * 0.998,
            "close": close,
            "volume": rng.uniform(10, 100, n_bars),
        },
        index=idx,
    )


def _make_funding_df() -> pd.DataFrame:
    return pd.DataFrame(columns=["fundingTime", "fundingRate"])


def _make_replay_result(n_trades: int = 5, final_cooldown: int = 0) -> dict[str, Any]:
    """Fake replay result mimicking what realistic_overlay_replay returns."""
    return {
        "total_return": 0.01,
        "sharpe": 0.5,
        "max_drawdown": -0.02,
        "n_wins": n_trades // 2,
        "n_losses": n_trades - n_trades // 2,
        "n_trades": n_trades,
        "roundtrip_win_rate": 0.5,
        # trace only present when return_trace=True
        "trace": {
            "cooldown_bars_left": np.array([0] * (n_trades * 10 - 1) + [final_cooldown]),
        },
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestRunPairReplay(unittest.TestCase):
    """Unit tests for _run_pair_replay with return_final_cooldown flag."""

    def _call(
        self,
        initial_cd: int = 0,
        return_final: bool = False,
        kernel_final_cd: int = 42,
    ) -> dict[str, Any] | None:
        fake_result = _make_replay_result(n_trades=10, final_cooldown=kernel_final_cd)

        with (
            patch.object(
                wf,
                "build_overlay_inputs",
                return_value={},
            ),
            patch.object(
                wf,
                "realistic_overlay_replay",
                return_value=fake_result,
            ) as mock_replay,
        ):
            df = _make_df(200)
            result = wf._run_pair_replay(
                df,
                "BTCUSDT",
                ("BTCUSDT",),
                lambda *a: np.zeros(len(a[0])),
                _make_funding_df(),
                [],
                (0,),
                0.5,
                "base",
                initial_cooldown_bars=initial_cd,
                return_final_cooldown=return_final,
            )
            # Verify the replay was called with the right args.
            call_kwargs = mock_replay.call_args.kwargs
            self.assertEqual(call_kwargs["initial_cooldown_bars"], initial_cd)
            self.assertEqual(call_kwargs["return_trace"], return_final)
        return result

    def test_carry_off_no_final_cooldown_key(self) -> None:
        result = self._call(initial_cd=0, return_final=False)
        self.assertIsNotNone(result)
        self.assertNotIn("final_cooldown_bars", result)

    def test_carry_on_final_cooldown_extracted(self) -> None:
        result = self._call(initial_cd=0, return_final=True, kernel_final_cd=77)
        self.assertIsNotNone(result)
        self.assertIn("final_cooldown_bars", result)
        self.assertEqual(result["final_cooldown_bars"], 77)

    def test_initial_cooldown_passed_through(self) -> None:
        result = self._call(initial_cd=150, return_final=True, kernel_final_cd=148)
        self.assertIsNotNone(result)
        self.assertEqual(result["final_cooldown_bars"], 148)

    def test_carry_off_zero_initial_passed(self) -> None:
        result = self._call(initial_cd=0, return_final=False)
        self.assertIsNotNone(result)
        # n_trades should still be present (basic metrics intact)
        self.assertEqual(result["n_trades"], 10)


class TestWalkForwardCooldownCarry(unittest.TestCase):
    """Integration-level tests for inter-window cooldown carry in walk_forward()."""

    PAIRS = ("BTCUSDT", "BNBUSDT")
    PAIR_CONFIGS = {
        p: {
            "mapping_indices": [0],
            "route_breadth_threshold": 0.5,
            "route_state_mode": "base",
        }
        for p in PAIRS
    }

    def _make_df_all(self, n_bars: int = 3000) -> pd.DataFrame:
        return _make_df(n_bars)

    def _make_funding_cache(self) -> dict[str, pd.DataFrame]:
        return {p: _make_funding_df() for p in self.PAIRS}

    def _fake_replay(
        self,
        df: pd.DataFrame,
        pair: str,
        *args: Any,
        initial_cooldown_bars: int = 0,
        return_trace: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Fake that returns a deterministic result with final cooldown = 100."""
        result = _make_replay_result(n_trades=5, final_cooldown=100)
        if not return_trace:
            del result["trace"]
        return result

    def _run_walk_forward(self, carry_cooldown: bool) -> list[dict[str, Any]]:
        df_all = self._make_df_all()
        funding_cache = self._make_funding_cache()

        with (
            patch.object(wf, "build_overlay_inputs", return_value={}),
            patch.object(wf, "realistic_overlay_replay", side_effect=self._fake_replay),
        ):
            folds = list(
                wf.walk_forward(
                    df_all,
                    self.PAIRS,
                    lambda *a: np.zeros(len(a[0])),
                    funding_cache,
                    [],
                    self.PAIR_CONFIGS,
                    start="2024-01-02",
                    end="2024-01-31",
                    train_days=5,
                    test_days=5,
                    step_days=5,
                    min_oos_bars=10,
                    carry_cooldown=carry_cooldown,
                )
            )
        return folds

    def test_carry_off_fold_has_zero_carry_in(self) -> None:
        folds = self._run_walk_forward(carry_cooldown=False)
        self.assertGreater(len(folds), 0)
        for fold in folds:
            for p in self.PAIRS:
                if p in fold.get("carry_cooldown_in", {}):
                    self.assertEqual(fold["carry_cooldown_in"][p], 0)
                if p in fold.get("carry_cooldown_out", {}):
                    self.assertEqual(fold["carry_cooldown_out"][p], 0)

    def test_carry_on_window_n1_carry_in_matches_window_n_carry_out(self) -> None:
        folds = self._run_walk_forward(carry_cooldown=True)
        self.assertGreaterEqual(len(folds), 2)
        for i in range(1, len(folds)):
            prev_out = folds[i - 1]["carry_cooldown_out"]
            curr_in = folds[i]["carry_cooldown_in"]
            for p in self.PAIRS:
                if p in prev_out and p in curr_in:
                    self.assertEqual(
                        curr_in[p],
                        prev_out[p],
                        msg=(
                            f"Fold {i} carry_in[{p}]={curr_in[p]} "
                            f"!= fold {i-1} carry_out[{p}]={prev_out[p]}"
                        ),
                    )

    def test_carry_on_first_fold_has_zero_carry_in(self) -> None:
        folds = self._run_walk_forward(carry_cooldown=True)
        self.assertGreater(len(folds), 0)
        first = folds[0]
        for p in self.PAIRS:
            self.assertEqual(first["carry_cooldown_in"].get(p, 0), 0)

    def test_carry_off_results_identical_to_no_carry_arg(self) -> None:
        """Regression: carry_cooldown=False must produce same n_trades as baseline."""
        folds_off = self._run_walk_forward(carry_cooldown=False)
        for fold in folds_off:
            for p in self.PAIRS:
                if p in fold["OOS"]:
                    self.assertEqual(fold["OOS"][p]["n_trades"], 5)

    def test_final_cooldown_bars_stripped_from_oos_metrics(self) -> None:
        """The final_cooldown_bars key must not leak into OOS metrics dict."""
        folds = self._run_walk_forward(carry_cooldown=True)
        for fold in folds:
            for p in self.PAIRS:
                if p in fold["OOS"]:
                    self.assertNotIn("final_cooldown_bars", fold["OOS"][p])


if __name__ == "__main__":
    unittest.main()
