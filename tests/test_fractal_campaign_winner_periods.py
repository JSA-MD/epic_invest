import sys
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import backtest_fractal_campaign_winner_periods as winner_periods
import strategy_replay_dispatch as replay_dispatch


class FractalCampaignWinnerPeriodsTests(unittest.TestCase):
    def test_select_candidate_entry_prefers_best_when_seed_missing(self) -> None:
        campaign_report = {"best_candidate": {"seed": 10}}
        selected = winner_periods.select_candidate_entry(campaign_report, seed=None)
        self.assertEqual(selected["seed"], 10)

    def test_select_candidate_entry_finds_requested_seed(self) -> None:
        campaign_report = {
            "jobs": [
                {"seed": 11, "artifacts": {"search_summary": "/tmp/a.json"}},
                {"seed": 12, "artifacts": {"search_summary": "/tmp/b.json"}},
            ]
        }
        selected = winner_periods.select_candidate_entry(campaign_report, seed=12)
        self.assertEqual(selected["artifacts"]["search_summary"], "/tmp/b.json")

    def test_infer_last_complete_day_skips_partial_tail(self) -> None:
        full_day = winner_periods.FULL_DAY_BARS_5M
        dates = (
            list(
                winner_periods.pd.date_range("2026-04-09", periods=full_day, freq="5min", tz="UTC")
            )
            + list(winner_periods.pd.date_range("2026-04-10", periods=full_day // 2, freq="5min", tz="UTC"))
        )
        resolved = winner_periods.infer_last_complete_day(winner_periods.pd.DatetimeIndex(dates))
        self.assertEqual(str(resolved.date()), "2026-04-09")

    def test_resolve_pairwise_route_state_mode_defaults_to_base(self) -> None:
        self.assertEqual(winner_periods.resolve_pairwise_route_state_mode({}, "BTCUSDT"), "base")

    def test_resolve_pairwise_route_state_mode_reads_pair_config(self) -> None:
        candidate = {
            "pair_configs": {
                "BTCUSDT": {
                    "route_state_mode": "equity_corr",
                    "mapping_indices": [0] * 12,
                    "route_breadth_threshold": 0.5,
                }
            }
        }
        self.assertEqual(winner_periods.resolve_pairwise_route_state_mode(candidate, "BTCUSDT"), "equity_corr")

    def test_evaluate_summary_periods_uses_pairwise_route_state_mode(self) -> None:
        index = winner_periods.pd.date_range("2026-04-08", periods=4, freq="5min", tz="UTC")
        df = winner_periods.pd.DataFrame(
            {
                "BTCUSDT_close": [1.0, 1.0, 1.0, 1.0],
                "BNBUSDT_close": [1.0, 1.0, 1.0, 1.0],
            },
            index=index,
        )
        summary = {
            "pairs": ["BTCUSDT", "BNBUSDT"],
            "selected_candidate": {
                "candidate_kind": "pairwise_candidate",
                "pair_configs": {
                    "BTCUSDT": {
                        "route_breadth_threshold": 0.5,
                        "mapping_indices": [0] * 12,
                        "route_state_mode": "equity_corr",
                    },
                    "BNBUSDT": {
                        "route_breadth_threshold": 0.5,
                        "mapping_indices": [0] * 12,
                        "route_state_mode": "equity_corr",
                    },
                },
            },
        }
        captured_modes: list[str] = []

        def fake_load_json(path: str | Path) -> dict:
            if str(path).endswith("summary.json"):
                return summary
            return {}

        def fake_build_fast_context(**kwargs):
            captured_modes.append(str(kwargs.get("route_state_mode")))
            return {"route_state_mode": kwargs.get("route_state_mode")}

        fake_result = {
            "avg_daily_return": 0.01,
            "total_return": 0.10,
            "max_drawdown": -0.05,
            "sharpe": 1.0,
            "n_trades": 1,
            "daily_target_hit_rate": 0.5,
            "daily_win_rate": 0.5,
            "worst_day": -0.01,
            "best_day": 0.02,
        }

        with (
            patch.object(winner_periods, "load_json", side_effect=fake_load_json),
            patch.object(
                winner_periods,
                "load_strategy_bundle",
                return_value={"library": [{"dummy": True}], "compiled_model": lambda *args: [0.0] * len(index)},
            ),
            patch.object(winner_periods.gp, "load_all_pairs", return_value=df),
            patch.object(
                winner_periods,
                "infer_last_complete_day",
                return_value=winner_periods.pd.Timestamp("2026-04-09", tz="UTC"),
            ),
            patch.object(
                winner_periods,
                "resolve_period_windows",
                return_value=[("2m", winner_periods.pd.Timestamp("2026-04-08", tz="UTC"))],
            ),
            patch.object(winner_periods.gp, "get_feature_arrays", return_value=([],)),
            patch.object(winner_periods, "build_market_features", return_value={}),
            patch.object(winner_periods, "build_overlay_inputs", return_value={}),
            patch.object(winner_periods, "load_funding_from_cache_or_empty", return_value=winner_periods.pd.DataFrame()),
            patch.object(winner_periods, "load_derivative_bundle", return_value={}),
            patch.object(winner_periods, "slice_derivative_bundle", return_value={}),
            patch.object(winner_periods, "build_library_lookup", return_value={"spans": [1]}),
            patch.object(winner_periods, "build_fast_context", side_effect=fake_build_fast_context),
            patch.object(winner_periods, "replay_candidate_from_context", return_value=fake_result),
        ):
            report = winner_periods.evaluate_summary_periods(
                summary_path=Path("summary.json"),
                pipeline_path=None,
                base_summary=Path("base_summary.json"),
                model_path=Path("model.dill"),
                candidate_role="selected",
                derivative_lookback_days=30,
                strict_summary_audit=False,
                window_source=winner_periods.WINDOW_SOURCE_CURRENT_MARKET,
            )

        self.assertEqual(report["selected_candidate"]["candidate_kind"], "pairwise_candidate")
        self.assertEqual(captured_modes, ["equity_corr", "equity_corr"])
        self.assertEqual(report["replay_audit"]["status"], "skipped")

    def test_evaluate_summary_periods_raises_when_summary_audit_mismatches(self) -> None:
        index = winner_periods.pd.date_range("2026-04-08", periods=4, freq="5min", tz="UTC")
        df = winner_periods.pd.DataFrame(
            {
                "BTCUSDT_close": [1.0, 1.0, 1.0, 1.0],
                "BNBUSDT_close": [1.0, 1.0, 1.0, 1.0],
            },
            index=index,
        )
        summary = {
            "pairs": ["BTCUSDT", "BNBUSDT"],
            "selected_candidate": {
                "candidate_kind": "pairwise_candidate",
                "pair_configs": {
                    "BTCUSDT": {
                        "route_breadth_threshold": 0.5,
                        "mapping_indices": [0] * 12,
                        "route_state_mode": "equity_corr",
                    },
                    "BNBUSDT": {
                        "route_breadth_threshold": 0.5,
                        "mapping_indices": [0] * 12,
                        "route_state_mode": "equity_corr",
                    },
                },
                "windows": {
                    "recent_2m": {
                        "start": "2026-04-08",
                        "end": "2026-04-09",
                        "per_pair": {
                            "BTCUSDT": {
                                "total_return": 0.99,
                                "daily_win_rate": 0.5,
                                "max_drawdown": -0.05,
                                "avg_daily_return": 0.01,
                                "n_trades": 1,
                            },
                            "BNBUSDT": {
                                "total_return": 0.99,
                                "daily_win_rate": 0.5,
                                "max_drawdown": -0.05,
                                "avg_daily_return": 0.01,
                                "n_trades": 1,
                            },
                        },
                        "aggregate": {
                            "mean_total_return": 0.99,
                            "worst_pair_total_return": 0.99,
                            "mean_avg_daily_return": 0.01,
                            "worst_pair_avg_daily_return": 0.01,
                            "worst_max_drawdown": -0.05,
                        },
                    }
                },
            },
        }
        fake_result = {
            "avg_daily_return": 0.01,
            "total_return": 0.10,
            "max_drawdown": -0.05,
            "sharpe": 1.0,
            "n_trades": 1,
            "daily_target_hit_rate": 0.5,
            "daily_win_rate": 0.5,
            "worst_day": -0.01,
            "best_day": 0.02,
        }

        def fake_load_json(path: str | Path) -> dict:
            if str(path).endswith("summary.json"):
                return summary
            return {}

        with (
            patch.object(winner_periods, "load_json", side_effect=fake_load_json),
            patch.object(
                winner_periods,
                "load_strategy_bundle",
                return_value={"library": [{"dummy": True}], "compiled_model": lambda *args: [0.0] * len(index)},
            ),
            patch.object(winner_periods.gp, "load_all_pairs", return_value=df),
            patch.object(
                winner_periods,
                "infer_last_complete_day",
                return_value=winner_periods.pd.Timestamp("2026-04-09", tz="UTC"),
            ),
            patch.object(
                winner_periods,
                "resolve_period_windows",
                return_value=[("2m", winner_periods.pd.Timestamp("2026-04-08", tz="UTC"))],
            ),
            patch.object(winner_periods.gp, "get_feature_arrays", return_value=([],)),
            patch.object(winner_periods, "build_market_features", return_value={}),
            patch.object(winner_periods, "build_overlay_inputs", return_value={}),
            patch.object(winner_periods, "load_funding_from_cache_or_empty", return_value=winner_periods.pd.DataFrame()),
            patch.object(winner_periods, "load_derivative_bundle", return_value={}),
            patch.object(winner_periods, "slice_derivative_bundle", return_value={}),
            patch.object(winner_periods, "build_library_lookup", return_value={"spans": [1]}),
            patch.object(winner_periods, "build_fast_context", return_value={"route_state_mode": "equity_corr"}),
            patch.object(winner_periods, "replay_candidate_from_context", return_value=fake_result),
        ):
            with self.assertRaises(RuntimeError):
                winner_periods.evaluate_summary_periods(
                    summary_path=Path("summary.json"),
                    pipeline_path=None,
                    base_summary=Path("base_summary.json"),
                    model_path=Path("model.dill"),
                    candidate_role="selected",
                    derivative_lookback_days=30,
                    strict_summary_audit=True,
                )

    def test_evaluate_summary_periods_anchors_periods_to_artifact_end_day_by_default(self) -> None:
        index = winner_periods.pd.date_range("2026-04-08", periods=4, freq="5min", tz="UTC")
        df = winner_periods.pd.DataFrame(
            {
                "BTCUSDT_close": [1.0, 1.0, 1.0, 1.0],
                "BNBUSDT_close": [1.0, 1.0, 1.0, 1.0],
            },
            index=index,
        )
        summary = {
            "pairs": ["BTCUSDT", "BNBUSDT"],
            "selected_candidate": {
                "candidate_kind": "pairwise_candidate",
                "pair_configs": {
                    "BTCUSDT": {"route_breadth_threshold": 0.5, "mapping_indices": [0] * 12, "route_state_mode": "equity_corr"},
                    "BNBUSDT": {"route_breadth_threshold": 0.5, "mapping_indices": [0] * 12, "route_state_mode": "equity_corr"},
                },
                "windows": {
                    "recent_2m": {"start": "2026-02-06", "end": "2026-04-06"},
                    "recent_6m": {"start": "2025-10-06", "end": "2026-04-06"},
                    "full_4y": {"start": "2022-04-06", "end": "2026-04-06"},
                },
            },
        }
        observed_anchors: list[str] = []
        fake_result = {
            "avg_daily_return": 0.01,
            "total_return": 0.10,
            "max_drawdown": -0.05,
            "sharpe": 1.0,
            "n_trades": 1,
            "daily_target_hit_rate": 0.5,
            "daily_win_rate": 0.5,
            "worst_day": -0.01,
            "best_day": 0.02,
        }

        def fake_load_json(path: str | Path) -> dict:
            if str(path).endswith("summary.json"):
                return summary
            return {}

        def fake_resolve_period_windows(end_day):
            observed_anchors.append(str(winner_periods.pd.Timestamp(end_day).date()))
            return []

        with (
            patch.object(winner_periods, "load_json", side_effect=fake_load_json),
            patch.object(
                winner_periods,
                "load_strategy_bundle",
                return_value={"library": [{"dummy": True}], "compiled_model": lambda *args: [0.0] * len(index)},
            ),
            patch.object(winner_periods, "build_pair_data_coverage", return_value={"BTCUSDT": {}, "BNBUSDT": {}}),
            patch.object(winner_periods.gp, "load_all_pairs", return_value=df),
            patch.object(
                winner_periods,
                "infer_last_complete_day",
                return_value=winner_periods.pd.Timestamp("2026-04-09", tz="UTC"),
            ),
            patch.object(winner_periods, "resolve_period_windows", side_effect=fake_resolve_period_windows),
            patch.object(winner_periods.gp, "get_feature_arrays", return_value=([],)),
            patch.object(winner_periods, "build_market_features", return_value={}),
            patch.object(winner_periods, "build_overlay_inputs", return_value={}),
            patch.object(winner_periods, "load_funding_from_cache_or_empty", return_value=winner_periods.pd.DataFrame()),
            patch.object(winner_periods, "load_derivative_bundle", return_value={}),
            patch.object(winner_periods, "slice_derivative_bundle", return_value={}),
            patch.object(winner_periods, "build_library_lookup", return_value={"spans": [1]}),
            patch.object(winner_periods, "build_fast_context", return_value={"route_state_mode": "equity_corr"}),
            patch.object(winner_periods, "replay_candidate_from_context", return_value=fake_result),
        ):
            winner_periods.evaluate_summary_periods(
                summary_path=Path("summary.json"),
                pipeline_path=None,
                base_summary=Path("base_summary.json"),
                model_path=Path("model.dill"),
                candidate_role="selected",
                derivative_lookback_days=30,
                strict_summary_audit=False,
                allow_truncated_data=True,
            )

        self.assertEqual(observed_anchors, ["2026-04-06"])

    def test_evaluate_summary_periods_fails_when_requested_window_is_truncated(self) -> None:
        index = winner_periods.pd.date_range("2022-04-06", periods=4, freq="5min", tz="UTC")
        df = winner_periods.pd.DataFrame(
            {
                "BTCUSDT_close": [1.0, 1.0, 1.0, 1.0],
                "BNBUSDT_close": [1.0, 1.0, 1.0, 1.0],
            },
            index=index,
        )
        summary = {
            "pairs": ["BTCUSDT", "BNBUSDT"],
            "selected_candidate": {
                "candidate_kind": "pairwise_candidate",
                "pair_configs": {
                    "BTCUSDT": {"route_breadth_threshold": 0.5, "mapping_indices": [0] * 12, "route_state_mode": "equity_corr"},
                    "BNBUSDT": {"route_breadth_threshold": 0.5, "mapping_indices": [0] * 12, "route_state_mode": "equity_corr"},
                },
                "windows": {
                    "recent_2m": {"start": "2026-02-06", "end": "2026-04-06"},
                    "recent_6m": {"start": "2025-10-06", "end": "2026-04-06"},
                    "full_4y": {"start": "2022-04-06", "end": "2026-04-06"},
                },
            },
        }

        def fake_load_json(path: str | Path) -> dict:
            if str(path).endswith("summary.json"):
                return summary
            return {}

        with (
            patch.object(winner_periods, "load_json", side_effect=fake_load_json),
            patch.object(
                winner_periods,
                "load_strategy_bundle",
                return_value={"library": [{"dummy": True}], "compiled_model": lambda *args: [0.0] * len(index)},
            ),
            patch.object(
                winner_periods,
                "build_pair_data_coverage",
                return_value={
                    "BTCUSDT": {"rows": 4, "start": "2022-04-06T00:00:00+00:00", "end": "2022-04-06T00:15:00+00:00"},
                    "BNBUSDT": {"rows": 4, "start": "2022-04-06T00:00:00+00:00", "end": "2022-04-06T00:15:00+00:00"},
                },
            ),
            patch.object(winner_periods.gp, "load_all_pairs", return_value=df),
            patch.object(
                winner_periods,
                "infer_last_complete_day",
                return_value=winner_periods.pd.Timestamp("2022-04-06", tz="UTC"),
            ),
        ):
            with self.assertRaises(RuntimeError):
                winner_periods.evaluate_summary_periods(
                    summary_path=Path("summary.json"),
                    pipeline_path=None,
                    base_summary=Path("base_summary.json"),
                    model_path=Path("model.dill"),
                    candidate_role="selected",
                    derivative_lookback_days=30,
                    strict_summary_audit=False,
                )

    def test_summary_audit_detects_pairwise_replay_drift(self) -> None:
        period_reports = [
            {
                "label": "2m",
                "start": "2026-02-10",
                "end": "2026-04-09",
                "per_pair": {
                    "BTCUSDT": {
                        "total_return": 0.10,
                        "daily_win_rate": 0.48,
                        "max_drawdown": -0.10,
                        "avg_daily_return": 0.01,
                        "n_trades": 100,
                    }
                },
                "aggregate": {
                    "mean_total_return": 0.10,
                    "worst_pair_total_return": 0.10,
                    "mean_avg_daily_return": 0.01,
                    "worst_pair_avg_daily_return": 0.01,
                    "worst_max_drawdown": -0.10,
                },
            }
        ]
        candidate = {
            "windows": {
                "recent_2m": {
                    "start": "2026-02-10",
                    "end": "2026-04-09",
                    "per_pair": {
                        "BTCUSDT": {
                            "total_return": 0.20,
                            "daily_win_rate": 0.48,
                            "max_drawdown": -0.10,
                            "avg_daily_return": 0.01,
                            "n_trades": 100,
                        }
                    },
                    "aggregate": {
                        "mean_total_return": 0.20,
                        "worst_pair_total_return": 0.20,
                        "mean_avg_daily_return": 0.01,
                        "worst_pair_avg_daily_return": 0.01,
                        "worst_max_drawdown": -0.10,
                    },
                }
            }
        }

        audit = replay_dispatch.audit_replay_against_candidate_windows(period_reports, candidate)

        self.assertEqual(audit["status"], "mismatch")
        self.assertEqual(audit["checked_windows"], 1)
        self.assertGreater(len(audit["mismatches"]), 0)

    def test_replay_dispatch_routes_pairwise_to_realistic_engine(self) -> None:
        candidate = {
            "pair_configs": {
                "BTCUSDT": {
                    "mapping_indices": [0] * 12,
                    "route_breadth_threshold": 0.5,
                    "execution_gene": {
                        "maker_priority": 0.80,
                        "max_wait_bars": 1,
                    },
                }
            }
        }
        with (
            patch.object(replay_dispatch, "realistic_overlay_replay_from_context", return_value={"ok": True}) as realistic_mock,
            patch.object(
                replay_dispatch,
                "fast_fractal_replay_from_context",
                side_effect=AssertionError("fractal engine should not be used"),
            ),
        ):
            result = replay_dispatch.replay_candidate_from_context(
                candidate=candidate,
                pair="BTCUSDT",
                context={"dummy": True},
                library_lookup={"dummy": True},
                route_thresholds=(0.5,),
                leaf_runtime_array=None,
                leaf_codes=None,
            )

        self.assertEqual(result, {"ok": True})
        realistic_mock.assert_called_once()
        self.assertEqual(realistic_mock.call_args.kwargs["execution_gene"]["maker_priority"], 0.80)

    def test_replay_dispatch_routes_fractal_to_fractal_engine(self) -> None:
        candidate = {"candidate_kind": "fractal_tree", "tree": {"type": "leaf", "expert_idx": 0}}
        with (
            patch.object(
                replay_dispatch,
                "realistic_overlay_replay_from_context",
                side_effect=AssertionError("pairwise engine should not be used"),
            ),
            patch.object(replay_dispatch, "fast_fractal_replay_from_context", return_value={"ok": True}) as fractal_mock,
        ):
            result = replay_dispatch.replay_candidate_from_context(
                candidate=candidate,
                pair="BTCUSDT",
                context={"dummy": True},
                library_lookup={"dummy": True},
                route_thresholds=(0.5,),
                leaf_runtime_array={"leaf": True},
                leaf_codes=[0, 0],
            )

        self.assertEqual(result, {"ok": True})
        fractal_mock.assert_called_once()


if __name__ == "__main__":
    unittest.main()
