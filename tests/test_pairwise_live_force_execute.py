import sys
import unittest
from argparse import Namespace
from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import pairwise_regime_live as pairwise_live


REPAIR_SUMMARY_NAME = "gp_regime_mixture_btc_bnb_pairwise_repair_summary.json"
VALIDATED_SUMMARY_NAME = "gp_regime_mixture_btc_bnb_pairwise_repair_equity_corr_validated_summary.json"


def make_args(*, execute: bool, force_execute: bool) -> Namespace:
    return Namespace(
        command="run-once",
        summary_path=Path("models/mock_summary.json"),
        model_path=Path("models/mock_model.dill"),
        promotion_report=Path("models/mock_promotion_report.json"),
        state_path=Path("models/mock_live_state.json"),
        decision_log_path=Path("logs/mock_pairwise_live.jsonl"),
        equity=100000.0,
        refresh_live_data=False,
        execute=execute,
        force_execute=force_execute,
        force_note="manual_primary_switch",
        mode="demo",
    )


class PairwiseLiveForceExecuteTests(unittest.TestCase):
    def test_load_promotion_gate_blocks_manual_live_override_when_base_gate_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            report_path = Path(tmpdir) / "promotion_report.json"
            report_path.write_text(
                json.dumps(
                    {
                        "decision": {
                            "status": "shadow_ready_only",
                            "ready_for_shadow_live": True,
                            "ready_for_live": False,
                            "ready_for_merge": False,
                            "stress_gate_passed": False,
                        },
                        "manual_override": {
                            "enabled": True,
                            "status": "ready_for_live",
                            "ready_for_shadow_live": True,
                            "ready_for_live": True,
                            "ready_for_merge": True,
                            "disable_shadow_runtime": True,
                        },
                    }
                )
            )

            gate = pairwise_live.load_promotion_gate(report_path)

        self.assertEqual(gate["status"], "demo_ready_only")
        self.assertTrue(gate["ready_for_demo"])
        self.assertTrue(gate["ready_for_shadow_live"])
        self.assertFalse(gate["ready_for_live"])
        self.assertFalse(gate["ready_for_merge"])
        self.assertFalse(gate["shadow_required"])
        self.assertTrue(gate["manual_override_active"])
        self.assertTrue(gate["manual_live_override_blocked"])

    def test_load_state_recovers_from_corrupted_primary_using_backup(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = Path(tmpdir) / "live_state.json"
            backup_path = Path(tmpdir) / "live_state.json.bak"
            state_path.write_text('{"broken": ')
            backup_path.write_text(json.dumps({"notification_state": {"position_loss_alerted": {"BNBUSDT": True}}}))

            state = pairwise_live.load_state(state_path)

        self.assertTrue(state["notification_state"]["position_loss_alerted"]["BNBUSDT"])

    def test_promotion_gate_uses_demo_and_live_readiness_separately(self) -> None:
        gate = {
            "ready_for_shadow_live": True,
            "ready_for_demo": True,
            "ready_for_live": False,
            "ready_for_merge": False,
        }
        self.assertTrue(pairwise_live.promotion_gate_allows_execution(gate, "demo"))
        self.assertFalse(pairwise_live.promotion_gate_allows_execution(gate, "live"))
        gate["ready_for_live"] = True
        gate["ready_for_merge"] = True
        self.assertTrue(pairwise_live.promotion_gate_allows_execution(gate, "demo"))
        self.assertTrue(pairwise_live.promotion_gate_allows_execution(gate, "live"))

    def test_sync_position_loss_notifications_updates_snapshot_and_dispatches(self) -> None:
        state = {"latest_runtime_snapshot": {}}
        rotation = MagicMock()
        rotation.collect_position_loss_notifications.return_value = ["포지션 손실 경고\n- 수익률: -2.10%"]

        with patch.object(pairwise_live, "load_notification_bridge", return_value=rotation):
            pairwise_live.sync_position_loss_notifications(
                state,
                {
                    "BNBUSDT": {
                        "pair": "BNBUSDT",
                        "side": "SHORT",
                        "entry_price": 600.0,
                        "mark_price": 612.6,
                    }
                },
            )

        self.assertEqual(state["latest_runtime_snapshot"]["positions"][0]["pair"], "BNBUSDT")
        self.assertIn("position_loss_alerted", state["notification_state"])
        rotation.collect_position_loss_notifications.assert_called_once_with(state)
        rotation.dispatch_notifications.assert_called_once_with(
            state,
            ["포지션 손실 경고\n- 수익률: -2.10%"],
        )

    def test_operational_defaults_use_validated_pairwise_summary(self) -> None:
        self.assertEqual(pairwise_live.DEFAULT_SUMMARY_PATH, ROOT_DIR / "models" / VALIDATED_SUMMARY_NAME)

        shadow_live_source = (SCRIPTS_DIR / "pairwise_regime_mixture_shadow_live.py").read_text()
        self.assertIn(VALIDATED_SUMMARY_NAME, shadow_live_source)
        self.assertNotIn(f'DEFAULT_SUMMARY_PATH = gp.MODELS_DIR / "{REPAIR_SUMMARY_NAME}"', shadow_live_source)

    def test_compute_requested_weight_de_risks_when_equity_corr_is_inverse(self) -> None:
        params = MagicMock(
            signal_span=2,
            regime_threshold=0.01,
            breadth_threshold=0.5,
            target_vol_ann=10.0,
            gross_cap=1.5,
        )
        baseline = pairwise_live.compute_requested_weight(
            raw_signal=pairwise_live.np.asarray([150.0, 150.0]),
            params=params,
            regime_score=0.05,
            breadth_score=0.8,
            bar_vol_ann=0.2,
            equity_corr_gross_scale=1.0,
            equity_corr_regime_mult=1.0,
        )
        reduced = pairwise_live.compute_requested_weight(
            raw_signal=pairwise_live.np.asarray([150.0, 150.0]),
            params=params,
            regime_score=0.05,
            breadth_score=0.8,
            bar_vol_ann=0.2,
            equity_corr_gross_scale=0.8,
            equity_corr_regime_mult=1.1,
        )
        self.assertAlmostEqual(baseline, 1.5)
        self.assertAlmostEqual(reduced, 1.2)

    def test_resolve_strategy_artifact_path_prefers_embedded_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            nested = tmp / "artifacts"
            nested.mkdir()
            summary_path = nested / "summary.json"
            embedded_model = nested / "embedded_model.dill"
            fallback_model = nested / "fallback_model.dill"
            embedded_model.write_text("embedded")
            fallback_model.write_text("fallback")
            summary_path.write_text(json.dumps({"model_path": embedded_model.name, "selected_candidate": {"pair_configs": {}}}))

            payload = pairwise_live.load_selected_candidate(summary_path)
            embedded_ref = pairwise_live.extract_strategy_artifact_reference(payload, "model_path")
            resolved = pairwise_live.resolve_strategy_artifact_path(embedded_ref or fallback_model, summary_path)

        self.assertEqual(resolved.name, "embedded_model.dill")

    def test_resolve_runtime_summary_path_prefers_market_os_candidate_when_report_is_runtime_ready(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            requested = tmp / VALIDATED_SUMMARY_NAME
            promoted = tmp / "gp_regime_mixture_btc_bnb_pairwise_market_os_candidate_summary.json"
            report_path = tmp / "promotion_report.json"
            requested.write_text(json.dumps({"selected_candidate": {"pair_configs": {}}}))
            promoted.write_text(json.dumps({"selected_candidate": {"pair_configs": {}}, "model_path": "model.dill"}))
            report_path.write_text(
                json.dumps(
                    {
                        "selected_candidate": {"candidate_id": "candidate-1"},
                        "ready_for_demo": True,
                        "ready_for_live": False,
                        "ready_for_merge": False,
                    }
                )
            )

            with (
                patch.object(pairwise_live, "DEFAULT_SUMMARY_PATH", requested),
                patch.object(pairwise_live, "DEFAULT_MARKET_OS_SUMMARY_PATH", promoted),
            ):
                resolved = pairwise_live.resolve_runtime_summary_path(requested, report_path)

        self.assertEqual(resolved, promoted)

    def test_resolve_runtime_summary_path_keeps_validated_summary_when_selected_candidate_is_blocked(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            requested = tmp / VALIDATED_SUMMARY_NAME
            promoted = tmp / "gp_regime_mixture_btc_bnb_pairwise_market_os_candidate_summary.json"
            report_path = tmp / "promotion_report.json"
            requested.write_text(json.dumps({"selected_candidate": {"pair_configs": {}}}))
            promoted.write_text(json.dumps({"selected_candidate": {"pair_configs": {}}, "model_path": "model.dill"}))
            report_path.write_text(
                json.dumps(
                    {
                        "selected_candidate": {"candidate_id": "candidate-1"},
                        "ready_for_demo": False,
                        "ready_for_live": False,
                        "ready_for_merge": False,
                    }
                )
            )

            with (
                patch.object(pairwise_live, "DEFAULT_SUMMARY_PATH", requested),
                patch.object(pairwise_live, "DEFAULT_MARKET_OS_SUMMARY_PATH", promoted),
            ):
                resolved = pairwise_live.resolve_runtime_summary_path(requested, report_path)

        self.assertEqual(resolved, requested)

    def test_resolve_runtime_summary_path_keeps_explicit_nondefault_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            explicit = tmp / "explicit_summary.json"
            report_path = tmp / "promotion_report.json"
            explicit.write_text(json.dumps({"selected_candidate": {"pair_configs": {}}, "model_path": "model.dill"}))
            report_path.write_text(json.dumps({"selected_candidate": {"candidate_id": "candidate-1"}}))

            resolved = pairwise_live.resolve_runtime_summary_path(explicit, report_path)

        self.assertEqual(resolved, explicit)

    def test_resolve_runtime_promotion_report_path_prefers_validated_stress_report_for_validated_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            validated_summary = tmp / VALIDATED_SUMMARY_NAME
            market_os_summary = tmp / "gp_regime_mixture_btc_bnb_pairwise_market_os_candidate_summary.json"
            market_os_report = tmp / "gp_regime_mixture_btc_bnb_pairwise_market_os_pipeline_report.json"
            validated_report = tmp / "gp_regime_mixture_btc_bnb_pairwise_validated_stress_report.json"
            validated_summary.write_text(json.dumps({"selected_candidate": {"pair_configs": {}}}))
            market_os_summary.write_text(json.dumps({"selected_candidate": {"pair_configs": {}}}))
            market_os_report.write_text(
                json.dumps(
                    {
                        "artifacts": {"search_summary": market_os_summary.name},
                        "decision": {
                            "status": "validation_gate_blocked",
                            "ready_for_demo": False,
                            "ready_for_live": False,
                            "ready_for_merge": False,
                        },
                    }
                )
            )
            validated_report.write_text(
                json.dumps(
                    {
                        "summary_path": validated_summary.name,
                        "promotion_decision": {
                            "status": "ready_for_live",
                            "ready_for_live": True,
                            "ready_for_merge": False,
                        },
                    }
                )
            )

            with (
                patch.object(pairwise_live, "DEFAULT_SUMMARY_PATH", validated_summary),
                patch.object(pairwise_live, "DEFAULT_VALIDATED_STRESS_REPORT_PATH", validated_report),
            ):
                resolved = pairwise_live.resolve_runtime_promotion_report_path(validated_summary, market_os_report)

        self.assertEqual(resolved, validated_report)

    def test_run_live_once_uses_resolved_promotion_report_from_plan(self) -> None:
        args = make_args(execute=False, force_execute=False)
        resolved_report = Path("/tmp/resolved_promotion_report.json")

        with (
            patch.object(pairwise_live, "load_state", return_value={}),
            patch.object(
                pairwise_live,
                "build_pairwise_plan",
                return_value={
                    "target_weights": {"BTCUSDT": 0.0, "BNBUSDT": 0.0},
                    "promotion_report_path": str(resolved_report),
                },
            ),
            patch.object(pairwise_live, "persist_runtime_plan_state"),
            patch.object(pairwise_live, "load_promotion_gate", return_value={"ready_for_demo": True}) as load_gate,
            patch.object(pairwise_live, "record_runtime_success"),
            patch.object(pairwise_live, "append_jsonl"),
            patch.object(pairwise_live, "save_state"),
        ):
            result = pairwise_live.run_live_once(args)

        self.assertEqual(result, 0)
        load_gate.assert_called_once_with(resolved_report)

    def test_trace_driven_pair_plan_uses_last_valid_trace_index(self) -> None:
        df = pd.DataFrame(
            {
                "BTCUSDT_close": [100.0, 101.0, 102.0, 103.0],
            },
            index=pd.date_range("2026-04-18 14:30:00+00:00", periods=4, freq="5min"),
        )
        overlay_series = pd.Series(["equity_mixed"] * len(df), index=df.index.normalize())
        missing_series = pd.Series(["low_corr"] * len(df), index=df.index.normalize())
        pair_config = {
            "route_breadth_threshold": 0.5,
            "mapping_indices": [0] * len(pairwise_live.route_state_names("equity_corr")),
            "route_state_mode": "equity_corr",
        }
        trace = {
            "target_weight": pairwise_live.np.asarray([0.1, 0.25], dtype="float64"),
            "requested_weight": pairwise_live.np.asarray([0.2, 0.5], dtype="float64"),
            "signal_pct": pairwise_live.np.asarray([10.0, 50.0], dtype="float64"),
            "role_idx": pairwise_live.np.asarray([0, 1], dtype="int64"),
            "cooldown_bars_left": pairwise_live.np.asarray([3, 1], dtype="int64"),
        }
        fast_context = {
            "bucket_codes": {0.5: pairwise_live.np.asarray([0, 1, 2, 3], dtype="int64")},
            "equity_corr": pairwise_live.np.asarray([0.1, 0.2, 0.3, 0.4], dtype="float64"),
            "regime": pairwise_live.np.asarray([0.01, 0.02, 0.03, 0.04], dtype="float64"),
            "breadth": pairwise_live.np.asarray([0.4, 0.5, 0.6, 0.7], dtype="float64"),
            "vol_ann": pairwise_live.np.asarray([0.2, 0.3, 0.4, 0.5], dtype="float64"),
            "equity_corr_gross_scale": pairwise_live.np.asarray([1.0, 1.0, 1.0, 1.0], dtype="float64"),
            "equity_corr_regime_mult": pairwise_live.np.asarray([1.0, 1.0, 1.0, 1.0], dtype="float64"),
        }
        library = list(pairwise_live.iter_params())

        with (
            patch.object(pairwise_live, "build_fast_context", return_value=fast_context),
            patch.object(
                pairwise_live,
                "realistic_overlay_replay_from_context",
                return_value={"trace": trace},
            ),
        ):
            plan = pairwise_live._build_trace_driven_pair_plan(
                df=df,
                pair="BTCUSDT",
                pair_config=pair_config,
                raw_signal=pairwise_live.np.asarray([1.0, 2.0, 3.0, 4.0], dtype="float64"),
                overlay_inputs={
                    "equity_corr_bucket_daily": overlay_series,
                    "equity_corr_quantile_state_daily": missing_series,
                    "equity_corr_context": "QQQ",
                    "equity_corr_source_mode": "market_context",
                },
                library=library,
                library_lookup={},
                current_weight=0.0,
                derivative_bundle=None,
            )

        self.assertEqual(plan["route_bucket"], 1)
        self.assertEqual(plan["route_mapping_index"], 0)
        self.assertAlmostEqual(plan["policy_current_weight"], 0.1)
        self.assertAlmostEqual(plan["requested_weight"], 0.5)
        self.assertAlmostEqual(plan["target_weight"], 0.25)
        self.assertAlmostEqual(plan["signal_pct"], 50.0)
        self.assertEqual(plan["cooldown_bars_left_after"], 1)
        self.assertEqual(plan["role_idx"], 1)
        self.assertAlmostEqual(plan["price"], 101.0)

    def test_build_pairwise_plan_applies_state_alpha_for_pair_specific_convex_blend(self) -> None:
        df = pd.DataFrame(
            {
                "BNBUSDT_open": [619.0, 620.0, 622.0],
                "BNBUSDT_high": [621.0, 622.0, 624.0],
                "BNBUSDT_low": [618.0, 619.0, 621.0],
                "BNBUSDT_close": [620.0, 621.0, 623.0],
                "BNBUSDT_volume": [10.0, 11.0, 12.0],
            },
            index=pd.date_range("2026-04-19 05:20:00+00:00", periods=3, freq="5min"),
        )
        summary = {
            "selected_candidate": {
                "pair_configs": {
                    "BNBUSDT": {
                        "mapping_indices": [1] * 12,
                        "route_breadth_threshold": 0.5,
                        "route_state_mode": "equity_corr",
                    }
                },
                "pair_convex_blends": {
                    "BNBUSDT": {
                        "alpha": 0.2,
                        "mode": "state_alphas",
                        "state_alphas": {"equity_mixed:bull_broad": 0.2},
                        "specialist_pair_config": {
                            "mapping_indices": [2] * 12,
                            "route_breadth_threshold": 0.5,
                            "route_state_mode": "equity_corr",
                        },
                    }
                },
            }
        }
        baseline_plan = {
            "requested_weight": 0.0,
            "target_weight": 0.0,
            "route_state_name": "equity_mixed:bull_broad",
            "price": 623.0,
        }
        specialist_plan = {
            "requested_weight": -0.105,
            "target_weight": -0.105,
            "route_state_name": "equity_mixed:bull_broad",
            "price": 623.0,
        }
        plan_iter = iter([baseline_plan, specialist_plan])

        with (
            patch.object(pairwise_live, "PAIRS", ("BNBUSDT",)),
            patch.object(pairwise_live, "resolve_runtime_summary_path", return_value=Path("summary.json")),
            patch.object(pairwise_live, "resolve_runtime_promotion_report_path", return_value=Path("promotion.json")),
            patch.object(pairwise_live, "load_selected_candidate", return_value=summary),
            patch.object(pairwise_live, "extract_strategy_artifact_reference", return_value=None),
            patch.object(pairwise_live, "resolve_strategy_artifact_path", return_value=Path("model.dill")),
            patch.object(pairwise_live, "iter_params", return_value=()),
            patch.object(pairwise_live, "build_library_lookup", return_value={}),
            patch.object(pairwise_live, "load_signal_model", return_value=(object(), None)),
            patch.object(pairwise_live.gp.toolbox, "compile", return_value=lambda *args: np.zeros(len(df), dtype=float)),
            patch.object(pairwise_live.gp, "get_feature_arrays", return_value=(np.zeros(len(df), dtype=float),)),
            patch.object(pairwise_live, "load_live_frame", return_value=df),
            patch.object(pairwise_live, "_append_synthetic_planning_bar", return_value=df),
            patch.object(pairwise_live, "build_overlay_inputs", return_value={}),
            patch.object(pairwise_live, "_load_derivative_bundle", return_value=None),
            patch.object(pairwise_live, "_build_trace_driven_pair_plan", side_effect=lambda **_: dict(next(plan_iter))),
            patch.object(pairwise_live, "get_btc_online_blend", return_value=None),
            patch.object(pairwise_live, "get_btc_event_blend", return_value=None),
        ):
            plan = pairwise_live.build_pairwise_plan(
                Path("summary.json"),
                Path("model.dill"),
                Path("promotion.json"),
                False,
                {"shadow_paper": {"current_weights": {"BNBUSDT": 0.0}}},
            )

        pair_plan = plan["pair_plans"]["BNBUSDT"]
        self.assertAlmostEqual(float(pair_plan["requested_weight"]), -0.021)
        self.assertAlmostEqual(float(pair_plan["target_weight"]), -0.021)
        self.assertEqual(pair_plan["blend"]["state_alphas"], {"equity_mixed:bull_broad": 0.2})

    def test_live_execute_blocks_when_gate_fails_without_force(self) -> None:
        args = make_args(execute=True, force_execute=False)
        bridge = MagicMock()
        bridge.get_exchange.return_value = object()
        bridge.fetch_equity.return_value = 2000.0
        bridge.fetch_open_position_map.return_value = {}

        with (
            patch.object(pairwise_live, "load_state", return_value={}),
            patch.object(pairwise_live, "build_pairwise_plan", return_value={"target_weights": {"BTCUSDT": 0.0, "BNBUSDT": -1.5}}),
            patch.object(pairwise_live, "load_promotion_gate", return_value={"ready_for_shadow_live": False, "ready_for_live": False, "ready_for_merge": False}),
            patch.object(pairwise_live, "record_runtime_success"),
            patch.object(pairwise_live, "sync_position_loss_notifications"),
            patch.object(pairwise_live, "append_jsonl"),
            patch.object(pairwise_live, "save_state"),
            patch.object(pairwise_live, "load_execution_bridge", return_value=bridge),
        ):
            result = pairwise_live.run_live_once(args)

        self.assertEqual(result, 2)
        bridge.get_exchange.assert_called_once_with("demo")
        bridge.fetch_equity.assert_called_once()
        bridge.fetch_open_position_map.assert_called_once()

    def test_live_execute_bypasses_failed_gate_when_forced(self) -> None:
        args = make_args(execute=True, force_execute=True)
        bridge = MagicMock()
        bridge.get_exchange.return_value = object()
        bridge.fetch_equity.return_value = 12345.0
        bridge.reconcile_target_positions.return_value = [{"pair": "BNBUSDT", "action": "SELL"}]
        bridge.install_shutdown_protection.return_value = {"installed": True}
        bridge.fetch_open_position_map.return_value = {}

        with (
            patch.object(pairwise_live, "load_state", return_value={}),
            patch.object(pairwise_live, "build_pairwise_plan", return_value={"target_weights": {"BTCUSDT": 0.0, "BNBUSDT": -1.5}}),
            patch.object(pairwise_live, "load_promotion_gate", return_value={"ready_for_shadow_live": False, "ready_for_live": False, "ready_for_merge": False}),
            patch.object(pairwise_live, "record_runtime_success"),
            patch.object(pairwise_live, "sync_position_loss_notifications"),
            patch.object(pairwise_live, "append_jsonl"),
            patch.object(pairwise_live, "save_state"),
            patch.object(pairwise_live, "load_execution_bridge", return_value=bridge),
        ):
            result = pairwise_live.run_live_once(args)

        self.assertEqual(result, 0)
        bridge.get_exchange.assert_called_once_with("demo")
        bridge.fetch_equity.assert_called_once()
        bridge.reconcile_target_positions.assert_called_once()
        bridge.install_shutdown_protection.assert_called_once()
        self.assertEqual(bridge.fetch_open_position_map.call_count, 2)

    def test_live_execute_force_bypasses_demo_gate(self) -> None:
        args = make_args(execute=True, force_execute=True)
        bridge = MagicMock()
        bridge.get_exchange.return_value = object()
        bridge.fetch_open_position_map.return_value = {}
        bridge.fetch_equity.return_value = 12345.0
        bridge.reconcile_target_positions.return_value = []
        bridge.install_shutdown_protection.return_value = {"installed": True}

        with (
            patch.object(pairwise_live, "load_state", return_value={}),
            patch.object(pairwise_live, "build_pairwise_plan", return_value={"target_weights": {"BTCUSDT": 0.0, "BNBUSDT": -1.5}}),
            patch.object(pairwise_live, "load_promotion_gate", return_value={"ready_for_shadow_live": True, "ready_for_live": False, "ready_for_merge": False}),
            patch.object(pairwise_live, "record_runtime_success"),
            patch.object(pairwise_live, "sync_position_loss_notifications"),
            patch.object(pairwise_live, "append_jsonl"),
            patch.object(pairwise_live, "save_state"),
            patch.object(pairwise_live, "load_execution_bridge", return_value=bridge),
        ):
            result = pairwise_live.run_live_once(args)

        self.assertEqual(result, 0)
        bridge.get_exchange.assert_called_once_with("demo")
        bridge.reconcile_target_positions.assert_called_once()
        self.assertEqual(bridge.fetch_open_position_map.call_count, 2)

    def test_load_live_frame_requests_recent_klines_with_datetimes(self) -> None:
        base_index = pd.date_range("2026-04-10 10:00", periods=25, freq="5min", tz="UTC")
        base = pd.DataFrame(
            {
                "BTCUSDT_close": [float(i) for i in range(25)],
                "BNBUSDT_close": [float(i + 100) for i in range(25)],
            },
            index=base_index,
        )
        recent_index = pd.date_range("2026-04-10 12:00", periods=2, freq="5min", tz="UTC")
        recent_btc = pd.DataFrame({"close": [2.1, 2.2]}, index=recent_index)
        recent_bnb = pd.DataFrame({"close": [4.1, 4.2]}, index=recent_index)

        with (
            patch.object(pairwise_live.gp, "load_all_pairs", return_value=base),
            patch.object(
                pairwise_live.gp,
                "fetch_klines",
                side_effect=[recent_btc, recent_bnb],
            ) as fetch_klines,
            patch.object(
                pairwise_live,
                "utc_now",
                return_value=datetime(2026, 4, 10, 12, 7, tzinfo=timezone.utc),
            ),
        ):
            df = pairwise_live.load_live_frame(("BTCUSDT", "BNBUSDT"), refresh_live_data=True, recent_days=1)

        first_call = fetch_klines.call_args_list[0]
        self.assertIsInstance(first_call.args[2], datetime)
        self.assertIsInstance(first_call.args[3], datetime)
        self.assertEqual(df.index.max().isoformat(), "2026-04-10T12:05:00+00:00")

    def test_load_live_frame_raises_when_latest_common_bar_is_stale(self) -> None:
        base_index = pd.date_range("2026-04-09 10:00", periods=25, freq="5min", tz="UTC")
        base = pd.DataFrame(
            {
                "BTCUSDT_close": [float(i) for i in range(25)],
                "BNBUSDT_close": [float(i + 100) for i in range(25)],
            },
            index=base_index,
        )
        recent_index = pd.date_range("2026-04-10 12:00", periods=2, freq="5min", tz="UTC")
        recent_btc = pd.DataFrame({"close": [2.1, 2.2]}, index=recent_index)
        recent_bnb = pd.DataFrame({"close": []}, index=pd.DatetimeIndex([], tz="UTC"))

        with (
            patch.object(pairwise_live.gp, "load_all_pairs", return_value=base),
            patch.object(
                pairwise_live.gp,
                "fetch_klines",
                side_effect=[recent_btc, recent_bnb],
            ),
            patch.object(
                pairwise_live,
                "utc_now",
                return_value=datetime(2026, 4, 10, 12, 7, tzinfo=timezone.utc),
            ),
        ):
            with self.assertRaises(RuntimeError):
                pairwise_live.load_live_frame(("BTCUSDT", "BNBUSDT"), refresh_live_data=True, recent_days=1)

    def test_run_sync_state_execute_reconciles_protection_orders(self) -> None:
        args = Namespace(
            command="sync-state",
            summary_path=Path("models/mock_summary.json"),
            model_path=Path("models/mock_model.dill"),
            promotion_report=Path("models/mock_promotion_report.json"),
            state_path=Path("models/mock_live_state.json"),
            decision_log_path=Path("logs/mock_pairwise_live.jsonl"),
            equity=100000.0,
            refresh_live_data=False,
            execute=True,
            force_execute=False,
            force_note="manual_primary_switch",
            mode="demo",
        )
        state = {"runtime_health": {}}
        bridge = MagicMock()
        bridge.get_exchange.return_value = object()
        bridge.install_shutdown_protection.return_value = {"status": "placed", "cancelled_count": 3}
        bridge.fetch_equity.return_value = 1234.5
        bridge.fetch_open_position_map.return_value = {
            "BNBUSDT": {"side": "SHORT", "qty": -2.0, "mark_price": 300.0}
        }
        bridge.fetch_strategy_protection_orders.return_value = [{"id": "1"}]

        saved: dict[str, object] = {}

        def _save(_path, payload):
            saved["state"] = payload

        with (
            patch.object(pairwise_live, "load_state", return_value=state),
            patch.object(pairwise_live, "load_execution_bridge", return_value=bridge),
            patch.object(pairwise_live, "save_state", side_effect=_save),
        ):
            rc = pairwise_live.run_sync_state(args)

        self.assertEqual(rc, 0)
        bridge.install_shutdown_protection.assert_called_once()
        self.assertEqual(saved["state"]["latest_live_sync"]["protection_cleanup"]["cancelled_count"], 3)
        self.assertAlmostEqual(saved["state"]["shadow_paper"]["current_weights"]["BNBUSDT"], -600.0 / 1234.5)

    def test_run_live_once_persists_plan_cooldowns(self) -> None:
        args = make_args(execute=True, force_execute=False)
        state = {
            "shadow_paper": {
                "source_mode": "live",
                "baseline_equity": 1200.0,
                "equity": 1234.5,
                "peak_equity": 1300.0,
                "max_drawdown": 0.05,
                "cooldown_bars_left": {"BTCUSDT": 7, "BNBUSDT": 3},
                "current_weights": {"BTCUSDT": 0.0, "BNBUSDT": 0.0},
                "last_prices": {"BTCUSDT": 70000.0, "BNBUSDT": 600.0},
            }
        }
        bridge = MagicMock()
        bridge.get_exchange.return_value = object()
        bridge.fetch_equity.return_value = 1234.5
        bridge.fetch_open_position_map.return_value = {}
        bridge.reconcile_target_positions.return_value = []
        bridge.install_shutdown_protection.return_value = {"installed": True}

        plan = {
            "signal_timestamp": "2026-04-15T00:35:00+00:00",
            "latest_prices": {"BTCUSDT": 74482.8, "BNBUSDT": 616.33},
            "target_weights": {"BTCUSDT": 0.0, "BNBUSDT": 0.0},
            "pair_plans": {
                "BTCUSDT": {"cooldown_bars_left_after": 6},
                "BNBUSDT": {"cooldown_bars_left_after": 2},
            },
        }

        saved: dict[str, object] = {}

        def _save(_path, payload):
            saved["state"] = json.loads(json.dumps(payload))

        with (
            patch.object(pairwise_live, "load_state", return_value=state),
            patch.object(pairwise_live, "build_pairwise_plan", return_value=plan),
            patch.object(
                pairwise_live,
                "load_promotion_gate",
                return_value={"ready_for_live": True, "ready_for_merge": True, "ready_for_shadow_live": True},
            ),
            patch.object(pairwise_live, "record_runtime_success"),
            patch.object(pairwise_live, "sync_position_loss_notifications"),
            patch.object(pairwise_live, "append_jsonl"),
            patch.object(pairwise_live, "save_state", side_effect=_save),
            patch.object(pairwise_live, "load_execution_bridge", return_value=bridge),
        ):
            rc = pairwise_live.run_live_once(args)

        self.assertEqual(rc, 0)
        shadow = saved["state"]["shadow_paper"]
        self.assertEqual(shadow["cooldown_bars_left"], {"BTCUSDT": 6, "BNBUSDT": 2})
        self.assertEqual(shadow["last_signal_timestamp"], "2026-04-15T00:35:00+00:00")

    def test_sync_shadow_paper_from_live_positions_uses_signed_position_weights(self) -> None:
        state: dict[str, object] = {}

        pairwise_live.sync_shadow_paper_from_live_positions(
            state,
            {
                "BTCUSDT": {"qty": -0.05, "mark_price": 70_000.0},
                "BNBUSDT": {"qty": 3.0, "mark_price": 600.0},
            },
            equity=10_000.0,
        )

        shadow = state["shadow_paper"]
        self.assertAlmostEqual(shadow["current_weights"]["BTCUSDT"], -0.35)
        self.assertAlmostEqual(shadow["current_weights"]["BNBUSDT"], 0.18)
        self.assertEqual(shadow["last_prices"]["BTCUSDT"], 70_000.0)

    def test_sync_shadow_paper_from_live_positions_resets_legacy_shadow_baseline(self) -> None:
        state: dict[str, object] = {
            "shadow_paper": {
                "enabled": True,
                "observations": 1,
                "equity": 100000.0,
                "peak_equity": 100000.0,
                "max_drawdown": 0.95,
                "return_pct": -95.0,
                "last_prices": {"BTCUSDT": 71640.9},
                "current_weights": {},
                "cooldown_bars_left": {"BTCUSDT": 0, "BNBUSDT": 0},
                "turnover_cost_paid": 90.0,
            }
        }

        pairwise_live.sync_shadow_paper_from_live_positions(
            state,
            {
                "BTCUSDT": {"qty": 0.0, "mark_price": 74_000.0},
                "BNBUSDT": {"qty": 0.0, "mark_price": 615.0},
            },
            equity=4_150.0,
        )

        shadow = state["shadow_paper"]
        self.assertEqual(shadow["source_mode"], "live")
        self.assertAlmostEqual(shadow["baseline_equity"], 4_150.0)
        self.assertAlmostEqual(shadow["peak_equity"], 4_150.0)
        self.assertAlmostEqual(shadow["max_drawdown"], 0.0)
        self.assertAlmostEqual(shadow["return_pct"], 0.0)
        self.assertAlmostEqual(shadow["turnover_cost_paid"], 0.0)


if __name__ == "__main__":
    unittest.main()
