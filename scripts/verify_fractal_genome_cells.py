#!/usr/bin/env python3
"""Independent verification assets for the fractal genome cell structure."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from fractal_genome_core import (
    AndCell,
    ConditionNode,
    ConditionSpec,
    FilterDecision,
    LeafNode,
    NotCell,
    OrCell,
    ThresholdCell,
    deserialize_logic,
    deserialize_tree,
    evaluate_tree_codes,
    heuristic_semantic_filter,
    logic_key,
    serialize_logic,
    serialize_tree,
    tree_key,
    tree_logic_depth,
    tree_logic_size,
    tree_size,
)
from search_pair_subset_fractal_genome import build_expert_pool
from search_pair_subset_fractal_genome import build_feature_specs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify fractal genome cells with self-checks and a tiny smoke backtest.",
    )
    parser.add_argument(
        "--mode",
        choices=("self-check", "smoke", "btc-backtest", "mock-openai", "both"),
        default="both",
    )
    parser.add_argument(
        "--smoke-summary-out",
        default="/tmp/fractal_genome_smoke_summary.json",
    )
    parser.add_argument(
        "--smoke-review-out",
        default="/tmp/fractal_genome_smoke_reviews.jsonl",
    )
    parser.add_argument(
        "--smoke-command-log",
        default="/tmp/fractal_genome_smoke_command.json",
    )
    parser.add_argument(
        "--search-script",
        default=str(Path(__file__).resolve().with_name("search_pair_subset_fractal_genome.py")),
    )
    parser.add_argument(
        "--btc-backtest-script",
        default=str(Path(__file__).resolve().with_name("verify_fractal_genome_btc_curriculum.py")),
    )
    parser.add_argument(
        "--mock-openai-script",
        default=str(Path(__file__).resolve().with_name("verify_fractal_genome_openai_mock.py")),
    )
    parser.add_argument(
        "--btc-summary-out",
        default="/tmp/fractal_genome_btc_curriculum_summary.json",
    )
    parser.add_argument(
        "--btc-command-log",
        default="/tmp/fractal_genome_btc_curriculum_command.json",
    )
    parser.add_argument(
        "--btc-depth-curriculum",
        default="1,2,3",
    )
    parser.add_argument(
        "--btc-logic-curriculum",
        default="1,1,2",
    )
    parser.add_argument(
        "--btc-population",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--btc-generations",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--btc-elite-count",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--btc-top-k",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--btc-seed",
        type=int,
        default=123,
    )
    parser.add_argument(
        "--mock-openai-summary-out",
        default="/tmp/fractal_genome_mock_openai_summary.json",
    )
    parser.add_argument(
        "--mock-openai-review-out",
        default="/tmp/fractal_genome_mock_openai_reviews.jsonl",
    )
    parser.add_argument(
        "--mock-openai-command-log",
        default="/tmp/fractal_genome_mock_openai_command.json",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python interpreter used for the smoke backtest subprocess.",
    )
    return parser.parse_args()


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, FilterDecision):
        return {
            "accepted": value.accepted,
            "source": value.source,
            "reason": value.reason,
            "llm_prompt": value.llm_prompt,
        }
    return value


def build_reference_tree() -> ConditionNode:
    root_condition = AndCell(
        left=ThresholdCell(ConditionSpec("btc_regime", ">=", 0.0)),
        right=NotCell(child=ThresholdCell(ConditionSpec("breadth", "<=", 0.50))),
    )
    nested_condition = OrCell(
        left=ThresholdCell(ConditionSpec("btc_vol_rel", "<=", 1.0)),
        right=ThresholdCell(ConditionSpec("breadth", ">=", 0.65)),
    )
    return ConditionNode(
        condition=root_condition,
        if_true=LeafNode(7),
        if_false=ConditionNode(
            condition=nested_condition,
            if_true=LeafNode(3),
            if_false=LeafNode(1),
        ),
    )


def assert_feature_expansion(report: dict[str, Any]) -> None:
    feature_set = report.get("search", {}).get("feature_set", {})
    features = {str(item.get("feature")) for item in feature_set.get("features", [])}
    required_features = {
        "btc_regime",
        "breadth",
        "btc_vol_rel",
        "btc_momentum_1d",
        "btc_momentum_3d",
        "btc_momentum_accel_1d_3d",
        "btc_drawdown_7d",
        "btc_rsi_14_1h",
        "btc_mfi_14_1h",
        "btc_atr_pct_1h",
        "btc_macd_h_pct_1h",
        "btc_volume_rel_1h",
        "btc_dc_trend_05_1h",
        "bnb_regime",
        "bnb_vol_rel",
        "bnb_momentum_1d",
        "bnb_rsi_14_1h",
        "bnb_macd_h_pct_1h",
        "rel_strength_bnb_btc_3d",
        "regime_spread_btc_minus_bnb",
        "breadth_change_1d",
        "vol_rel_spread_btc_minus_bnb",
        "macd_h_pct_spread_btc_minus_bnb_1h",
        "volume_rel_spread_btc_minus_bnb_1h",
        "intraday_return_spread_btc_minus_bnb_1h",
    }
    missing = sorted(required_features - features)
    assert not missing, f"search feature expansion missing required features: {missing}"
    assert len(features) >= 70, f"expected materially expanded feature catalog, got only {len(features)} features"
    assert feature_set.get("feature_context", {}).get("primary_pair") == "BTCUSDT", "primary pair must remain BTCUSDT"
    assert feature_set.get("feature_context", {}).get("secondary_pair") == "BNBUSDT", "secondary pair must remain BNBUSDT"
    assert feature_set.get("feature_context", {}).get("single_asset_mode") is False, "smoke search must remain multi-pair"


def self_check() -> dict[str, Any]:
    reference_tree = build_reference_tree()

    roundtrip = deserialize_tree(serialize_tree(reference_tree))
    assert tree_key(reference_tree) == tree_key(roundtrip), "tree roundtrip changed the serialized shape"
    roundtrip_logic = deserialize_logic(serialize_logic(reference_tree.condition))
    assert logic_key(reference_tree.condition) == logic_key(roundtrip_logic), "logic roundtrip changed the serialized shape"
    assert tree_logic_depth(reference_tree) >= 1, "reference tree must exercise recursive logic depth"
    assert tree_logic_size(reference_tree) >= 5, "reference tree should contain more than a trivial logic cell budget"

    features = {
        "btc_regime": np.asarray([-0.10, 0.01, 0.20, 0.20], dtype="float64"),
        "breadth": np.asarray([0.40, 0.70, 0.40, 0.90], dtype="float64"),
        "btc_vol_rel": np.asarray([1.20, 1.10, 0.90, 1.20], dtype="float64"),
    }
    codes = evaluate_tree_codes(reference_tree, features)
    expected = np.asarray([1, 7, 3, 7], dtype="int16")
    assert np.array_equal(codes, expected), f"composite logic evaluation mismatch: {codes.tolist()} != {expected.tolist()}"

    feature_specs = build_feature_specs(("BTCUSDT", "BNBUSDT"))
    feature_names = {spec[0] for spec in feature_specs}
    required_feature_names = {
        "btc_regime",
        "breadth",
        "btc_vol_rel",
        "btc_momentum_1d",
        "btc_momentum_3d",
        "btc_momentum_accel_1d_3d",
        "btc_drawdown_7d",
        "btc_rsi_14_1h",
        "btc_mfi_14_1h",
        "btc_atr_pct_1h",
        "btc_macd_h_pct_1h",
        "btc_volume_rel_1h",
        "btc_dc_trend_05_1h",
        "bnb_regime",
        "bnb_vol_rel",
        "bnb_momentum_1d",
        "bnb_rsi_14_1h",
        "bnb_macd_h_pct_1h",
        "rel_strength_bnb_btc_3d",
        "regime_spread_btc_minus_bnb",
        "breadth_change_1d",
        "vol_rel_spread_btc_minus_bnb",
        "macd_h_pct_spread_btc_minus_bnb_1h",
        "volume_rel_spread_btc_minus_bnb_1h",
        "intraday_return_spread_btc_minus_bnb_1h",
    }
    missing_features = sorted(required_feature_names - feature_names)
    assert not missing_features, f"feature expansion regression: missing {missing_features}"
    assert len(feature_names) >= 70, f"feature expansion regression: expected >=70 features, got {len(feature_names)}"

    reject_subtree = ConditionNode(
        condition=ThresholdCell(ConditionSpec("breadth", ">=", 0.65)),
        if_true=LeafNode(0),
        if_false=LeafNode(1),
    )
    reject_tree = ConditionNode(
        condition=ThresholdCell(ConditionSpec("btc_regime", ">=", 0.0)),
        if_true=reject_subtree,
        if_false=reject_subtree,
    )
    rejection = heuristic_semantic_filter(
        reject_tree,
        expert_pool=[
            {"pair_configs": {"BTCUSDT": {"route_breadth_threshold": 0.35, "mapping_indices": [0, 1, 2]}}, "score": 1.0},
            {"pair_configs": {"BTCUSDT": {"route_breadth_threshold": 0.50, "mapping_indices": [1, 2, 3]}}, "score": 0.8},
        ],
        max_depth=3,
    )
    assert not rejection.accepted, "heuristic reject case should be rejected"
    assert rejection.reason == "identical_branches", f"unexpected heuristic reject reason: {rejection.reason}"

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        btc_only = {
            "selected_candidate": {
                "pair_configs": {
                    "BTCUSDT": {"route_breadth_threshold": 0.35, "mapping_indices": [0, 1, 2]}
                },
                "score": 1.0,
            }
        }
        btc_duplicate = {
            "selected_candidate": {
                "pair_configs": {
                    "BTCUSDT": {"route_breadth_threshold": 0.35, "mapping_indices": [0, 1, 2]}
                },
                "score": 0.5,
            }
        }
        bnb_only = {
            "selected_candidate": {
                "pair_configs": {
                    "BNBUSDT": {"route_breadth_threshold": 0.35, "mapping_indices": [0, 1, 2]}
                },
                "score": 0.9,
            }
        }
        paths = []
        for idx, payload in enumerate((btc_only, btc_duplicate, bnb_only), start=1):
            path = tmp_path / f"expert_{idx}.json"
            path.write_text(json.dumps(payload, ensure_ascii=False) + "\n")
            paths.append(str(path))
        pooled, diagnostics = build_expert_pool(paths, 10, ("BTCUSDT",))
        assert len(pooled) == 1, f"BTC projected pool should dedupe and filter to one candidate, got {len(pooled)}"
        assert diagnostics["projection_collision_count"] == 1, "BTC projected pool should record one projected duplicate collision"
        assert "BTCUSDT" in pooled[0]["pair_configs"], "BTC projected pool lost required pair config"
        assert "BNBUSDT" not in pooled[0]["pair_configs"], "BTC projected pool should exclude non-required pair configs"

    summary = {
        "roundtrip_tree_key": tree_key(roundtrip),
        "logic_key": logic_key(reference_tree.condition),
        "codes": codes.tolist(),
        "tree_size": tree_size(reference_tree),
        "tree_logic_size": tree_logic_size(reference_tree),
        "tree_logic_depth": tree_logic_depth(reference_tree),
        "heuristic_reject": rejection,
        "expert_pool_projection": {
            "deduped_count": 1,
            "required_pairs": ["BTCUSDT"],
        },
    }
    return summary


def run_btc_backtest(
    summary_out: str,
    command_log: str,
    btc_backtest_script: str,
    depth_curriculum: str,
    logic_curriculum: str,
    population: int,
    generations: int,
    elite_count: int,
    top_k: int,
    seed: int,
    python: str,
) -> dict[str, Any]:
    summary_path = Path(summary_out)
    command_path = Path(command_log)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    command_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        python,
        btc_backtest_script,
        "--summary-out",
        str(summary_path),
        "--depth-curriculum",
        depth_curriculum,
        "--logic-curriculum",
        logic_curriculum,
        "--population",
        str(population),
        "--generations",
        str(generations),
        "--elite-count",
        str(elite_count),
        "--top-k",
        str(top_k),
        "--seed",
        str(seed),
    ]
    completed = subprocess.run(cmd, capture_output=True, text=True, check=True)
    report = json.loads(summary_path.read_text())
    command_log_obj = {
        "command": cmd,
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
        "summary_path": str(summary_path),
    }
    command_path.write_text(json.dumps(json_safe(command_log_obj), ensure_ascii=False, indent=2) + "\n")
    btc_summary = {
        "command": cmd,
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
        "summary_path": str(summary_path),
        "selection": report.get("selection", {}),
        "overall": report.get("overall", {}),
        "stage_count": len(report.get("stages", [])),
    }
    assert btc_summary["stage_count"] >= 1, "btc backtest should produce at least one curriculum stage"
    assert report.get("window_contract", {}).get("full_start_matches_first_data"), "full-range window must start at the first collected bar"
    return btc_summary


def run_smoke_backtest(summary_out: str, review_out: str, search_script: str, python: str) -> dict[str, Any]:
    summary_path = Path(summary_out)
    review_path = Path(review_out)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    review_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        python,
        search_script,
        "--pairs",
        "BTCUSDT,BNBUSDT",
        "--expert-pool-size",
        "4",
        "--population",
        "6",
        "--generations",
        "1",
        "--elite-count",
        "2",
        "--top-k",
        "2",
        "--max-depth",
        "2",
        "--logic-max-depth",
        "1",
        "--seed",
        "123",
        "--filter-mode",
        "heuristic",
        "--summary-out",
        str(summary_path),
        "--llm-review-out",
        str(review_path),
    ]
    completed = subprocess.run(cmd, capture_output=True, text=True, check=True)
    report = json.loads(summary_path.read_text())
    assert_feature_expansion(report)
    smoke_summary = {
        "command": cmd,
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
        "summary_path": str(summary_path),
        "review_path": str(review_path),
        "selection": report.get("selection", {}),
        "runtime": report.get("runtime", {}),
        "top_candidates": len(report.get("top_candidates", [])),
        "selected_candidate_present": report.get("selected_candidate") is not None,
    }
    assert smoke_summary["top_candidates"] >= 1, "smoke backtest should produce at least one candidate"
    assert smoke_summary["runtime"].get("evaluated_unique_candidates", 0) >= 1, "smoke backtest should evaluate at least one unique tree"
    return smoke_summary


def run_mock_openai_backtest(
    summary_out: str,
    review_out: str,
    command_log: str,
    mock_openai_script: str,
    search_script: str,
    python: str,
) -> dict[str, Any]:
    summary_path = Path(summary_out)
    review_path = Path(review_out)
    command_path = Path(command_log)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    review_path.parent.mkdir(parents=True, exist_ok=True)
    command_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        python,
        mock_openai_script,
        "--search-script",
        search_script,
        "--command-log",
        str(command_path),
        "--summary-out",
        str(summary_path),
        "--review-out",
        str(review_path),
    ]
    completed = subprocess.run(cmd, capture_output=True, text=True, check=True)
    report = json.loads(summary_path.read_text())
    assert_feature_expansion(report)

    mock_summary = {
        "command": cmd,
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
        "summary_path": str(summary_path),
        "review_path": str(review_path),
        "mock_openai": report.get("mock_openai", {}),
        "search": report.get("search", {}),
    }
    command_path.write_text(json.dumps(json_safe(mock_summary), ensure_ascii=False, indent=2) + "\n")
    assert mock_summary["mock_openai"].get("request_count", 0) >= 1, "mock OpenAI server should receive at least one request"
    assert mock_summary["mock_openai"].get("path") == "/v1/chat/completions", "mock OpenAI path must match the search client"
    assert mock_summary["mock_openai"].get("auth_header", "").startswith("Bearer "), "mock OpenAI path should use bearer auth even without an external key"
    assert mock_summary["search"].get("auto_llm_review_events"), "mock OpenAI run should record auto LLM review events"
    first_event = mock_summary["search"]["auto_llm_review_events"][0]
    assert first_event.get("enabled") is True, "auto LLM review must be enabled in the mock OpenAI run"
    assert first_event.get("attempted", 0) >= 1, "auto LLM review should attempt at least one review"
    assert first_event.get("added", 0) >= 1, "auto LLM review should add at least one reviewed candidate"
    return mock_summary


def main() -> None:
    args = parse_args()
    report: dict[str, Any] = {"mode": args.mode}

    if args.mode in {"self-check", "both"}:
        report["self_check"] = self_check()

    if args.mode in {"smoke", "both"}:
        smoke = run_smoke_backtest(args.smoke_summary_out, args.smoke_review_out, args.search_script, args.python)
        Path(args.smoke_command_log).write_text(json.dumps(json_safe(smoke), ensure_ascii=False, indent=2) + "\n")
        report["smoke"] = smoke

    if args.mode in {"mock-openai", "both"}:
        mock_openai = run_mock_openai_backtest(
            args.mock_openai_summary_out,
            args.mock_openai_review_out,
            args.mock_openai_command_log,
            args.mock_openai_script,
            args.search_script,
            args.python,
        )
        report["mock_openai"] = mock_openai

    if args.mode in {"btc-backtest", "both"}:
        btc = run_btc_backtest(
            args.btc_summary_out,
            args.btc_command_log,
            args.btc_backtest_script,
            args.btc_depth_curriculum,
            args.btc_logic_curriculum,
            args.btc_population,
            args.btc_generations,
            args.btc_elite_count,
            args.btc_top_k,
            args.btc_seed,
            args.python,
        )
        report["btc_backtest"] = btc

    print(json.dumps(json_safe(report), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
