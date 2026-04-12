import sys
import unittest
from pathlib import Path
from unittest import mock


ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import search_pair_subset_pairwise_moo_router as moo_router


class PairwiseMooRouterTests(unittest.TestCase):
    def test_encode_candidate_from_individual_builds_hierarchical_pair_config(self) -> None:
        gene_len = len(moo_router.EXECUTION_GENE_OPTIONS)
        block_len = 1 + len(moo_router.SPECIALIST_ROLE_NAMES) + 12 + gene_len
        individual = [0] * (block_len * 2)
        individual[0] = 1
        individual[1:5] = [10, 20, 30, 40]
        individual[5:17] = [0, 1, 2, 3] * 3
        individual[block_len] = 0
        individual[block_len + 1:block_len + 5] = [11, 21, 31, 41]
        individual[block_len + 5:block_len + 17] = [3, 2, 1, 0] * 3

        candidate = moo_router.encode_candidate_from_individual(
            individual,
            pairs=("BTCUSDT", "BNBUSDT"),
            route_thresholds=(0.35, 0.50, 0.65),
        )

        btc = candidate["pair_configs"]["BTCUSDT"]
        self.assertEqual(btc["route_breadth_threshold"], 0.50)
        self.assertEqual(btc["mapping_indices"][:4], [10, 20, 30, 40])
        self.assertEqual(btc["route_state_mode"], "equity_corr")
        self.assertEqual(len(btc["mapping_indices"]), 12)
        self.assertIn("execution_gene", btc)
        self.assertIn("abstain_edge_pct", btc["execution_gene"])
        self.assertIn("flow_alignment_threshold", btc["execution_gene"])
        self.assertIn("dc_alignment_threshold", btc["execution_gene"])
        self.assertIn("min_alignment_votes", btc["execution_gene"])
        self.assertEqual(btc["execution_gene"]["flow_alignment_threshold"], 0.0)
        self.assertEqual(btc["execution_gene"]["dc_alignment_threshold"], 0.0)
        self.assertEqual(btc["execution_gene"]["min_alignment_votes"], 0)

    def test_neighbor_variants_mutate_alignment_gene_keys(self) -> None:
        gene_keys = list(moo_router.EXECUTION_GENE_OPTIONS)
        gene_len = len(gene_keys)
        block_len = 1 + len(moo_router.SPECIALIST_ROLE_NAMES) + 12 + gene_len
        individual = [0] * block_len
        start = 1 + len(moo_router.SPECIALIST_ROLE_NAMES) + 12
        individual[start + gene_keys.index("flow_alignment_threshold")] = 2
        individual[start + gene_keys.index("dc_alignment_threshold")] = 2
        individual[start + gene_keys.index("min_alignment_votes")] = 1
        individual[start + gene_keys.index("chase_distance_bp")] = 1

        candidate = moo_router.encode_candidate_from_individual(
            individual,
            pairs=("BTCUSDT",),
            route_thresholds=(0.35, 0.50, 0.65),
        )
        variants = moo_router.build_neighbor_variants(
            candidate,
            pairs=("BTCUSDT",),
            route_thresholds=(0.35, 0.50, 0.65),
        )

        flow_values = {
            variant["pair_configs"]["BTCUSDT"]["execution_gene"]["flow_alignment_threshold"]
            for variant in variants
        }
        dc_values = {
            variant["pair_configs"]["BTCUSDT"]["execution_gene"]["dc_alignment_threshold"]
            for variant in variants
        }
        vote_values = {
            variant["pair_configs"]["BTCUSDT"]["execution_gene"]["min_alignment_votes"]
            for variant in variants
        }
        chase_values = {
            variant["pair_configs"]["BTCUSDT"]["execution_gene"]["chase_distance_bp"]
            for variant in variants
        }

        self.assertIn(0.05, flow_values)
        self.assertIn(0.20, flow_values)
        self.assertIn(0.05, dc_values)
        self.assertIn(0.20, dc_values)
        self.assertIn(0, vote_values)
        self.assertIn(1, vote_values)
        self.assertIn(2, vote_values)
        self.assertIn(1.0, chase_values)
        self.assertIn(4.0, chase_values)
        self.assertTrue(
            any(
                variant["pair_configs"]["BTCUSDT"]["execution_gene"].get("signal_gate_pct") == 0.15
                and variant["pair_configs"]["BTCUSDT"]["execution_gene"].get("confirm_bars") == 2
                for variant in variants
            )
        )

    def test_compute_diversity_score_rewards_role_mix(self) -> None:
        concentrated = {
            "pair_configs": {
                "BTCUSDT": {
                    "specialist_indices": [10, 10, 10, 10],
                    "state_specialists": [0] * 12,
                }
            }
        }
        diversified = {
            "pair_configs": {
                "BTCUSDT": {
                    "specialist_indices": [10, 20, 30, 40],
                    "state_specialists": [0, 1, 2, 3] * 3,
                }
            }
        }

        self.assertGreater(
            moo_router.compute_diversity_score(diversified, ("BTCUSDT",)),
            moo_router.compute_diversity_score(concentrated, ("BTCUSDT",)),
        )

    def test_candidate_id_distinguishes_execution_gene_variants(self) -> None:
        base = {
            "candidate_kind": "pairwise_candidate",
            "pair_configs": {
                "BTCUSDT": {
                    "route_breadth_threshold": 0.5,
                    "route_state_mode": "equity_corr",
                    "mapping_indices": [10] * 12,
                },
                "BNBUSDT": {
                    "route_breadth_threshold": 0.5,
                    "route_state_mode": "equity_corr",
                    "mapping_indices": [20] * 12,
                },
            },
        }
        variant = {
            "candidate_kind": "pairwise_candidate",
            "pair_configs": {
                "BTCUSDT": {
                    "route_breadth_threshold": 0.5,
                    "route_state_mode": "equity_corr",
                    "mapping_indices": [10] * 12,
                    "execution_gene": {"signal_gate_pct": 0.3},
                },
                "BNBUSDT": {
                    "route_breadth_threshold": 0.5,
                    "route_state_mode": "equity_corr",
                    "mapping_indices": [20] * 12,
                },
            },
        }
        self.assertNotEqual(
            moo_router.candidate_id(base, ("BTCUSDT", "BNBUSDT")),
            moo_router.candidate_id(variant, ("BTCUSDT", "BNBUSDT")),
        )

    def test_build_role_specific_specialist_pools_uses_special_regime_profiles(self) -> None:
        def fake_fast_overlay(
            fast_context,
            library,
            library_lookup,
            mapping,
            route_threshold,
            fast_engine,
            **kwargs,
        ):
            marker = int(mapping[0])
            return {
                "daily_metrics": {
                    "avg_daily_return": 0.001,
                    "daily_target_hit_rate": 0.50,
                    "daily_win_rate": 0.50,
                    "worst_day": -0.01,
                    "best_day": 0.01,
                    "daily_returns": [float(marker)],
                },
                "total_return": 0.10,
                "max_drawdown": -0.05,
                "sharpe": 1.0,
                "n_trades": 10,
            }

        def fake_special_payload(fast_context, daily_returns):
            marker = int(daily_returns[0])
            regime_returns = {
                "trend_specialist_regime": [-0.01],
                "range_repair_regime": [-0.01],
                "panic_deleveraging_regime": [-0.01],
                "carry_basis_regime": [-0.01],
            }
            role_name = moo_router.SPECIALIST_ROLE_NAMES[marker]
            regime_returns[f"{role_name}_specialist_regime" if role_name == "trend" else (
                "range_repair_regime" if role_name == "range" else (
                    "panic_deleveraging_regime" if role_name == "panic" else "carry_basis_regime"
                )
            )] = [0.03, 0.02]
            return {
                "special_regime_returns": regime_returns,
                "daily_features": {},
            }

        window_cache = {
            label: {
                "pairs": {
                    "BTCUSDT": {
                        "fast_context": {
                            "validation_daily_index": [],
                        }
                    }
                }
            }
            for label in ("recent_2m", "recent_4m", "recent_6m", "recent_1y", "full_4y")
        }
        baseline_pair_configs = {
            "BTCUSDT": {
                "route_breadth_threshold": 0.50,
                "route_state_mode": "equity_corr",
                "specialist_indices": [],
                "mapping_indices": [],
            }
        }

        with mock.patch.object(
            moo_router,
            "fast_overlay_replay_from_context",
            side_effect=fake_fast_overlay,
        ), mock.patch.object(
            moo_router,
            "summarize_special_regime_payload_from_fast_context",
            side_effect=fake_special_payload,
        ):
            pools = moo_router.build_role_specific_specialist_pools(
                pairs=("BTCUSDT",),
                library=[object(), object(), object(), object()],
                library_lookup={},
                window_cache=window_cache,
                baseline_pair_configs=baseline_pair_configs,
                fast_engine="python",
                pool_size=4,
            )

        pair_pools = pools["BTCUSDT"]
        self.assertEqual(pair_pools["trend"][0], 0)
        self.assertEqual(pair_pools["range"][0], 1)
        self.assertEqual(pair_pools["panic"][0], 2)
        self.assertEqual(pair_pools["carry"][0], 3)


if __name__ == "__main__":
    unittest.main()
