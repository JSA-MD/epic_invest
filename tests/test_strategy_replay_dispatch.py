import sys
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import strategy_replay_dispatch as replay_dispatch


class StrategyReplayDispatchTests(unittest.TestCase):
    def test_pair_specific_convex_blend_dispatches_for_bnb(self) -> None:
        candidate = {
            "pair_configs": {
                "BNBUSDT": {
                    "mapping_indices": [1] * 12,
                    "route_breadth_threshold": 0.5,
                }
            },
            "pair_convex_blends": {
                "BNBUSDT": {
                    "alpha": 0.2,
                    "mode": "always",
                    "specialist_pair_config": {
                        "mapping_indices": [2] * 12,
                        "route_breadth_threshold": 0.5,
                    },
                }
            },
        }
        with (
            patch.object(replay_dispatch, "detect_candidate_kind", return_value="pairwise_candidate"),
            patch.object(replay_dispatch, "replay_btc_convex_blend_candidate", return_value={"engine": "blend"}) as replay_blend,
            patch.object(replay_dispatch, "realistic_overlay_replay_from_context") as replay_plain,
        ):
            result = replay_dispatch.replay_candidate_from_context(
                candidate=candidate,
                pair="BNBUSDT",
                context={},
                library_lookup={},
                route_thresholds=(0.5,),
                leaf_runtime_array=None,
                leaf_codes=None,
            )

        self.assertEqual(result, {"engine": "blend"})
        replay_blend.assert_called_once()
        replay_plain.assert_not_called()


if __name__ == "__main__":
    unittest.main()
