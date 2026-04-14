import sys
import unittest
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


if __name__ == "__main__":
    unittest.main()
