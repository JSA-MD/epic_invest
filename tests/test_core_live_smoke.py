import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import core_strategy_registry
import rotation_target_050_live


class CoreLiveSmokeTests(unittest.TestCase):
    def test_core_champion_artifact_loads(self) -> None:
        artifact = core_strategy_registry.load_core_artifact(ROOT_DIR / "models" / "core_champion.json")
        self.assertIn(artifact.family, {core_strategy_registry.LONG_ONLY_FAMILY, core_strategy_registry.LONG_SHORT_FAMILY})
        self.assertTrue(artifact.key)
        self.assertTrue(artifact.source)
        self.assertIsNotNone(artifact.params)

    def test_live_state_example_matches_loader_defaults(self) -> None:
        with open(ROOT_DIR / "models" / "rotation_target_050_live_state.example.json", "r") as f:
            example = json.load(f)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "missing_state.json"
            generated = rotation_target_050_live.load_state(path)
        self.assertEqual(example["notification_state"].keys(), generated["notification_state"].keys())
        self.assertEqual(example["runtime_health"].keys(), generated["runtime_health"].keys())
        self.assertEqual(example["latest_runtime_snapshot"].keys(), generated["latest_runtime_snapshot"].keys())
        self.assertEqual(example["decision_journal"].keys(), generated["decision_journal"].keys())


if __name__ == "__main__":
    unittest.main()
