"""Unit tests for _on_bar_outcome: run_live_loop retry/give-up state machine."""
import sys
import unittest
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from pairwise_regime_live import _on_bar_outcome  # noqa: E402

TARGET = 1_000_000_200.0  # arbitrary bar-close timestamp
MAX = 3


class TestOnBarOutcome(unittest.TestCase):
    """_on_bar_outcome state-machine correctness."""

    # ------------------------------------------------------------------
    # Success path
    # ------------------------------------------------------------------

    def test_success_first_attempt_advances_last_processed(self) -> None:
        """성공 → last_processed = target, attempt_count = 0."""
        lp, la, ac = _on_bar_outcome(
            success=True,
            target=TARGET,
            last_processed=None,
            last_attempted=None,
            attempt_count=0,
            max_attempts=MAX,
        )
        self.assertEqual(lp, TARGET)
        self.assertEqual(la, TARGET)
        self.assertEqual(ac, 0)

    def test_success_after_retry_advances_last_processed(self) -> None:
        """재시도 중 성공 → last_processed = target, attempt_count = 0."""
        lp, la, ac = _on_bar_outcome(
            success=True,
            target=TARGET,
            last_processed=None,
            last_attempted=TARGET,
            attempt_count=1,
            max_attempts=MAX,
        )
        self.assertEqual(lp, TARGET)
        self.assertEqual(ac, 0)

    # ------------------------------------------------------------------
    # Failure path — retries remaining
    # ------------------------------------------------------------------

    def test_failure_first_attempt_keeps_last_processed(self) -> None:
        """실패 1회, 재시도 남음 → last_processed 유지, attempt_count=1."""
        prev_lp = TARGET - 300.0  # 이전 봉
        lp, la, ac = _on_bar_outcome(
            success=False,
            target=TARGET,
            last_processed=prev_lp,
            last_attempted=None,
            attempt_count=0,
            max_attempts=MAX,
        )
        self.assertEqual(lp, prev_lp)   # 유지
        self.assertEqual(la, TARGET)
        self.assertEqual(ac, 1)

    def test_failure_second_attempt_keeps_last_processed(self) -> None:
        """실패 2회, 재시도 1회 남음 → last_processed 유지, attempt_count=2."""
        prev_lp = TARGET - 300.0
        lp, la, ac = _on_bar_outcome(
            success=False,
            target=TARGET,
            last_processed=prev_lp,
            last_attempted=TARGET,
            attempt_count=1,
            max_attempts=MAX,
        )
        self.assertEqual(lp, prev_lp)
        self.assertEqual(ac, 2)

    # ------------------------------------------------------------------
    # Failure path — cap reached → give up
    # ------------------------------------------------------------------

    def test_failure_cap_reached_advances_last_processed(self) -> None:
        """실패 3회 (cap) → last_processed = target (포기), attempt_count = 0."""
        prev_lp = TARGET - 300.0
        lp, la, ac = _on_bar_outcome(
            success=False,
            target=TARGET,
            last_processed=prev_lp,
            last_attempted=TARGET,
            attempt_count=2,  # already at attempt 2; this call bumps to 3 == MAX
            max_attempts=MAX,
        )
        self.assertEqual(lp, TARGET)   # 진행
        self.assertEqual(la, TARGET)
        self.assertEqual(ac, 0)        # 리셋

    # ------------------------------------------------------------------
    # New-bar detection — attempt counter resets across bars
    # ------------------------------------------------------------------

    def test_different_bar_resets_attempt_count(self) -> None:
        """다른 봉 → attempt_count 1로 리셋 (이전 봉 누적 카운트 무관)."""
        new_target = TARGET + 300.0
        prev_lp = TARGET
        lp, la, ac = _on_bar_outcome(
            success=False,
            target=new_target,
            last_processed=prev_lp,
            last_attempted=TARGET,   # 이전 봉 attempted
            attempt_count=2,         # 이전 봉에서 쌓인 count
            max_attempts=MAX,
        )
        # New bar: attempt_count should reset to 1, not 3
        self.assertEqual(la, new_target)
        self.assertEqual(ac, 1)
        self.assertEqual(lp, prev_lp)  # 실패이므로 유지


if __name__ == "__main__":
    unittest.main()
