"""Unit tests for _seconds_until_next_bar bar-close alignment helper."""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


# Import only the helpers; avoid triggering heavy module-level side effects
# by importing the functions directly after path setup.
from pairwise_regime_live import _bar_close_at, _seconds_until_next_bar  # noqa: E402


class TestSecondsUntilNextBar(unittest.TestCase):
    """_seconds_until_next_bar correctness cases."""

    def _mock_time(
        self,
        epoch: float,
        timeframe: int = 300,
        offset: float = 5.0,
        last_processed_close: float | None = None,
    ) -> float:
        with patch("time.time", return_value=epoch):
            return _seconds_until_next_bar(last_processed_close, timeframe, offset)

    # ------------------------------------------------------------------
    # Existing 8 cases (last_processed_close=None → first-run semantics)
    # ------------------------------------------------------------------

    def test_just_after_bar_open(self) -> None:
        # epoch where now mod 300 == 1 -> 1 second into bar
        # 1_000_000_201 % 300 == 1, next close = 1_000_000_500, sleep = 299 + 5 = 304
        # With last=None: target_close = floor(now/300)*300 = 1_000_000_200
        # sleep = 1_000_000_200 + 5 - 1_000_000_201 = 4.0
        epoch = 1_000_000_201  # 1_000_000_201 % 300 == 1
        result = self._mock_time(epoch, last_processed_close=None)
        expected = (1_000_000_200 + 5.0) - epoch  # 4.0
        self.assertAlmostEqual(result, expected, places=6)

    def test_just_before_bar_close(self) -> None:
        # epoch where mod 300 == 299 -> 1s before close of current bar
        # last=None: target_close = floor(now/300)*300 = 1_000_000_200
        # sleep = 1_000_000_200 + 5 - 1_000_000_299 = -94 -> clamped to 0
        epoch = 1_000_000_299
        result = self._mock_time(epoch, last_processed_close=None)
        self.assertEqual(result, 0.0)

    def test_exactly_on_bar_open(self) -> None:
        # now == exact bar close (e.g. 10:05:00.000)
        # last=None: target_close = floor(now/300)*300 = now itself = 1_000_000_200
        # sleep = 1_000_000_200 + 5 - 1_000_000_200 = 5.0
        epoch = 1_000_000_200  # 1_000_000_200 % 300 == 0
        result = self._mock_time(epoch, last_processed_close=None)
        expected = 5.0
        self.assertAlmostEqual(result, expected, places=6)

    def test_mid_bar(self) -> None:
        # now == 10:02:30 -> 150 s into bar
        # last=None: target_close = floor(1_000_000_350/300)*300 = 1_000_000_200
        # sleep = 1_000_000_200 + 5 - 1_000_000_350 = -145 -> 0
        epoch = 1_000_000_350  # % 300 == 150
        result = self._mock_time(epoch, last_processed_close=None)
        self.assertEqual(result, 0.0)

    def test_zero_offset(self) -> None:
        # offset=0: target_close = 1_000_000_200, sleep = 1_000_000_200 - 1_000_000_201 = -1 -> 0
        epoch = 1_000_000_201
        result = self._mock_time(epoch, timeframe=300, offset=0.0, last_processed_close=None)
        self.assertEqual(result, 0.0)

    def test_custom_timeframe(self) -> None:
        # 15-min bars (900 s)
        # last=None: target_close = floor(1_000_000_001/900)*900 = 1_111_111*900 = 999_999_900
        # sleep = 999_999_900 + 5 - 1_000_000_001 = -96 -> 0
        epoch = 1_000_000_001  # % 900 == 101
        result = self._mock_time(epoch, timeframe=900, offset=5.0, last_processed_close=None)
        self.assertEqual(result, 0.0)

    def test_never_returns_negative(self) -> None:
        # result should always be >= 0
        epoch = 1_000_000_205  # 5 s after bar close at ...200
        result = self._mock_time(epoch, offset=5.0, last_processed_close=None)
        self.assertGreaterEqual(result, 0.0)

    def test_custom_post_close_offset(self) -> None:
        # With last_processed provided so target is in the future, offset diff is exactly 5s
        # last_processed = 1_000_000_200, next target = 1_000_000_500
        # epoch = 1_000_000_201 (now just after last close, waiting for next bar)
        epoch = 1_000_000_201
        last = 1_000_000_200
        result_5 = self._mock_time(epoch, offset=5.0, last_processed_close=last)
        result_10 = self._mock_time(epoch, offset=10.0, last_processed_close=last)
        self.assertAlmostEqual(result_10 - result_5, 5.0, places=6)

    # ------------------------------------------------------------------
    # New 4 cases: catch-up / first-run semantics
    # ------------------------------------------------------------------

    def test_skips_bar_when_work_overruns(self) -> None:
        """시나리오 B: work 320초 걸려 10:05 봉이 닫힌 후 종료 → 즉시 catch-up (0 반환)."""
        # last_processed_close = 10:00:00 = base + 0
        # next target = 10:05:00 + 5 = 10:05:05
        # now = 10:05:25 → target already passed → 0
        base = 1_000_000_200  # 10:00:00 bar close
        last = base           # last processed close
        epoch = base + 325.0  # 10:05:25
        result = self._mock_time(epoch, last_processed_close=last)
        self.assertEqual(result, 0.0)

    def test_first_run_targets_most_recent_close(self) -> None:
        """시나리오 C: 프로세스가 봉 close 직후 2초에 시작 → 3초 대기 (10:00:05까지)."""
        # now = 10:00:02, last=None
        # target_close = floor(now/300)*300 = 10:00:00
        # sleep = 10:00:05 - 10:00:02 = 3.0
        bar_close = 1_000_000_200  # represents 10:00:00
        epoch = bar_close + 2.0    # 10:00:02
        result = self._mock_time(epoch, offset=5.0, last_processed_close=None)
        self.assertAlmostEqual(result, 3.0, places=6)

    def test_first_run_at_exact_bar_close(self) -> None:
        """last=None, now=10:00:00.000 → 5초 대기 (offset=5s)."""
        bar_close = 1_000_000_200
        epoch = float(bar_close)
        result = self._mock_time(epoch, offset=5.0, last_processed_close=None)
        self.assertAlmostEqual(result, 5.0, places=6)

    def test_no_drift_after_long_work(self) -> None:
        """연속 2사이클: 첫 번째 처리 후 last 설정 → 두 번째 호출이 정확히 다음 봉 타겟."""
        # Cycle 1: now=10:00:10, last=None → target=10:00:00+5=10:00:05 → 0 (past)
        bar_close = 1_000_000_200
        epoch1 = bar_close + 10.0
        r1 = self._mock_time(epoch1, offset=5.0, last_processed_close=None)
        self.assertEqual(r1, 0.0)

        # After cycle 1 finishes at 10:00:55, last_processed_close = bar_close = 10:00:00
        epoch2 = bar_close + 55.0  # 10:00:55
        last_after_cycle1 = bar_close  # floor(epoch2/300)*300 = bar_close
        r2 = self._mock_time(epoch2, offset=5.0, last_processed_close=last_after_cycle1)
        # next target = 10:05:00 + 5 = 10:05:05; sleep = 10:05:05 - 10:00:55 = 250.0
        self.assertAlmostEqual(r2, 250.0, places=6)


class TestBarCloseAt(unittest.TestCase):
    """_bar_close_at: 시작 시각 기준 봉 close 캡처 정확성 테스트."""

    def test_normal(self) -> None:
        # 10:02:30 → 속한 봉 close = 10:00:00
        bar_close = 1_000_000_200
        ts = bar_close + 150.0  # 10:02:30
        self.assertEqual(_bar_close_at(ts, 300), float(bar_close))

    def test_exact_bar_boundary(self) -> None:
        # 정각 10:05:00 → 속한 봉 close = 10:05:00 (다음 봉이 아닌 현재 경계)
        bar_close = 1_000_000_500
        self.assertEqual(_bar_close_at(float(bar_close), 300), float(bar_close))

    def test_overrun_does_not_skip_next_bar(self) -> None:
        """오버런 시나리오: run_live_once가 305초 걸려도 last_processed는 시작 시각 봉으로 기록.

        시뮬레이션:
          - start_ts = 10:05:05 → 처리하려는 봉 = 10:05:00
          - end_ts   = 10:10:10 (305초 경과)
          - 올바른 last_processed_close = 10:05:00 (시작 시각 봉)
          - 버그 동작 = 10:10:00 (종료 시각 봉) → 10:10 봉 누락
        """
        interval = 300
        start_ts = 1_000_000_505  # 10:05:05
        end_ts = start_ts + 305   # 10:10:10

        # 수정된 동작: 시작 시각에 캡처
        correct = _bar_close_at(start_ts, interval)  # 10:05:00
        # 버그 동작: 종료 시각에 캡처
        buggy = _bar_close_at(end_ts, interval)      # 10:10:00

        self.assertEqual(correct, 1_000_000_500)  # 10:05:00
        self.assertEqual(buggy, 1_000_000_800)    # 10:10:00 (버그 시 누락되는 봉)
        self.assertNotEqual(correct, buggy)        # 두 값이 다름을 명시적으로 검증

        # 수정 후: _seconds_until_next_bar(correct, interval, 5) 는 10:15:05까지 대기
        # _seconds_until_next_bar(buggy,   interval, 5) 는 10:15:05까지 대기하지만
        # 10:10 봉을 이미 건너뛴 상태이므로 누락 발생
        with patch("time.time", return_value=float(end_ts)):
            sleep_after_correct = _seconds_until_next_bar(correct, interval, 5.0)
            sleep_after_buggy = _seconds_until_next_bar(buggy, interval, 5.0)

        # correct: 다음 타겟 = 10:05:00 + 300 + 5 = 10:10:05; end_ts=10:10:10 → 이미 지남 → 0 (즉시 catch-up)
        self.assertAlmostEqual(sleep_after_correct, 0.0, places=6)
        # buggy:   다음 타겟 = 10:10:00 + 300 + 5 = 10:15:05; sleep = 10:15:05 - 10:10:10 = 295
        # → 10:10 봉을 건너뛰고 10:15까지 대기 (봉 누락)
        self.assertAlmostEqual(sleep_after_buggy, 295.0, places=6)


if __name__ == "__main__":
    unittest.main()
