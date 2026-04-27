"""Regression tests for the two-step promotion approval workflow.

These pin the security-critical invariants:

- Tokens cannot be self-approved (skin-in-the-game).
- Expired requests cannot be confirmed.
- `is_approval_active` returns False for missing / unconfirmed / expired
  tokens and True only for fresh confirmed approvals.
- `clean_expired` removes stale files but spares fresh ones.

CRITICAL: each test redirects PA.APPROVAL_DIR to a per-test
TemporaryDirectory and restores the production path on teardown. Without
this, running the suite would delete real `.bkit/runtime/promotion_approvals/*`
state — a live operator's pending unfreeze approval would vanish mid-flight.
The original production path is captured *before* any patching so it
cannot accidentally be lost.
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
import time
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import promotion_approval as PA  # noqa: E402


class TestPromotionApproval(unittest.TestCase):
    def setUp(self):
        # Snapshot the production APPROVAL_DIR so we can restore it.
        self._prod_dir = PA.APPROVAL_DIR
        # Per-test isolated directory — never touch production state.
        self._tmpdir = tempfile.TemporaryDirectory()
        sandbox = Path(self._tmpdir.name) / "promotion_approvals"
        sandbox.mkdir(parents=True, exist_ok=True)
        PA.APPROVAL_DIR = sandbox

    def tearDown(self):
        # Restore the production constant before cleanup so any teardown
        # in the module sees the canonical path.
        PA.APPROVAL_DIR = self._prod_dir
        self._tmpdir.cleanup()

    def test_request_creates_file(self):
        req = PA.request_approval("test reason", requester="alice", ttl_minutes=5)
        self.assertTrue((PA.APPROVAL_DIR / f"{req.token}.request.json").exists())

    def test_unconfirmed_request_is_not_active(self):
        req = PA.request_approval("test reason", requester="alice", ttl_minutes=5)
        status = PA.is_approval_active(req.token)
        self.assertFalse(status.active)
        self.assertIn("awaiting", status.reason)

    def test_self_approve_blocked(self):
        # Skin-in-the-game: requester cannot confirm their own request.
        req = PA.request_approval("test reason", requester="alice", ttl_minutes=5)
        result = PA.confirm_approval(req.token, approver="alice")
        self.assertFalse(result.active)
        self.assertIn("self-approve", result.reason)

    def test_other_party_approval_unlocks(self):
        req = PA.request_approval("test reason", requester="alice", ttl_minutes=5)
        result = PA.confirm_approval(req.token, approver="bob")
        self.assertTrue(result.active)
        status = PA.is_approval_active(req.token)
        self.assertTrue(status.active)

    def test_expired_ttl_blocks_confirmation(self):
        req = PA.request_approval("expired demo", requester="alice", ttl_minutes=0)
        time.sleep(0.05)
        result = PA.confirm_approval(req.token, approver="bob")
        self.assertFalse(result.active)
        self.assertIn("expired", result.reason)

    def test_expired_after_confirmation_loses_active(self):
        # Confirm immediately, then look at it after the TTL passes.
        req = PA.request_approval("short ttl", requester="alice", ttl_minutes=0)
        # Force-confirm before expiry by injecting the approval file directly.
        PA.APPROVAL_DIR.mkdir(parents=True, exist_ok=True)
        appr = {
            "token": req.token,
            "approver": "bob",
            "approved_at": datetime.now(timezone.utc).isoformat(),
        }
        (PA.APPROVAL_DIR / f"{req.token}.approved.json").write_text(json.dumps(appr))
        time.sleep(0.05)
        status = PA.is_approval_active(req.token)
        self.assertFalse(status.active)
        self.assertIn("expired", status.reason)

    def test_unknown_token_inactive(self):
        status = PA.is_approval_active("not_a_real_token")
        self.assertFalse(status.active)

    def test_clean_expired_removes_stale(self):
        # Create a stale request manually
        old = datetime.now(timezone.utc) - timedelta(days=14)
        PA.APPROVAL_DIR.mkdir(parents=True, exist_ok=True)
        stale_path = PA.APPROVAL_DIR / "deadbeef.request.json"
        stale_path.write_text(json.dumps({
            "token": "deadbeef",
            "reason": "old",
            "requested_at": old.isoformat(),
            "expires_at": (old + timedelta(minutes=5)).isoformat(),
            "requester": "alice",
            "files_changed": [],
        }))
        # And a fresh one
        fresh = PA.request_approval("fresh", requester="alice", ttl_minutes=10)
        n = PA.clean_expired(max_age_days=7.0)
        self.assertGreaterEqual(n, 1)
        self.assertFalse(stale_path.exists())
        self.assertTrue((PA.APPROVAL_DIR / f"{fresh.token}.request.json").exists())


if __name__ == "__main__":
    unittest.main()
