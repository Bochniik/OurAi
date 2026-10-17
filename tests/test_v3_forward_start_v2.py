"""Protect forward-only revision after original start was missed."""
from __future__ import annotations

import json
import unittest
from datetime import datetime, timezone
from pathlib import Path

from scripts.prepare_v3_forward_shadow import _validate_protocol
from scripts.run_v3_forward_shadow import (
    DEFAULT_JOURNAL, _write_current_predictions,
)


ROOT = Path(__file__).resolve().parents[1]
V1_PATH = ROOT / "research/v3_15m_magnitude_forward_protocol.json"
V2_PATH = ROOT / "research/v3_15m_magnitude_forward_protocol_v2.json"


class FutureForwardStartRevisionTests(unittest.TestCase):
    def test_new_protocol_keeps_champion_and_evaluation_frozen(self):
        old = json.loads(V1_PATH.read_text(encoding="utf-8"))
        new = json.loads(V2_PATH.read_text(encoding="utf-8"))
        _validate_protocol(new)
        self.assertEqual(new["forward_start_utc"], "2026-10-19T13:30:00Z")
        self.assertEqual(new["champion"], old["champion"])
        self.assertEqual(new["evaluation"], old["evaluation"])
        self.assertEqual(new["research_safety"], old["research_safety"])
        self.assertTrue(new["prediction_logging"]["never_backfill_before_forward_start"])

    def test_missing_future_start_or_changed_model_is_rejected(self):
        new = json.loads(V2_PATH.read_text(encoding="utf-8"))
        modified = dict(new)
        modified["forward_start_utc"] = "2026-10-15T13:30:00Z"
        with self.assertRaisesRegex(ValueError, "future-only"):
            _validate_protocol(modified)
        modified = json.loads(V2_PATH.read_text(encoding="utf-8"))
        modified["champion"]["learning_rate"] = 0.1
        with self.assertRaises(ValueError):
            _validate_protocol(modified)

    def test_no_prediction_before_new_start(self):
        bundle = {"forward_start_utc": "2026-10-19T13:30:00Z"}
        actual = _write_current_predictions(
            bundle=bundle, nonlinear=None, linear=None,
            journal_path="unused.jsonl",
            state={"predictions": {}, "resolutions": {}},
            now_utc=datetime(2026, 10, 18, 20, 0, tzinfo=timezone.utc),
            max_prediction_lag_seconds=90,
        )
        self.assertEqual(actual, 0)

    def test_new_shadow_journal_is_isolated_from_legacy(self):
        self.assertIn("forward_shadow_v2/", DEFAULT_JOURNAL)


if __name__ == "__main__":
    unittest.main()
