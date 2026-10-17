"""Freeze and causality checks for v6 non-overlap direction research.

No GPU training or holdout access is performed in CI.
"""
from __future__ import annotations

import inspect
import unittest
from datetime import datetime, timedelta, timezone

from scripts.evaluate_v6_nonoverlap_gru_direction import (
    GRU_CONFIG, _load_frozen_protocol,
    run_experiment,
)
from trading.dataset import TradingSample
from trading.nonoverlap_selection import anchored_training_refs
from trading.temporal_sequences import SequenceRef


def _ref(minutes: int, direction: str = "UP", symbol: str = "NVDA") -> SequenceRef:
    timestamp = datetime(2025, 1, 6, 15, 0, tzinfo=timezone.utc) + timedelta(minutes=minutes)
    sample = TradingSample(
        symbol=symbol, timestamp=timestamp, session_date="2025-01-06",
        session="regular", features={},
        targets={"15m": {"direction": direction, "future_return": 0.0001}},
    )
    return SequenceRef(group_key=(symbol, "2025-01-06", "regular"), end_index=0, sample=sample)


class V6AnchoredDirectionTests(unittest.TestCase):
    def test_frozen_protocol_and_small_model(self) -> None:
        protocol = _load_frozen_protocol()
        self.assertEqual(protocol["training"]["anchor_modulus_minutes"], 15)
        self.assertTrue(protocol["training"]["no_label_selection_prior_to_time_anchors"])
        self.assertEqual(GRU_CONFIG.hidden_size, 32)
        self.assertEqual(GRU_CONFIG.num_layers, 1)
        self.assertEqual(GRU_CONFIG.epochs, 2)
        self.assertFalse(protocol["evaluation"]["pnl_used"])
        self.assertFalse(protocol["promotion"]["validation_or_test_opened"])

    def test_anchors_are_fixed_and_independent_of_future_labels(self) -> None:
        rows = [_ref(i, "DOWN" if i % 2 else "UP") for i in range(61)]
        selected = anchored_training_refs(rows)
        self.assertEqual([r.sample.timestamp for r in selected], [
            rows[i].sample.timestamp for i in (0, 15, 30, 45, 60)
        ])
        mutated = [_ref(i, "UP" if i % 2 else "DOWN") for i in range(61)]
        self.assertEqual(
            [r.sample.timestamp for r in anchored_training_refs(mutated)],
            [r.sample.timestamp for r in selected],
        )

    def test_anchors_do_not_select_by_realized_magnitude_or_neutral(self) -> None:
        rows = [_ref(15, "NEUTRAL"), _ref(16, "UP"), _ref(30, "DOWN")]
        self.assertEqual(
            [r.sample.timestamp for r in anchored_training_refs(rows)],
            [rows[0].sample.timestamp, rows[2].sample.timestamp],
        )

    def test_duplicate_anchor_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate"):
            anchored_training_refs([_ref(0), _ref(0)])

    def test_report_runner_does_not_open_global_holdout_files(self) -> None:
        source = inspect.getsource(run_experiment)
        self.assertIn('"train.jsonl"', source)
        self.assertNotIn('"validation.jsonl"', source)
        self.assertNotIn('"test.jsonl"', source)
        self.assertIn("anchored_training_refs", source)
        self.assertNotIn("_tail_directional_samples", source)


if __name__ == "__main__":
    unittest.main()
