from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.build_v4_direction_dataset import (
    ADJUSTMENT,
    FEED,
    HORIZONS_MINUTES,
    NEXT_KNOWN_RESEARCH_PERIOD_START,
    SESSIONS,
    SOURCE_END,
    SOURCE_START,
    SYMBOLS,
    TRAIN_FRACTION,
    VALIDATION_FRACTION,
    build_v4_direction_dataset,
    preregistered_protocol,
)


class V4DirectionProtocolTests(unittest.TestCase):
    def test_protocol_is_fresh_non_overlapping_and_frozen(self) -> None:
        protocol = preregistered_protocol()
        self.assertLessEqual(SOURCE_END, NEXT_KNOWN_RESEARCH_PERIOD_START)
        self.assertEqual(protocol["source"]["start"], "2024-10-01T00:00:00Z")
        self.assertEqual(protocol["source"]["end"], "2025-10-01T00:00:00Z")
        self.assertEqual(protocol["source"]["symbols"], list(SYMBOLS))
        self.assertEqual(protocol["source"]["sessions"], list(SESSIONS))
        self.assertEqual(protocol["source"]["feed"], FEED)
        self.assertEqual(protocol["source"]["adjustment"], ADJUSTMENT)
        self.assertEqual(protocol["targets"]["horizons_minutes"], list(HORIZONS_MINUTES))
        self.assertEqual(protocol["targets"]["direction_contract"], "conditional_up_vs_down")
        self.assertEqual(protocol["split"]["train_fraction"], TRAIN_FRACTION)
        self.assertEqual(protocol["split"]["validation_fraction"], VALIDATION_FRACTION)
        self.assertAlmostEqual(protocol["split"]["test_fraction"], 0.15)
        self.assertEqual(protocol["frozen_challenger"]["family"], "HistGradientBoostingClassifier")
        self.assertEqual(protocol["frozen_challenger"]["loss"], "log_loss")
        self.assertEqual(protocol["development_gate"]["expected_fold_count"], 8)
        self.assertEqual(protocol["development_gate"]["minimum_brier_fold_wins_vs_linear"], 6)
        self.assertFalse(protocol["development_gate"]["pnl_used"])
        self.assertFalse(protocol["research_safety"]["validation_may_be_opened_during_development"])
        self.assertFalse(protocol["research_safety"]["test_may_be_opened_during_model_selection"])
        self.assertEqual(len(protocol["protocol_sha256"]), 64)
        self.assertEqual(protocol["protocol_sha256"], preregistered_protocol()["protocol_sha256"])

    def test_preregistration_is_written_before_fetch(self) -> None:
        captured = {}

        def fake_fetcher(output_dir, **kwargs):
            root = Path(output_dir)
            path = root / "v4_direction_preregistration.json"
            self.assertTrue(path.is_file())
            stored = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(stored["protocol_sha256"], preregistered_protocol()["protocol_sha256"])
            captured.update(kwargs)
            return {
                "builder": "ourai-historical-market-v2",
                "dataset_id": "synthetic-v4-direction",
                "manifest_sha256": "abc",
                "source": {"partial": False},
                "source_cache": {"row_count": 2000},
                "dataset": {
                    "sample_count": 1800,
                    "market_date_count": 250,
                    "splits": {
                        "train": {"sample_count": 1260},
                        "validation": {"sample_count": 270},
                        "test": {"sample_count": 270},
                    },
                },
            }

        with tempfile.TemporaryDirectory() as temp_dir:
            summary = build_v4_direction_dataset(temp_dir, dataset_fetcher=fake_fetcher)
            config = captured["market_config"]
            self.assertEqual(config.start, SOURCE_START)
            self.assertEqual(config.end, SOURCE_END)
            self.assertEqual(config.symbols, SYMBOLS)
            self.assertEqual(config.sessions, SESSIONS)
            self.assertEqual(config.feed, FEED)
            self.assertFalse(config.allow_partial)
            self.assertEqual(captured["horizons_minutes"], HORIZONS_MINUTES)
            self.assertEqual(captured["train_fraction"], TRAIN_FRACTION)
            self.assertEqual(captured["validation_fraction"], VALIDATION_FRACTION)
            self.assertEqual(summary["dataset_id"], "synthetic-v4-direction")
            self.assertFalse(summary["overlap_with_v3"])
            self.assertEqual(summary["validation_status"], "UNTOUCHED")
            self.assertEqual(summary["test_status"], "UNTOUCHED")

    def test_partial_source_is_rejected(self) -> None:
        def fake_fetcher(output_dir, **kwargs):
            del output_dir, kwargs
            return {
                "builder": "ourai-historical-market-v2",
                "dataset_id": "partial",
                "manifest_sha256": "partial",
                "source": {"partial": True},
                "source_cache": {},
                "dataset": {},
            }

        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(ValueError, "partial source history"):
                build_v4_direction_dataset(temp_dir, dataset_fetcher=fake_fetcher)


if __name__ == "__main__":
    unittest.main()
