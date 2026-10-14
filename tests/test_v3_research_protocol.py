from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.build_v3_research_dataset import (
    ADJUSTMENT,
    FEED,
    HORIZONS_MINUTES,
    PRIOR_V2_SOURCE_START,
    SESSIONS,
    SOURCE_END,
    SOURCE_START,
    SYMBOLS,
    TRAIN_FRACTION,
    VALIDATION_FRACTION,
    build_v3_research_dataset,
    preregistered_protocol,
)


class V3ResearchProtocolTests(unittest.TestCase):
    def test_protocol_is_non_overlapping_and_frozen(self) -> None:
        protocol = preregistered_protocol()

        self.assertLessEqual(SOURCE_END, PRIOR_V2_SOURCE_START)
        self.assertEqual(protocol["source"]["start"], "2025-10-01T00:00:00Z")
        self.assertEqual(protocol["source"]["end"], "2026-08-01T00:00:00Z")
        self.assertEqual(protocol["source"]["symbols"], list(SYMBOLS))
        self.assertEqual(protocol["source"]["sessions"], list(SESSIONS))
        self.assertEqual(protocol["source"]["feed"], FEED)
        self.assertEqual(protocol["source"]["adjustment"], ADJUSTMENT)
        self.assertEqual(protocol["targets"]["horizons_minutes"], list(HORIZONS_MINUTES))
        self.assertEqual(protocol["split"]["train_fraction"], TRAIN_FRACTION)
        self.assertEqual(protocol["split"]["validation_fraction"], VALIDATION_FRACTION)
        self.assertAlmostEqual(protocol["split"]["test_fraction"], 0.15)
        self.assertFalse(
            protocol["research_safety"]["prior_v2_period_may_be_used_for_tuning"]
        )
        self.assertFalse(
            protocol["research_safety"]["new_test_may_be_opened_during_model_selection"]
        )
        self.assertTrue(
            protocol["research_safety"]["test_open_once_after_challenger_frozen"]
        )
        self.assertFalse(
            protocol["research_safety"]["post_test_tuning_on_same_test"]
        )
        self.assertEqual(len(protocol["protocol_sha256"]), 64)
        self.assertEqual(
            protocol["protocol_sha256"], preregistered_protocol()["protocol_sha256"]
        )

    def test_preregistration_is_written_before_fetch_and_contract_is_passed(self) -> None:
        captured = {}

        def fake_fetcher(output_dir, **kwargs):
            root = Path(output_dir)
            preregistration_path = root / "v3_preregistration.json"
            self.assertTrue(preregistration_path.is_file())
            preregistration = json.loads(preregistration_path.read_text(encoding="utf-8"))
            self.assertEqual(
                preregistration["protocol_sha256"],
                preregistered_protocol()["protocol_sha256"],
            )
            captured.update(kwargs)
            return {
                "builder": "ourai-historical-market-v2",
                "dataset_id": "synthetic-v3",
                "manifest_sha256": "abc123",
                "source": {"partial": False},
                "source_cache": {"row_count": 1234},
                "dataset": {
                    "sample_count": 1000,
                    "market_date_count": 100,
                    "splits": {
                        "train": {"sample_count": 700},
                        "validation": {"sample_count": 150},
                        "test": {"sample_count": 150},
                    },
                },
            }

        with tempfile.TemporaryDirectory() as temp_dir:
            summary = build_v3_research_dataset(temp_dir, dataset_fetcher=fake_fetcher)
            root = Path(temp_dir)

            config = captured["market_config"]
            self.assertEqual(config.start, SOURCE_START)
            self.assertEqual(config.end, SOURCE_END)
            self.assertEqual(config.symbols, SYMBOLS)
            self.assertEqual(config.sessions, SESSIONS)
            self.assertEqual(config.feed, FEED)
            self.assertEqual(config.adjustment, ADJUSTMENT)
            self.assertFalse(config.allow_partial)
            self.assertEqual(captured["horizons_minutes"], HORIZONS_MINUTES)
            self.assertEqual(captured["train_fraction"], TRAIN_FRACTION)
            self.assertEqual(captured["validation_fraction"], VALIDATION_FRACTION)

            self.assertEqual(summary["dataset_id"], "synthetic-v3")
            self.assertFalse(summary["prior_v2_overlap"])
            self.assertEqual(summary["new_test_status"], "UNTOUCHED")
            self.assertTrue((root / "v3_protocol_result.json").is_file())

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
                build_v3_research_dataset(temp_dir, dataset_fetcher=fake_fetcher)


if __name__ == "__main__":
    unittest.main()
