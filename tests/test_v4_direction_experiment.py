from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from scripts.build_v4_direction_dataset import preregistered_protocol
from scripts.evaluate_v4_nonlinear_direction import (
    BASE_EXPECTED_FOLD_COUNT,
    EXPECTED_DATASET_ID,
    EXPECTED_FOLD_COUNT,
    EXPECTED_TRAIN_MARKET_DATES,
    run_experiment,
)


class V4DirectionExperimentTests(unittest.TestCase):
    def _write_dataset(self, root: Path) -> None:
        protocol = preregistered_protocol()
        (root / "v4_direction_preregistration.json").write_text(
            json.dumps(protocol), encoding="utf-8"
        )
        manifest = {
            "builder": "ourai-historical-market-v2",
            "dataset_id": EXPECTED_DATASET_ID,
            "source": {"partial": False},
        }
        (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        result = {
            "protocol_name": protocol["protocol_name"],
            "protocol_sha256": protocol["protocol_sha256"],
            "dataset_id": EXPECTED_DATASET_ID,
            "validation_status": "UNTOUCHED",
            "test_status": "UNTOUCHED",
            "overlap_with_v3": False,
        }
        (root / "v4_direction_protocol_result.json").write_text(
            json.dumps(result), encoding="utf-8"
        )

        start = datetime(2024, 10, 1, 14, 31, tzinfo=timezone.utc)
        with (root / "train.jsonl").open("w", encoding="utf-8") as handle:
            for day in range(EXPECTED_TRAIN_MARKET_DATES):
                day_start = start + timedelta(days=day)
                for index in range(2):
                    timestamp = day_start + timedelta(minutes=index * 30)
                    up = (day + index) % 2 == 1
                    row = {
                        "symbol": "NVDA" if index else "AMD",
                        "timestamp": timestamp.isoformat(),
                        "session_date": timestamp.date().isoformat(),
                        "session": "regular",
                        "features": {
                            "price": 100.0 + day * 0.1 + index,
                            "return_1m": 0.002 if up else -0.002,
                            "realized_volatility_20": 0.012 if day % 3 == 0 else 0.005,
                            "relative_volume_20": 1.6 if up else 0.75,
                            "session_progress": 0.2 + index * 0.1,
                        },
                        "targets": {
                            "15m": {
                                "future_timestamp": (timestamp + timedelta(minutes=15)).isoformat(),
                                "future_return": 0.0015 if up else -0.0015,
                                "direction": "UP" if up else "DOWN",
                                "neutral_threshold_bps": 3.0,
                            }
                        },
                    }
                    handle.write(json.dumps(row) + "\n")

        # A successful run proves neither holdout is parsed during development.
        (root / "validation.jsonl").write_text("MUST NOT BE READ", encoding="utf-8")
        (root / "test.jsonl").write_text("MUST NOT BE READ", encoding="utf-8")

    def test_runner_uses_train_only_and_keeps_holdouts_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._write_dataset(root)
            payload = run_experiment(root, root / "report.json")
            self.assertEqual(payload["source"], "train_split_only")
            self.assertFalse(payload["validation_used"])
            self.assertFalse(payload["test_lockbox_accessed"])
            self.assertEqual(payload["summary"]["fold_count"], EXPECTED_FOLD_COUNT)
            self.assertEqual(
                payload["development_amendment"]["base_expected_fold_count"],
                BASE_EXPECTED_FOLD_COUNT,
            )
            self.assertEqual(
                payload["development_amendment"]["amended_expected_fold_count"],
                EXPECTED_FOLD_COUNT,
            )
            self.assertTrue(
                payload["development_amendment"]["created_before_model_outcomes"]
            )
            self.assertEqual(payload["promotion_rule"]["minimum_brier_fold_wins_vs_linear"], 6)
            self.assertEqual(payload["promotion_rule"]["minimum_brier_fold_wins_vs_null"], 6)
            self.assertFalse(payload["promotion_rule"]["pnl_used"])

    def test_base_protocol_remains_immutable_at_eight_folds(self) -> None:
        protocol = preregistered_protocol()
        self.assertEqual(protocol["development_gate"]["expected_fold_count"], 8)
        self.assertEqual(BASE_EXPECTED_FOLD_COUNT, 8)
        self.assertEqual(EXPECTED_FOLD_COUNT, 7)


if __name__ == "__main__":
    unittest.main()
