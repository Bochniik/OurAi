from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from scripts.test_magnitude_lockbox import run_test_lockbox_gate


class MagnitudeTestLockboxTests(unittest.TestCase):
    def _write_split(self, path: Path, *, start_day: int, day_count: int) -> None:
        rows = []
        base = datetime(2026, 1, 5, 14, 31, tzinfo=timezone.utc)
        for day_offset in range(day_count):
            day = start_day + day_offset
            day_start = base + timedelta(days=day)
            for index in range(10):
                timestamp = day_start + timedelta(minutes=index * 20)
                large = (index % 2) == 0
                magnitude = 0.006 if large else 0.001
                sign = 1.0 if (day + index) % 3 else -1.0
                future_return = sign * magnitude
                rows.append(
                    {
                        "symbol": "NVDA" if index % 2 else "AMD",
                        "timestamp": timestamp.isoformat(),
                        "session_date": timestamp.date().isoformat(),
                        "session": "regular",
                        "features": {
                            "magnitude_signal": 1.0 if large else 0.0,
                        },
                        "targets": {
                            "15m": {
                                "future_timestamp": (
                                    timestamp + timedelta(minutes=15)
                                ).isoformat(),
                                "future_return": future_return,
                                "direction": "UP" if future_return > 0 else "DOWN",
                                "neutral_threshold_bps": 3.0,
                            }
                        },
                    }
                )
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")

    def _write_dataset_and_reports(self, root: Path) -> tuple[Path, Path]:
        manifest = {
            "builder": "ourai-historical-market-v2",
            "dataset_id": "synthetic-lockbox-v2",
            "configuration": {"bar_availability_delay_minutes": 1.0},
        }
        (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        self._write_split(root / "train.jsonl", start_day=0, day_count=6)
        self._write_split(root / "validation.jsonl", start_day=6, day_count=3)
        self._write_split(root / "test.jsonl", start_day=9, day_count=3)

        development_path = root / "development.json"
        development = {
            "experiment": "absolute_return_linear_magnitude_v1",
            "dataset_id": manifest["dataset_id"],
            "dataset_builder": manifest["builder"],
            "original_validation_used": False,
            "test_lockbox_accessed": False,
            "promotion_rule": {
                "required_fold_count": 4,
                "minimum_fold_wins": 3,
                "require_lower_mean_mae": True,
            },
            "summary": {
                "fold_count": 4,
                "fold_wins": 4,
                "magnitude_challenger_supported": True,
            },
        }
        development_path.write_text(json.dumps(development), encoding="utf-8")

        validation_path = root / "validation_report.json"
        validation = {
            "experiment": "absolute_return_linear_magnitude_validation_v1",
            "dataset_id": manifest["dataset_id"],
            "dataset_builder": manifest["builder"],
            "development_report": str(development_path),
            "validation_supported": True,
            "original_validation_used": True,
            "test_lockbox_accessed": False,
            "frozen_model": {
                "family": "ridge_absolute_return",
                "regression_l2": 1.0,
                "minimum_training_samples": 40,
                "negative_prediction_policy": "clip_to_zero",
            },
            "validation_rule": {
                "overall_model_mae_must_beat_train_median_null": True,
                "require_strict_majority_market_date_mae_wins": True,
                "tuning_after_validation": False,
                "pnl_used": False,
            },
        }
        validation_path.write_text(json.dumps(validation), encoding="utf-8")
        return development_path, validation_path

    def test_final_gate_uses_train_plus_validation_and_opens_test_once(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            development_path, validation_path = self._write_dataset_and_reports(root)
            output = root / "test_report.json"

            payload = run_test_lockbox_gate(
                root,
                development_path,
                validation_path,
                output,
            )

            self.assertTrue(output.is_file())
            self.assertEqual(payload["final_training_source"], "train_plus_validation")
            self.assertTrue(payload["test_lockbox_accessed"])
            self.assertFalse(payload["post_test_tuning_allowed"])
            self.assertFalse(payload["test_rule"]["pnl_used"])
            self.assertTrue(
                payload["test_rule"]["require_strict_majority_market_date_mae_wins"]
            )
            self.assertTrue(payload["test_supported"])
            self.assertGreater(
                payload["test"]["market_date_win_count"],
                payload["test"]["market_date_count"] / 2,
            )

    def test_failed_validation_gate_stops_before_test_is_parsed(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            development_path, validation_path = self._write_dataset_and_reports(root)
            validation = json.loads(validation_path.read_text(encoding="utf-8"))
            validation["validation_supported"] = False
            validation_path.write_text(json.dumps(validation), encoding="utf-8")
            (root / "test.jsonl").write_text("THIS MUST NOT BE PARSED", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "validation magnitude challenger was not supported"):
                run_test_lockbox_gate(
                    root,
                    development_path,
                    validation_path,
                    root / "should_not_exist.json",
                )


if __name__ == "__main__":
    unittest.main()
