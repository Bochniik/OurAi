from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from scripts.evaluate_magnitude_head import run_experiment


class MagnitudeExperimentTests(unittest.TestCase):
    def _write_dataset(self, root: Path) -> None:
        manifest = {
            "builder": "ourai-historical-market-v2",
            "dataset_id": "synthetic-v2",
            "configuration": {
                "bar_availability_delay_minutes": 1.0,
            },
        }
        (root / "manifest.json").write_text(
            json.dumps(manifest),
            encoding="utf-8",
        )

        rows = []
        base_date = datetime(2026, 1, 5, 14, 31, tzinfo=timezone.utc)
        for day in range(34):
            day_start = base_date + timedelta(days=day)
            for index in range(4):
                timestamp = day_start + timedelta(minutes=index * 30)
                large = (day + index) % 2 == 0
                magnitude = 0.006 if large else 0.0015
                sign = 1.0 if (day + index) % 3 else -1.0
                future_return = sign * magnitude
                rows.append(
                    {
                        "symbol": "NVDA" if index % 2 else "AMD",
                        "timestamp": timestamp.isoformat(),
                        "session_date": timestamp.date().isoformat(),
                        "session": "regular",
                        "features": {
                            "price": 100.0 + day + index * 0.1,
                            "realized_volatility_20": 0.02 if large else 0.004,
                            "relative_volume_20": 1.8 if large else 0.8,
                            "spread_bps": 1.5,
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

        with (root / "train.jsonl").open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")

        # These files are deliberately invalid. A successful experiment proves the
        # train-only runner never tried to parse either held-out split.
        (root / "validation.jsonl").write_text("THIS MUST NOT BE READ", encoding="utf-8")
        (root / "test.jsonl").write_text("THIS MUST NOT BE READ", encoding="utf-8")

    def test_runner_uses_train_only_and_records_preregistered_rule(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._write_dataset(root)
            output = root / "report.json"

            payload = run_experiment(root, output)

            self.assertTrue(output.is_file())
            self.assertEqual(payload["source"], "train_split_only")
            self.assertFalse(payload["original_validation_used"])
            self.assertFalse(payload["test_lockbox_accessed"])
            self.assertEqual(payload["promotion_rule"]["minimum_fold_wins"], 3)
            self.assertEqual(payload["promotion_rule"]["required_fold_count"], 4)
            self.assertTrue(payload["promotion_rule"]["require_lower_mean_mae"])
            self.assertFalse(payload["promotion_rule"]["pnl_used"])
            self.assertEqual(
                payload["null"]["family"],
                "training_fold_median_absolute_return",
            )
            self.assertEqual(payload["summary"]["fold_count"], 4)

    def test_rejects_non_v2_builder(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._write_dataset(root)
            manifest_path = root / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["builder"] = "ourai-historical-market-v1"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "expected builder"):
                run_experiment(root, root / "report.json")


if __name__ == "__main__":
    unittest.main()
