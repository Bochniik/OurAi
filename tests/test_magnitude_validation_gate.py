from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from scripts.validate_magnitude_head import run_validation_gate


class MagnitudeValidationGateTests(unittest.TestCase):
    def _write_dataset(self, root: Path) -> None:
        manifest = {
            "builder": "ourai-historical-market-v2",
            "dataset_id": "synthetic-v2-validation",
            "configuration": {
                "bar_availability_delay_minutes": 1.0,
            },
        }
        (root / "manifest.json").write_text(
            json.dumps(manifest),
            encoding="utf-8",
        )

        base = datetime(2026, 1, 5, 14, 31, tzinfo=timezone.utc)
        train_rows = []
        for day in range(20):
            for index in range(4):
                timestamp = base + timedelta(days=day, minutes=index * 30)
                signal = float((day + index) % 5)
                magnitude = 0.001 + signal * 0.0008
                sign = -1.0 if (day + index) % 3 == 0 else 1.0
                future_return = sign * magnitude
                train_rows.append(
                    {
                        "symbol": "NVDA" if index % 2 else "AMD",
                        "timestamp": timestamp.isoformat(),
                        "session_date": timestamp.date().isoformat(),
                        "session": "regular",
                        "features": {
                            "price": 100.0 + day,
                            "magnitude_signal": signal,
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

        validation_rows = []
        validation_base = base + timedelta(days=25)
        for day in range(3):
            for index in range(4):
                timestamp = validation_base + timedelta(
                    days=day,
                    minutes=index * 30,
                )
                signal = float((day + index + 1) % 5)
                magnitude = 0.001 + signal * 0.0008
                future_return = magnitude if index % 2 else -magnitude
                validation_rows.append(
                    {
                        "symbol": "NVDA" if index % 2 else "AMD",
                        "timestamp": timestamp.isoformat(),
                        "session_date": timestamp.date().isoformat(),
                        "session": "regular",
                        "features": {
                            "price": 125.0 + day,
                            "magnitude_signal": signal,
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
            for row in train_rows:
                handle.write(json.dumps(row) + "\n")
        with (root / "validation.jsonl").open("w", encoding="utf-8") as handle:
            for row in validation_rows:
                handle.write(json.dumps(row) + "\n")

        # Deliberately invalid: a successful validation run proves the test lockbox
        # was not parsed.
        (root / "test.jsonl").write_text("THIS MUST NOT BE READ", encoding="utf-8")

    def _write_development_report(
        self,
        root: Path,
        *,
        supported: bool = True,
    ) -> Path:
        report = {
            "experiment": "absolute_return_linear_magnitude_v1",
            "dataset_id": "synthetic-v2-validation",
            "dataset_builder": "ourai-historical-market-v2",
            "original_validation_used": False,
            "test_lockbox_accessed": False,
            "promotion_rule": {
                "minimum_fold_wins": 3,
                "required_fold_count": 4,
                "require_lower_mean_mae": True,
                "pnl_used": False,
            },
            "summary": {
                "fold_count": 4,
                "fold_wins": 4 if supported else 2,
                "magnitude_challenger_supported": supported,
            },
        }
        path = root / "development.json"
        path.write_text(json.dumps(report), encoding="utf-8")
        return path

    def test_validation_gate_uses_validation_but_never_test_lockbox(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._write_dataset(root)
            development = self._write_development_report(root)
            output = root / "validation-report.json"

            payload = run_validation_gate(root, development, output)

            self.assertTrue(output.is_file())
            self.assertTrue(payload["original_validation_used"])
            self.assertFalse(payload["test_lockbox_accessed"])
            self.assertEqual(payload["dataset_id"], "synthetic-v2-validation")
            self.assertEqual(payload["horizon_minutes"], 15)
            self.assertEqual(payload["validation"]["market_date_count"], 3)
            self.assertIn("overall", payload["validation"])
            self.assertIn("validation_supported", payload)
            self.assertTrue(
                payload["validation_rule"][
                    "require_strict_majority_market_date_mae_wins"
                ]
            )

    def test_rejects_failed_development_gate_before_parsing_validation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._write_dataset(root)
            development = self._write_development_report(root, supported=False)
            (root / "validation.jsonl").write_text(
                "INVALID VALIDATION MUST NOT BE PARSED",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "did not pass fold-win gate"):
                run_validation_gate(
                    root,
                    development,
                    root / "validation-report.json",
                )


if __name__ == "__main__":
    unittest.main()
