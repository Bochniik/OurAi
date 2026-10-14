from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from scripts.build_v3_research_dataset import preregistered_protocol
from scripts.validate_v3_nonlinear_magnitude import (
    _expected_nonlinear_config,
    run_validation_gate,
)


class V3NonlinearValidationGateTests(unittest.TestCase):
    def _write_contract(self, root: Path) -> dict:
        protocol = preregistered_protocol()
        dataset_id = "synthetic-v3"
        (root / "manifest.json").write_text(
            json.dumps(
                {
                    "builder": "ourai-historical-market-v2",
                    "dataset_id": dataset_id,
                    "source": {"partial": False},
                }
            ),
            encoding="utf-8",
        )
        (root / "v3_preregistration.json").write_text(
            json.dumps(protocol), encoding="utf-8"
        )
        (root / "v3_protocol_result.json").write_text(
            json.dumps(
                {
                    "protocol_name": protocol["protocol_name"],
                    "protocol_sha256": protocol["protocol_sha256"],
                    "dataset_id": dataset_id,
                    "new_test_status": "UNTOUCHED",
                    "prior_v2_overlap": False,
                }
            ),
            encoding="utf-8",
        )
        return protocol

    def _development_report(self, protocol: dict, *, supported: bool = True) -> dict:
        return {
            "experiment": "v3_hist_gradient_absolute_return_magnitude_v1",
            "dataset_id": "synthetic-v3",
            "protocol_name": protocol["protocol_name"],
            "protocol_sha256": protocol["protocol_sha256"],
            "original_validation_used": False,
            "test_lockbox_accessed": False,
            "nonlinear_model": {
                "family": "HistGradientBoostingRegressor",
                "loss": "absolute_error",
                "config": _expected_nonlinear_config(),
            },
            "incumbent": {
                "family": "ridge_absolute_return",
                "regression_l2": 1.0,
            },
            "walk_forward": {"expected_fold_count": 6},
            "promotion_rule": {
                "minimum_fold_wins_vs_linear": 4,
                "minimum_fold_wins_vs_null": 4,
                "require_lower_mean_mae_than_linear": True,
                "require_lower_mean_mae_than_null": True,
                "validation_split_used_for_tuning": False,
                "pnl_used": False,
            },
            "summary": {
                "fold_count": 6,
                "fold_wins_vs_linear": 6 if supported else 0,
                "fold_wins_vs_null": 6 if supported else 0,
                "mean_nonlinear_mae_bps": 10.0 if supported else 13.0,
                "mean_linear_mae_bps": 11.0,
                "mean_null_mae_bps": 12.0,
                "nonlinear_challenger_supported": supported,
            },
        }

    def _write_samples(self, path: Path, *, count: int, start: datetime) -> None:
        rows = []
        for index in range(count):
            timestamp = start + timedelta(minutes=index)
            x = ((index % 31) - 15) / 15.0
            future_return = 0.0005 + 0.0015 * (x * x)
            rows.append(
                {
                    "symbol": "AAPL",
                    "timestamp": timestamp.isoformat(),
                    "session_date": timestamp.date().isoformat(),
                    "session": "regular",
                    "features": {"signal": x},
                    "targets": {
                        "15m": {
                            "future_return": future_return,
                            "future_timestamp": (timestamp + timedelta(minutes=15)).isoformat(),
                        }
                    },
                }
            )
        path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )

    def test_validation_succeeds_without_reading_invalid_test_lockbox(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            protocol = self._write_contract(root)
            self._write_samples(
                root / "train.jsonl",
                count=180,
                start=datetime(2025, 10, 1, 14, 0, tzinfo=timezone.utc),
            )
            self._write_samples(
                root / "validation.jsonl",
                count=40,
                start=datetime(2026, 5, 1, 14, 0, tzinfo=timezone.utc),
            )
            (root / "test.jsonl").write_text("THIS IS DELIBERATELY INVALID\n", encoding="utf-8")
            development = root / "development.json"
            development.write_text(
                json.dumps(self._development_report(protocol)), encoding="utf-8"
            )

            report = run_validation_gate(root, development, root / "validation_report.json")
            self.assertTrue(report["original_validation_used"])
            self.assertFalse(report["test_lockbox_accessed"])
            self.assertEqual(report["validation"]["overall"]["evaluated_count"], 40)

    def test_failed_development_gate_stops_before_validation_is_parsed(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            protocol = self._write_contract(root)
            self._write_samples(
                root / "train.jsonl",
                count=180,
                start=datetime(2025, 10, 1, 14, 0, tzinfo=timezone.utc),
            )
            (root / "validation.jsonl").write_text("INVALID VALIDATION\n", encoding="utf-8")
            development = root / "development.json"
            development.write_text(
                json.dumps(self._development_report(protocol, supported=False)),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "fold-win gate vs linear"):
                run_validation_gate(root, development, root / "validation_report.json")


if __name__ == "__main__":
    unittest.main()
