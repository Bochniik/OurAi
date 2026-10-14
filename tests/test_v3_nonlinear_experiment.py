from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from scripts.build_v3_research_dataset import preregistered_protocol
from scripts.evaluate_v3_nonlinear_magnitude import run_experiment


class V3NonlinearExperimentTests(unittest.TestCase):
    def _write_dataset(self, root: Path) -> None:
        protocol = preregistered_protocol()
        (root / "v3_preregistration.json").write_text(
            json.dumps(protocol), encoding="utf-8"
        )
        manifest = {
            "builder": "ourai-historical-market-v2",
            "dataset_id": "synthetic-v3",
            "source": {"partial": False},
        }
        (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        result = {
            "protocol_name": protocol["protocol_name"],
            "protocol_sha256": protocol["protocol_sha256"],
            "dataset_id": "synthetic-v3",
            "new_test_status": "UNTOUCHED",
            "prior_v2_overlap": False,
        }
        (root / "v3_protocol_result.json").write_text(
            json.dumps(result), encoding="utf-8"
        )

        start = datetime(2025, 10, 1, 14, 31, tzinfo=timezone.utc)
        with (root / "train.jsonl").open("w", encoding="utf-8") as handle:
            for day in range(145):
                day_start = start + timedelta(days=day)
                for index in range(2):
                    timestamp = day_start + timedelta(minutes=index * 30)
                    high_regime = (day + index) % 3 == 0
                    magnitude = 0.0025 if high_regime else 0.0008
                    row = {
                        "symbol": "NVDA" if index else "AMD",
                        "timestamp": timestamp.isoformat(),
                        "session_date": timestamp.date().isoformat(),
                        "session": "regular",
                        "features": {
                            "price": 100.0 + day * 0.1 + index,
                            "realized_volatility_20": 0.02 if high_regime else 0.004,
                            "relative_volume_20": 1.8 if high_regime else 0.8,
                            "session_progress": 0.2 + index * 0.1,
                        },
                        "targets": {
                            "15m": {
                                "future_timestamp": (
                                    timestamp + timedelta(minutes=15)
                                ).isoformat(),
                                "future_return": magnitude if day % 2 else -magnitude,
                                "direction": "UP" if day % 2 else "DOWN",
                                "neutral_threshold_bps": 3.0,
                            }
                        },
                    }
                    handle.write(json.dumps(row) + "\n")

        # A successful run proves the development experiment does not parse either holdout.
        (root / "validation.jsonl").write_text("MUST NOT BE READ", encoding="utf-8")
        (root / "test.jsonl").write_text("MUST NOT BE READ", encoding="utf-8")

    def test_runner_uses_train_only_and_keeps_v3_test_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._write_dataset(root)
            payload = run_experiment(root, root / "report.json")
            self.assertEqual(payload["source"], "train_split_only")
            self.assertFalse(payload["original_validation_used"])
            self.assertFalse(payload["test_lockbox_accessed"])
            self.assertEqual(payload["summary"]["fold_count"], 6)
            self.assertEqual(
                payload["promotion_rule"]["minimum_fold_wins_vs_linear"], 4
            )
            self.assertFalse(payload["promotion_rule"]["pnl_used"])


if __name__ == "__main__":
    unittest.main()
