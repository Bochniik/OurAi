from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.diagnose_v5_gru_failure import (
    AuditRow,
    EXPECTED_DATASET_ID,
    EXPECTED_REPORT,
    mark_sequence_eligible,
    nonoverlapping_label_count,
    run_audit,
    summarize_fold,
)


class V5GRUPostmortemTests(unittest.TestCase):
    def test_sequence_eligibility_and_overlapping_tail_proxy(self) -> None:
        rows = [
            AuditRow(
                symbol="NVDA",
                market_date="2024-12-01",
                session="regular",
                minute_utc=2000 + index,
                direction="UP" if index % 2 else "DOWN",
                magnitude_bps=50.0,
            )
            for index in range(75)
        ]
        mark_sequence_eligible(rows, length=60)
        self.assertEqual(sum(row.sequence_eligible for row in rows), 16)
        tail = [row for row in rows if row.sequence_eligible]
        self.assertEqual(nonoverlapping_label_count(tail, horizon_minutes=15), 2)
        fold = {
            "fold": 0,
            "validation_market_dates": ["2024-12-18", "2024-12-19"],
            "actual_tail_threshold_bps": 40.0,
            "gru_training_sequences": 16,
            "gru_epoch_losses": [0.69, 0.45],
            "gru_brier": 0.38,
            "tail_null_brier": 0.25,
            "selected_coverage": 0.20,
            "magnitude_enrichment_ratio": 1.4,
        }
        result = summarize_fold(rows, fold)
        self.assertTrue(result["sequence_count_matches_report"])
        self.assertEqual(result["approx_nonoverlapping_tail_intervals"], 2)
        self.assertAlmostEqual(result["tail_up_fraction"], 0.5)
        self.assertAlmostEqual(result["nonoverlap_fraction_of_tail"], 0.125)
        self.assertAlmostEqual(result["train_loss_drop"], 0.24)

    def test_gaps_and_new_session_reset_eligibility(self) -> None:
        rows = [
            AuditRow("NVDA", "2024-12-01", "regular", 100 + index, "UP", 50)
            for index in range(60)
        ]
        rows.extend([
            AuditRow("NVDA", "2024-12-01", "regular", 200, "UP", 50),
            AuditRow("NVDA", "2024-12-02", "regular", 201, "UP", 50),
        ])
        mark_sequence_eligible(rows)
        self.assertTrue(rows[59].sequence_eligible)
        self.assertFalse(rows[60].sequence_eligible)
        self.assertFalse(rows[61].sequence_eligible)

    def test_global_validation_or_test_flag_fails_before_reading_train(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            data = root / "data"
            data.mkdir()
            (data / "manifest.json").write_text(json.dumps({"dataset_id": EXPECTED_DATASET_ID}), encoding="utf-8")
            for holdout_flag in ("validation_used", "test_lockbox_accessed"):
                report = {
                    "dataset_id": EXPECTED_DATASET_ID,
                    "experiment": EXPECTED_REPORT,
                    "source": "train_splits_only",
                    "summary": {"supported": False},
                    "fold_results": [{}] * 7,
                    "validation_used": False,
                    "test_lockbox_accessed": False,
                    "context_validation_used": False,
                    "context_test_used": False,
                    "context_targets_used": False,
                    "pnl_used": False,
                }
                report[holdout_flag] = True
                path = root / "report.json"
                path.write_text(json.dumps(report), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, holdout_flag):
                    run_audit(data, path, root / "out.json")
                self.assertFalse((root / "out.json").exists())


if __name__ == "__main__":
    unittest.main()
