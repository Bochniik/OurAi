from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone

from trading.diagnostics import validation_slice_diagnostics
from trading.evaluation import ForecastPrediction


class ValidationSliceDiagnosticsTests(unittest.TestCase):
    def _rows(self) -> list[ForecastPrediction]:
        return [
            ForecastPrediction(
                symbol="NVDA",
                timestamp=datetime(2026, 1, 5, 14, 35, tzinfo=timezone.utc),
                horizon_minutes=15,
                p_up=0.80,
                expected_return=0.004,
                actual_return=0.005,
                spread_bps=2.0,
            ),
            ForecastPrediction(
                symbol="AMD",
                timestamp=datetime(2026, 3, 9, 13, 35, tzinfo=timezone.utc),
                horizon_minutes=15,
                p_up=0.20,
                expected_return=-0.004,
                actual_return=-0.005,
                spread_bps=2.0,
            ),
            ForecastPrediction(
                symbol="AMD",
                timestamp=datetime(2026, 3, 9, 13, 50, tzinfo=timezone.utc),
                horizon_minutes=15,
                p_up=0.55,
                expected_return=0.0001,
                actual_return=0.0001,
            ),
            ForecastPrediction(
                symbol="NVDA",
                timestamp=datetime(2026, 3, 9, 14, 5, tzinfo=timezone.utc),
                horizon_minutes=15,
                p_up=0.99,
                expected_return=0.004,
                actual_return=-0.006,
                spread_bps=2.0,
            ),
        ]

    def test_deterministic_slices_and_dst_aware_new_york_buckets(self) -> None:
        rows = self._rows()
        report = validation_slice_diagnostics(rows)
        reversed_report = validation_slice_diagnostics(reversed(rows))
        self.assertEqual(report, reversed_report)
        self.assertEqual(report["purpose"], "validation_diagnostics_only")
        self.assertEqual(report["horizon_minutes"], 15)
        self.assertEqual(report["overall"]["sample_count"], 4)
        self.assertEqual(report["overall"]["neutral_count"], 1)
        self.assertEqual(list(report["by_symbol"]), ["AMD", "NVDA"])
        self.assertEqual(list(report["by_market_date"]), ["2026-01-05", "2026-03-09"])
        self.assertEqual(report["by_ny_half_hour"]["09:30"]["sample_count"], 3)
        self.assertEqual(report["by_ny_half_hour"]["10:00"]["sample_count"], 1)
        self.assertEqual(report["by_p_up_bin"]["0.9-1.0"]["sample_count"], 1)
        self.assertEqual(report["by_confidence_bin"]["0.9-1.0"]["sample_count"], 1)
        self.assertEqual(
            sum(item["sample_count"] for item in report["by_symbol"].values()), 4
        )
        self.assertEqual(report["overall"]["trade_count"], 3)

    def test_neutral_only_slice_has_null_direction_metrics(self) -> None:
        report = validation_slice_diagnostics([self._rows()[2]])
        self.assertEqual(report["overall"]["directional_count"], 0)
        self.assertIsNone(report["overall"]["brier_score"])
        self.assertIsNone(report["overall"]["balanced_accuracy"])

    def test_unresolved_rows_are_excluded(self) -> None:
        rows = self._rows()
        rows.append(
            ForecastPrediction(
                symbol="MSFT",
                timestamp=datetime(2026, 3, 9, 15, 0, tzinfo=timezone.utc),
                horizon_minutes=15,
                p_up=0.5,
                expected_return=0.0,
            )
        )
        report = validation_slice_diagnostics(rows)
        self.assertEqual(report["overall"]["sample_count"], 4)
        self.assertNotIn("MSFT", report["by_symbol"])

    def test_mixed_horizons_fail_closed(self) -> None:
        rows = self._rows()
        rows.append(
            ForecastPrediction(
                symbol="AMD",
                timestamp=rows[0].timestamp + timedelta(minutes=5),
                horizon_minutes=5,
                p_up=0.6,
                expected_return=0.002,
                actual_return=0.002,
            )
        )
        with self.assertRaisesRegex(ValueError, "exactly one prediction horizon"):
            validation_slice_diagnostics(rows)

    def test_requires_resolved_rows(self) -> None:
        with self.assertRaisesRegex(ValueError, "resolved forecasts"):
            validation_slice_diagnostics([])


if __name__ == "__main__":
    unittest.main()
