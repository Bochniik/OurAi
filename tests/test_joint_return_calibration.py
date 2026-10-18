"""Contract tests for the experimental, train-only joint-return calibration layer."""

from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone

from trading.joint_return_calibration import (
    JointCalibrationConfig,
    OOFReturnObservation,
    evaluate_joint_return_distributions,
    fit_joint_return_calibrator,
)


class JointReturnCalibrationTests(unittest.TestCase):
    def _rows(self, count: int = 80):
        start = datetime(2026, 1, 5, 14, 30, tzinfo=timezone.utc)
        rows = []
        for index in range(count):
            # An outcome-independent 15-minute schedule with unique anchors.
            timestamp = start + timedelta(minutes=index * 15)
            realized = 0.004 if index % 8 < 3 else -0.002 if index % 8 < 6 else 0.0
            rows.append(OOFReturnObservation(
                symbol="NVDA",
                timestamp=timestamp,
                horizon_minutes=15,
                raw_p_up=0.72,
                predicted_abs_return=0.003,
                realized_return=realized,
                is_train_oof=True,
            ))
        return rows

    def test_joint_outputs_share_one_empirical_distribution(self):
        rows = self._rows()
        # Non-anchor must not inflate evidence or change estimated moments.
        extra = OOFReturnObservation(
            symbol="NVDA",
            timestamp=rows[-1].timestamp + timedelta(minutes=1),
            horizon_minutes=15,
            raw_p_up=0.72,
            predicted_abs_return=0.003,
            realized_return=0.4,
            is_train_oof=True,
        )
        model = fit_joint_return_calibrator(rows + [extra])
        pred = model.predict(raw_p_up=0.72, predicted_abs_return=0.003,
                             horizon_minutes=15)
        self.assertIsNotNone(pred)
        self.assertEqual(model.included_anchor_count, 80)
        self.assertEqual(model.skipped_nonanchor_count, 1)
        self.assertEqual(pred.supporting_observations, 80)
        self.assertAlmostEqual(pred.p_up, 0.375)
        self.assertAlmostEqual(pred.p_down, 0.375)
        self.assertAlmostEqual(pred.p_neutral, 0.25)
        self.assertAlmostEqual(pred.p_up + pred.p_down + pred.p_neutral, 1)
        self.assertAlmostEqual(pred.expected_signed_return, 0.00075)
        self.assertAlmostEqual(pred.expected_abs_return, 0.00225)
        self.assertAlmostEqual(pred.q10_return, -0.002)
        self.assertAlmostEqual(pred.q90_return, 0.004)
        self.assertLessEqual(abs(pred.expected_signed_return), pred.expected_abs_return)
        prices = pred.price_estimates(100)
        self.assertAlmostEqual(prices["expected_price"], 100.075)
        self.assertAlmostEqual(prices["q10_price"], 99.8)
        self.assertAlmostEqual(prices["q90_price"], 100.4)
        with self.assertRaises(ValueError):
            pred.price_estimates(-1)

    def test_abstains_in_unsupported_bin_and_rejects_horizon_mismatch(self):
        model = fit_joint_return_calibrator(self._rows())
        self.assertIsNone(model.predict(raw_p_up=0.55, predicted_abs_return=0.0005,
                                        horizon_minutes=15))
        with self.assertRaises(ValueError):
            model.predict(raw_p_up=0.72, predicted_abs_return=0.003,
                          horizon_minutes=30)

    def test_fit_fails_closed_for_non_oof_and_wrong_horizon(self):
        with self.assertRaisesRegex(ValueError, "out-of-fold"):
            OOFReturnObservation(
                symbol="NVDA",
                timestamp=datetime.now(timezone.utc),
                horizon_minutes=15, raw_p_up=0.6,
                predicted_abs_return=0.002, realized_return=0.01,
            )
        rows = self._rows()
        wrong = OOFReturnObservation(
            symbol="NVDA", timestamp=rows[-1].timestamp + timedelta(minutes=15),
            horizon_minutes=30, raw_p_up=0.72,
            predicted_abs_return=0.003, realized_return=0.002,
            is_train_oof=True,
        )
        with self.assertRaisesRegex(ValueError, "mixed calibration horizons"):
            fit_joint_return_calibrator(rows + [wrong])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            fit_joint_return_calibrator(rows + [rows[0]])

    def test_sparse_cells_refuse_to_fit(self):
        with self.assertRaisesRegex(ValueError, "no calibration cell"):
            fit_joint_return_calibrator(self._rows(12))
        with self.assertRaises(ValueError):
            JointCalibrationConfig(neutral_band_bps=-1)

    def test_metrics_are_evaluation_only(self):
        model = fit_joint_return_calibrator(self._rows())
        forecast = model.predict(raw_p_up=0.72, predicted_abs_return=0.003,
                                 horizon_minutes=15)
        report = evaluate_joint_return_distributions([
            (forecast, 0.004),
            (forecast, -0.002),
            (forecast, 0.0),
        ])
        self.assertEqual(report["evaluated_count"], 3)
        self.assertGreater(report["signed_return_mae_bps"], 0)
        self.assertGreater(report["absolute_magnitude_mae_bps"], 0)
        self.assertGreaterEqual(report["three_class_brier"], 0)
        self.assertAlmostEqual(report["q10_q90_empirical_coverage"], 1.0)
        with self.assertRaises(ValueError):
            evaluate_joint_return_distributions([])


if __name__ == "__main__":
    unittest.main()
