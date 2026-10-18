"""Small synthetic TRAIN-only tests for purged joint-return OOF generation."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
import unittest
from unittest.mock import patch

from trading.baseline import BaselineConfig, UnifiedLinearBaseline
from trading.dataset import TradingSample
from trading.joint_oof import JointOOFConfig, generate_joint_train_oof
from trading.joint_return_calibration import (
    JointCalibrationConfig, fit_joint_return_calibrator,
)
from trading.magnitude import UnifiedLinearMagnitudeBaseline


class JointOOFTests(unittest.TestCase):
    def setUp(self):
        self.oof_config = JointOOFConfig(
            min_train_market_dates=2,
            validation_market_dates=2,
            step_market_dates=2,
            embargo_minutes=60.0,
        )
        self.linear = BaselineConfig(
            horizon_minutes=15,
            probability_epochs=4,
            probability_batch_size=128,
            minimum_training_samples=40,
        )

    def _rows(self):
        rows = []
        start = datetime(2026, 1, 5, 14, 30, tzinfo=timezone.utc)
        for day in range(6):
            day_start = start + timedelta(days=day)
            for minute in range(60):
                stamp = day_start + timedelta(minutes=minute)
                for symbol_index, symbol in enumerate(("NVDA", "AMD")):
                    up = (day + minute + symbol_index) % 3 != 0
                    absolute = 0.001 + 0.0001 * (minute % 5)
                    rows.append(TradingSample(
                        symbol=symbol,
                        timestamp=stamp,
                        session_date=day_start.date().isoformat(),
                        session="regular",
                        features={
                            "price": 100.0 + day + minute / 100,
                            "volume_ratio": 1.0 + (minute % 6) / 10,
                            "momentum": 0.001 if up else -0.001,
                        },
                        targets={"15m": {
                            "future_timestamp": (stamp + timedelta(minutes=15)).isoformat(),
                            "future_return": absolute if up else -absolute,
                            "direction": "UP" if up else "DOWN",
                        }},
                    ))
        return rows

    def _generate(self, rows=None):
        return generate_joint_train_oof(
            self._rows() if rows is None else rows,
            source_split="train",
            config=self.oof_config,
            direction_config=self.linear,
            magnitude_config=self.linear,
        )

    def test_ooof_rows_are_nonoverlap_and_fold_history_is_purged(self):
        result = self._generate()
        self.assertEqual(result.manifest["source_split"], "train")
        self.assertEqual(result.manifest["oof_observation_count"], 32)
        self.assertEqual(result.manifest["walk_forward"]["fold_count"], 2)
        self.assertTrue(result.observations)
        self.assertEqual(len(result.observations), len({
            (row.symbol, row.timestamp) for row in result.observations
        }))
        for row in result.observations:
            self.assertTrue(row.is_train_oof)
            self.assertEqual(int(row.timestamp.timestamp()) % 900, 0)
            self.assertGreaterEqual(row.predicted_abs_return, 0)
            self.assertGreaterEqual(row.raw_p_up, 0)
            self.assertLessEqual(row.raw_p_up, 1)
        self.assertEqual(
            sum(fold["selected_nonoverlap_oof_rows"]
                for fold in result.manifest["fold_summaries"]),
            len(result.observations),
        )

    def test_predict_called_only_with_feature_only_target_free_rows(self):
        d_predict = UnifiedLinearBaseline.predict_sample
        m_predict = UnifiedLinearMagnitudeBaseline.predict_sample
        calls = []

        def guarded_direction(model, sample, *, include_actual=True):
            self.assertFalse(include_actual)
            self.assertEqual(sample["targets"], {})
            calls.append("direction")
            return d_predict(model, sample, include_actual=include_actual)

        def guarded_magnitude(model, sample, *, include_actual=True):
            self.assertFalse(include_actual)
            self.assertEqual(sample["targets"], {})
            calls.append("magnitude")
            return m_predict(model, sample, include_actual=include_actual)

        with patch.object(UnifiedLinearBaseline, "predict_sample", guarded_direction):
            with patch.object(UnifiedLinearMagnitudeBaseline, "predict_sample", guarded_magnitude):
                result = self._generate()
        self.assertEqual(len(calls), 2 * len(result.observations))

    def test_train_oof_can_feed_joint_calibration_without_claiming_skill(self):
        result = self._generate()
        model = fit_joint_return_calibrator(
            result.observations,
            config=JointCalibrationConfig(minimum_cell_support=2),
        )
        row = result.observations[0]
        forecast = model.predict(
            raw_p_up=row.raw_p_up,
            predicted_abs_return=row.predicted_abs_return,
            horizon_minutes=15,
        )
        self.assertIsNotNone(forecast)
        self.assertGreaterEqual(forecast.supporting_observations, 2)
        self.assertAlmostEqual(forecast.p_up + forecast.p_down + forecast.p_neutral, 1)

    def test_rejects_non_train_split_and_mismatched_horizon(self):
        with self.assertRaisesRegex(ValueError, "train split"):
            generate_joint_train_oof(self._rows(), source_split="validation")
        with self.assertRaisesRegex(ValueError, "horizons must match"):
            generate_joint_train_oof(
                self._rows(), source_split="train",
                config=self.oof_config,
                direction_config=BaselineConfig(horizon_minutes=5),
                magnitude_config=self.linear,
            )
        with self.assertRaisesRegex(ValueError, "step_market_dates"):
            JointOOFConfig(validation_market_dates=3, step_market_dates=1)

    def test_rejects_missing_future_timestamp_and_duplicates(self):
        rows = self._rows()
        broken = replace(rows[0], targets={"15m": {
            "future_return": 0.001,
            "direction": "UP",
        }})
        with self.assertRaisesRegex(ValueError, "future_timestamp"):
            self._generate([broken, *rows[1:]])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self._generate([*rows, rows[0]])

    def test_rejects_noncausal_or_invalid_label_end(self):
        rows = self._rows()
        broken = replace(rows[0], targets={"15m": {
            "future_timestamp": (rows[0].timestamp + timedelta(minutes=30)).isoformat(),
            "future_return": 0.001,
            "direction": "UP",
        }})
        with self.assertRaisesRegex(ValueError, "exact forecast horizon"):
            self._generate([broken, *rows[1:]])


if __name__ == "__main__":
    unittest.main()
