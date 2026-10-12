from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone

from trading.evaluation import ForecastPrediction
from trading.null_baselines import compare_to_training_prior, training_direction_prior


class TrainingPriorBenchmarkTests(unittest.TestCase):
    def _forecast(self, offset: int, p_up: float, actual_return: float | None) -> ForecastPrediction:
        return ForecastPrediction(
            symbol="NVDA",
            timestamp=datetime(2026, 9, 21, 14, 30, tzinfo=timezone.utc) + timedelta(minutes=offset),
            horizon_minutes=15,
            p_up=p_up,
            expected_return=0.0,
            actual_return=actual_return,
        )

    def test_train_prior_excludes_neutral_and_ignores_other_horizons(self) -> None:
        rows = [
            {"targets": {"15m": {"direction": "UP"}}},
            {"targets": {"15m": {"direction": "UP"}}},
            {"targets": {"15m": {"direction": "UP"}}},
            {"targets": {"15m": {"direction": "DOWN"}}},
            {"targets": {"15m": {"direction": "NEUTRAL"}}},
            {"targets": {"5m": {"direction": "DOWN"}}},
        ]
        prior = training_direction_prior(rows, horizon_minutes=15)
        self.assertEqual((prior.up_count, prior.down_count, prior.neutral_count), (3, 1, 1))
        self.assertEqual(prior.p_up, 0.75)

    def test_null_comparison_uses_only_frozen_train_prior(self) -> None:
        prior = training_direction_prior([
            {"targets": {"15m": {"direction": "UP"}}},
            {"targets": {"15m": {"direction": "UP"}}},
            {"targets": {"15m": {"direction": "UP"}}},
            {"targets": {"15m": {"direction": "DOWN"}}},
        ])
        rows = [
            self._forecast(0, 0.9, 0.005),
            self._forecast(15, 0.1, -0.005),
        ]
        result = compare_to_training_prior(rows, prior=prior)
        self.assertEqual(result["purpose"], "validation_only_null_comparison")
        self.assertEqual(result["train_prior"]["p_up"], 0.75)
        self.assertEqual(result["validation_count"], 2)
        self.assertAlmostEqual(result["null"]["brier_score"], 0.3125)
        self.assertAlmostEqual(result["candidate"]["brier_score"], 0.01)
        self.assertLess(result["candidate_minus_null"]["brier_score"], 0)
        self.assertEqual(result["null"]["direction_accuracy"], 0.5)
        self.assertIsNotNone(result["candidate"]["ece_10_bins"])

    def test_unresolved_and_neutral_validation_rows_are_excluded_from_binary_scores(self) -> None:
        prior = training_direction_prior([{"targets": {"15m": {"direction": "UP"}}}])
        rows = [
            self._forecast(0, 0.9, 0.005),
            self._forecast(15, 0.1, 0.0001),
            self._forecast(30, 0.2, None),
        ]
        result = compare_to_training_prior(rows, prior=prior, neutral_threshold_bps=3.0)
        self.assertEqual(result["validation_count"], 2)
        self.assertEqual(result["directional_count"], 1)

    def test_zero_directional_training_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "directional examples"):
            training_direction_prior([{"targets": {"15m": {"direction": "NEUTRAL"}}}])

    def test_mixed_horizon_fails_closed(self) -> None:
        prior = training_direction_prior([{"targets": {"15m": {"direction": "UP"}}}])
        row = self._forecast(0, 0.9, 0.005)
        other = ForecastPrediction(
            symbol="NVDA",
            timestamp=row.timestamp,
            horizon_minutes=5,
            p_up=0.9,
            expected_return=0.0,
            actual_return=0.005,
        )
        with self.assertRaisesRegex(ValueError, "horizons must match"):
            compare_to_training_prior([row, other], prior=prior)


if __name__ == "__main__":
    unittest.main()
