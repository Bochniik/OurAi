from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone

from trading.direction import (
    HistGradientDirectionConfig,
    evaluate_direction_predictions,
    fit_hist_gradient_direction,
    forecast_hist_gradient_directions,
)


class DirectionHeadTests(unittest.TestCase):
    def _samples(self):
        start = datetime(2025, 1, 2, 14, 31, tzinfo=timezone.utc)
        rows = []
        for index in range(160):
            timestamp = start + timedelta(minutes=index)
            up = index % 2 == 1
            neutral = index % 13 == 0
            direction = "NEUTRAL" if neutral else "UP" if up else "DOWN"
            future_return = 0.0 if neutral else 0.0015 if up else -0.0015
            rows.append(
                {
                    "symbol": "NVDA" if index % 3 else "AMD",
                    "timestamp": timestamp.isoformat(),
                    "session": "regular",
                    "features": {
                        "price": 100.0 + index * 0.01,
                        "return_1m": 0.002 if up else -0.002,
                        "relative_volume_20": 1.4 if up else 0.7,
                        "session_progress": min(0.99, index / 390.0),
                    },
                    "targets": {
                        "15m": {
                            "future_return": future_return,
                            "direction": direction,
                        }
                    },
                }
            )
        return rows

    def test_fit_predict_and_probability_metrics(self) -> None:
        config = HistGradientDirectionConfig(
            horizon_minutes=15,
            learning_rate=0.08,
            max_iter=40,
            max_leaf_nodes=7,
            min_samples_leaf=5,
            l2_regularization=0.2,
            max_bins=64,
            random_state=7,
            minimum_training_samples=40,
        )
        model, summary = fit_hist_gradient_direction(self._samples(), config=config)
        self.assertGreater(summary.training_directional_sample_count, 100)
        self.assertGreater(summary.excluded_neutral_sample_count, 0)
        self.assertGreater(summary.training_up_rate, 0.4)
        self.assertLess(summary.training_up_rate, 0.6)

        predictions = forecast_hist_gradient_directions(model, self._samples())
        self.assertTrue(predictions)
        self.assertTrue(all(0.0 <= row.p_up <= 1.0 for row in predictions))
        self.assertTrue(all(row.actual_direction in (0, 1) for row in predictions))

        metrics = evaluate_direction_predictions(
            predictions,
            null_p_up=summary.training_up_rate,
        )
        self.assertEqual(metrics["evaluated_count"], len(predictions))
        self.assertGreaterEqual(metrics["brier_score"], 0.0)
        self.assertGreaterEqual(metrics["log_loss"], 0.0)
        self.assertIsNotNone(metrics["balanced_accuracy"])

    def test_null_probability_must_be_strictly_between_zero_and_one(self) -> None:
        config = HistGradientDirectionConfig(
            max_iter=5,
            min_samples_leaf=5,
            minimum_training_samples=40,
        )
        model, _ = fit_hist_gradient_direction(self._samples(), config=config)
        predictions = forecast_hist_gradient_directions(model, self._samples())
        with self.assertRaises(ValueError):
            evaluate_direction_predictions(predictions, null_p_up=1.0)


if __name__ == "__main__":
    unittest.main()
