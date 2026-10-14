from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone

from trading.nonlinear import (
    HistGradientMagnitudeConfig,
    fit_hist_gradient_magnitude,
    forecast_hist_gradient_magnitudes,
)


class NonlinearMagnitudeTests(unittest.TestCase):
    def _rows(self) -> list[dict]:
        rows = []
        start = datetime(2026, 1, 5, 14, 31, tzinfo=timezone.utc)
        for index in range(240):
            timestamp = start + timedelta(minutes=index)
            regime = index % 4
            magnitude = 0.0008 + (0.0007 if regime in (1, 3) else 0.0001)
            rows.append(
                {
                    "symbol": "NVDA" if index % 2 else "AMD",
                    "timestamp": timestamp.isoformat(),
                    "session": "regular",
                    "features": {
                        "price": 100.0 + index * 0.01,
                        "realized_volatility_20": 0.004 + regime * 0.003,
                        "relative_volume_20": 0.7 + regime * 0.4,
                        "session_progress": (index % 390) / 390.0,
                    },
                    "targets": {
                        "15m": {
                            "future_timestamp": (timestamp + timedelta(minutes=15)).isoformat(),
                            "future_return": magnitude if index % 3 else -magnitude,
                            "direction": "UP" if index % 3 else "DOWN",
                            "neutral_threshold_bps": 3.0,
                        }
                    },
                }
            )
        return rows

    def test_fit_and_forecast_are_non_negative_and_resolved(self) -> None:
        model, summary = fit_hist_gradient_magnitude(
            self._rows(),
            config=HistGradientMagnitudeConfig(
                max_iter=20,
                min_samples_leaf=10,
                minimum_training_samples=100,
            ),
        )
        self.assertEqual(summary.training_sample_count, 240)
        predictions = forecast_hist_gradient_magnitudes(model, self._rows()[:12])
        self.assertEqual(len(predictions), 12)
        self.assertTrue(all(row.expected_magnitude >= 0 for row in predictions))
        self.assertTrue(all(row.actual_magnitude is not None for row in predictions))

    def test_rejects_too_few_training_samples(self) -> None:
        with self.assertRaisesRegex(ValueError, "not enough resolved training samples"):
            fit_hist_gradient_magnitude(
                self._rows()[:30],
                config=HistGradientMagnitudeConfig(minimum_training_samples=100),
            )


if __name__ == "__main__":
    unittest.main()
