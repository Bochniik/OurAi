from __future__ import annotations

import json
import unittest
from datetime import datetime, timedelta, timezone

from trading.baseline import BaselineConfig
from trading.dataset import TradingSample
from trading.magnitude import (
    UnifiedLinearMagnitudeBaseline,
    evaluate_magnitude_predictions,
    fit_unified_linear_magnitude_baseline,
    forecast_magnitudes,
)


class MagnitudeBaselineTests(unittest.TestCase):
    def _samples(self, count: int = 240) -> list[TradingSample]:
        rows: list[TradingSample] = []
        start = datetime(2026, 1, 5, 14, 30, tzinfo=timezone.utc)
        for index in range(count):
            high_vol = index % 2 == 0
            signal = 0.018 if high_vol else 0.003
            magnitude = 0.006 if high_vol else 0.0015
            sign = 1.0 if index % 3 else -1.0
            future_return = sign * magnitude
            direction = "UP" if future_return > 0 else "DOWN"
            rows.append(
                TradingSample(
                    symbol="NVDA" if index % 4 else "AMD",
                    timestamp=start + timedelta(minutes=index),
                    session_date=(start + timedelta(days=index // 80)).date().isoformat(),
                    session="regular",
                    features={
                        "price": 100.0 + index * 0.01,
                        "realized_volatility_20": signal,
                        "relative_volume_20": 1.8 if high_vol else 0.8,
                        "spread_bps": 1.5,
                        "optional_feature": None if index % 5 == 0 else signal,
                    },
                    targets={
                        "15m": {
                            "future_timestamp": (
                                start + timedelta(minutes=index + 15)
                            ).isoformat(),
                            "future_return": future_return,
                            "direction": direction,
                            "neutral_threshold_bps": 3.0,
                        }
                    },
                )
            )
        return rows

    def _config(self) -> BaselineConfig:
        return BaselineConfig(
            horizon_minutes=15,
            regression_l2=0.1,
            minimum_training_samples=40,
            seed=7,
        )

    def test_magnitude_baseline_learns_large_vs_small_moves(self) -> None:
        samples = self._samples()
        model, summary = fit_unified_linear_magnitude_baseline(
            samples,
            config=self._config(),
        )

        self.assertEqual(summary.training_sample_count, 240)
        self.assertGreater(summary.training_median_magnitude_bps, 0)
        high = model.predict_sample(samples[0])
        low = model.predict_sample(samples[1])
        self.assertGreater(high.expected_magnitude_bps, low.expected_magnitude_bps)
        self.assertGreaterEqual(high.expected_magnitude_bps, 0)
        self.assertGreaterEqual(low.expected_magnitude_bps, 0)
        self.assertAlmostEqual(high.actual_magnitude_bps or 0.0, 60.0, places=8)
        self.assertAlmostEqual(low.actual_magnitude_bps or 0.0, 15.0, places=8)

    def test_model_round_trip_is_prediction_stable(self) -> None:
        samples = self._samples()
        model, _ = fit_unified_linear_magnitude_baseline(
            samples,
            config=self._config(),
        )
        restored = UnifiedLinearMagnitudeBaseline.from_dict(
            json.loads(json.dumps(model.to_dict()))
        )
        original = model.predict_sample(samples[0])
        reloaded = restored.predict_sample(samples[0])
        self.assertAlmostEqual(
            original.expected_magnitude,
            reloaded.expected_magnitude,
            places=12,
        )

    def test_forecast_magnitudes_skips_unresolved_rows_for_evaluation(self) -> None:
        samples = self._samples()
        model, _ = fit_unified_linear_magnitude_baseline(
            samples,
            config=self._config(),
        )
        unresolved = TradingSample(
            symbol="AAPL",
            timestamp=datetime(2026, 1, 8, 17, 0, tzinfo=timezone.utc),
            session_date="2026-01-08",
            session="regular",
            features={
                "price": 100.0,
                "realized_volatility_20": 0.01,
                "relative_volume_20": 1.0,
                "spread_bps": 2.0,
                "optional_feature": None,
            },
            targets={},
        )
        evaluated = forecast_magnitudes(model, [samples[0], unresolved])
        self.assertEqual(len(evaluated), 1)
        live = forecast_magnitudes(model, [unresolved], include_actual=False)
        self.assertEqual(len(live), 1)
        self.assertIsNone(live[0].actual_magnitude)
        self.assertGreaterEqual(live[0].expected_magnitude, 0)

    def test_evaluation_uses_fixed_train_only_null(self) -> None:
        samples = self._samples()
        model, summary = fit_unified_linear_magnitude_baseline(
            samples[:160],
            config=self._config(),
        )
        predictions = forecast_magnitudes(model, samples[160:])
        report = evaluate_magnitude_predictions(
            predictions,
            null_magnitude_bps=summary.training_median_magnitude_bps,
        )

        self.assertEqual(report["evaluated_count"], 80)
        self.assertEqual(
            report["null_magnitude_bps"],
            summary.training_median_magnitude_bps,
        )
        self.assertGreaterEqual(report["model_mae_bps"], 0)
        self.assertGreaterEqual(report["null_mae_bps"], 0)
        self.assertEqual(
            report["model_wins_mae"],
            report["model_mae_bps"] < report["null_mae_bps"],
        )

    def test_fit_fails_with_too_few_resolved_targets(self) -> None:
        rows = self._samples(20)
        with self.assertRaisesRegex(ValueError, "not enough resolved training samples"):
            fit_unified_linear_magnitude_baseline(rows, config=self._config())


if __name__ == "__main__":
    unittest.main()
