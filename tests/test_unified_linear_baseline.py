from __future__ import annotations

import json
import unittest
from datetime import datetime, timedelta, timezone

from trading.baseline import (
    BaselineConfig,
    UnifiedLinearBaseline,
    fit_unified_linear_baseline,
    forecast_samples,
)
from trading.dataset import TradingSample


class UnifiedLinearBaselineTests(unittest.TestCase):
    def _samples(self, count: int = 240) -> list[TradingSample]:
        rows: list[TradingSample] = []
        start = datetime(2026, 1, 5, 14, 30, tzinfo=timezone.utc)
        for index in range(count):
            bullish = index % 2 == 0
            signal = 0.012 if bullish else -0.012
            secondary = ((index % 7) - 3) / 100.0
            future_return = (0.0045 if bullish else -0.0040) + secondary * 0.01
            direction = "UP" if future_return > 0 else "DOWN"
            features = {
                "price": 100.0 + index * 0.01,
                "return_5m": signal,
                "relative_volume_20": 1.8 if bullish else 0.7,
                "spread_bps": 1.5 + (index % 3) * 0.2,
                "optional_feature": None if index % 5 == 0 else secondary,
            }
            rows.append(
                TradingSample(
                    symbol="NVDA" if index % 3 else "AMD",
                    timestamp=start + timedelta(minutes=index),
                    session_date=(start + timedelta(days=index // 80)).date().isoformat(),
                    session="regular",
                    features=features,
                    targets={
                        "15m": {
                            "future_timestamp": (start + timedelta(minutes=index + 15)).isoformat(),
                            "future_return": future_return,
                            "future_return_pct": future_return * 100.0,
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
            probability_epochs=80,
            probability_learning_rate=0.08,
            probability_batch_size=64,
            minimum_training_samples=40,
            regression_l2=0.1,
            seed=7,
        )

    def test_shared_baseline_learns_direction_and_return_sign(self) -> None:
        model, summary = fit_unified_linear_baseline(
            self._samples(),
            config=self._config(),
        )

        self.assertEqual(summary.training_sample_count, 240)
        self.assertEqual(summary.directional_sample_count, 240)
        self.assertIn("NVDA", summary.symbols)
        self.assertIn("AMD", summary.symbols)
        self.assertIn("num__return_5m", model.schema.expanded_feature_names)
        self.assertIn("missing__optional_feature", model.schema.expanded_feature_names)

        bullish = TradingSample(
            symbol="NVDA",
            timestamp=datetime(2026, 1, 8, 15, 0, tzinfo=timezone.utc),
            session_date="2026-01-08",
            session="regular",
            features={
                "price": 103.0,
                "return_5m": 0.012,
                "relative_volume_20": 1.8,
                "spread_bps": 1.5,
                "optional_feature": None,
            },
            targets={
                "15m": {
                    "future_return": 0.004,
                    "direction": "UP",
                }
            },
        )
        bearish = TradingSample(
            symbol="AMD",
            timestamp=datetime(2026, 1, 8, 15, 1, tzinfo=timezone.utc),
            session_date="2026-01-08",
            session="regular",
            features={
                "price": 103.0,
                "return_5m": -0.012,
                "relative_volume_20": 0.7,
                "spread_bps": 1.5,
                "optional_feature": 0.0,
            },
            targets={
                "15m": {
                    "future_return": -0.004,
                    "direction": "DOWN",
                }
            },
        )

        bullish_forecast = model.predict_sample(bullish)
        bearish_forecast = model.predict_sample(bearish)
        self.assertGreater(bullish_forecast.p_up, 0.75)
        self.assertLess(bearish_forecast.p_up, 0.25)
        self.assertGreater(bullish_forecast.expected_return, 0)
        self.assertLess(bearish_forecast.expected_return, 0)

    def test_model_round_trip_is_prediction_stable(self) -> None:
        samples = self._samples()
        model, _ = fit_unified_linear_baseline(samples, config=self._config())
        restored = UnifiedLinearBaseline.from_dict(
            json.loads(json.dumps(model.to_dict()))
        )

        original = model.predict_sample(samples[0])
        reloaded = restored.predict_sample(samples[0])
        self.assertAlmostEqual(original.p_up, reloaded.p_up, places=12)
        self.assertAlmostEqual(
            original.expected_return,
            reloaded.expected_return,
            places=12,
        )

    def test_unseen_symbol_uses_shared_features_without_failure(self) -> None:
        model, _ = fit_unified_linear_baseline(
            self._samples(),
            config=self._config(),
        )
        sample = TradingSample(
            symbol="AAPL",
            timestamp=datetime(2026, 1, 8, 16, 0, tzinfo=timezone.utc),
            session_date="2026-01-08",
            session="regular",
            features={
                "price": 200.0,
                "return_5m": 0.012,
                "relative_volume_20": 1.8,
                "spread_bps": 1.0,
                "optional_feature": 0.0,
            },
            targets={
                "15m": {
                    "future_return": 0.003,
                    "direction": "UP",
                }
            },
        )
        forecast = model.predict_sample(sample)
        self.assertEqual(forecast.symbol, "AAPL")
        self.assertGreater(forecast.p_up, 0.5)

    def test_forecast_samples_skips_unresolved_rows_when_actuals_requested(self) -> None:
        samples = self._samples()
        model, _ = fit_unified_linear_baseline(samples, config=self._config())
        unresolved = TradingSample(
            symbol="NVDA",
            timestamp=datetime(2026, 1, 8, 17, 0, tzinfo=timezone.utc),
            session_date="2026-01-08",
            session="regular",
            features={
                "price": 100.0,
                "return_5m": 0.01,
                "relative_volume_20": 1.0,
                "spread_bps": 2.0,
                "optional_feature": None,
            },
            targets={},
        )
        forecasts = forecast_samples(model, [samples[0], unresolved])
        self.assertEqual(len(forecasts), 1)
        live_forecasts = forecast_samples(
            model,
            [unresolved],
            include_actual=False,
        )
        self.assertEqual(len(live_forecasts), 1)
        self.assertIsNone(live_forecasts[0].actual_return)

    def test_fit_fails_when_one_direction_is_missing(self) -> None:
        rows = self._samples(60)
        only_up = [
            TradingSample(
                symbol=row.symbol,
                timestamp=row.timestamp,
                session_date=row.session_date,
                session=row.session,
                features=row.features,
                targets={
                    "15m": {
                        "future_return": abs(row.targets["15m"]["future_return"]),
                        "direction": "UP",
                    }
                },
            )
            for row in rows
        ]
        with self.assertRaisesRegex(ValueError, "both UP and DOWN"):
            fit_unified_linear_baseline(only_up, config=self._config())


if __name__ == "__main__":
    unittest.main()
