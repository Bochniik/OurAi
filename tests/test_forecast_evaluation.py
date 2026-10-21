from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone

from trading.direction import JointReturnResearchEstimate

from trading.evaluation import (
    LONG,
    NO_TRADE,
    SHORT,
    DecisionPolicyConfig,
    ForecastPrediction,
    calibration_table,
    decide_forecast,
    evaluate_forecasts,
)


class ForecastEvaluationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.now = datetime(2026, 1, 5, 14, 30, tzinfo=timezone.utc)

    def _forecast(self, **overrides) -> ForecastPrediction:
        values = {
            "symbol": "NVDA",
            "timestamp": self.now,
            "horizon_minutes": 15,
            "p_up": 0.70,
            "expected_return": 0.005,
            "actual_return": 0.004,
            "spread_bps": 2.0,
        }
        values.update(overrides)
        return ForecastPrediction(**values)

    def test_long_short_and_no_trade_decisions(self) -> None:
        long_decision = decide_forecast(self._forecast(p_up=0.72, expected_return=0.004))
        self.assertEqual(long_decision.action, LONG)
        self.assertGreater(long_decision.expected_net_edge_bps, 0)

        short_decision = decide_forecast(
            self._forecast(p_up=0.20, expected_return=-0.006)
        )
        self.assertEqual(short_decision.action, SHORT)
        self.assertGreater(short_decision.expected_net_edge_bps, 0)

        weak_decision = decide_forecast(
            self._forecast(p_up=0.55, expected_return=0.010)
        )
        self.assertEqual(weak_decision.action, NO_TRADE)
        self.assertEqual(weak_decision.reason, "direction_probability_below_threshold")

    def test_probability_return_disagreement_abstains(self) -> None:
        decision = decide_forecast(
            self._forecast(p_up=0.80, expected_return=-0.010)
        )
        self.assertEqual(decision.action, NO_TRADE)
        self.assertEqual(decision.reason, "probability_return_sign_disagreement")

    def test_expected_move_must_clear_costs_and_safety_edge(self) -> None:
        config = DecisionPolicyConfig(
            min_direction_probability=0.60,
            minimum_edge_bps=3.0,
            slippage_bps=2.0,
            fees_bps=1.0,
            fallback_spread_bps=4.0,
        )
        decision = decide_forecast(
            self._forecast(
                p_up=0.90,
                expected_return=0.0007,
                spread_bps=2.0,
            ),
            config=config,
        )
        self.assertEqual(decision.action, NO_TRADE)
        self.assertEqual(decision.reason, "expected_edge_does_not_clear_costs")

    def test_evaluate_forecasts_reports_probability_return_and_policy_metrics(self) -> None:
        forecasts = [
            self._forecast(
                symbol="NVDA",
                p_up=0.80,
                expected_return=0.008,
                actual_return=0.010,
                spread_bps=2.0,
            ),
            self._forecast(
                symbol="AMD",
                timestamp=self.now + timedelta(minutes=1),
                p_up=0.20,
                expected_return=-0.004,
                actual_return=-0.005,
                spread_bps=2.0,
            ),
            self._forecast(
                symbol="AAPL",
                timestamp=self.now + timedelta(minutes=2),
                p_up=0.55,
                expected_return=0.0001,
                actual_return=0.002,
                spread_bps=1.0,
            ),
            self._forecast(
                symbol="META",
                timestamp=self.now + timedelta(minutes=3),
                p_up=0.75,
                expected_return=0.005,
                actual_return=-0.003,
                spread_bps=2.0,
            ),
        ]

        report = evaluate_forecasts(
            forecasts,
            neutral_threshold_bps=3.0,
            calibration_bins=5,
        )

        self.assertEqual(report["horizon_minutes"], 15)
        self.assertEqual(report["forecast_count"], 4)
        self.assertEqual(report["directional_count"], 4)
        self.assertEqual(report["decision_policy"]["trade_count"], 3)
        self.assertEqual(report["decision_policy"]["no_trade_count"], 1)
        self.assertAlmostEqual(report["decision_policy"]["coverage"], 0.75)
        self.assertAlmostEqual(
            report["decision_policy"]["trade_direction_accuracy"],
            2 / 3,
        )
        self.assertGreater(report["probability"]["direction_accuracy"], 0.5)
        self.assertGreater(report["return"]["mae_bps"], 0)
        self.assertEqual(len(report["probability"]["calibration"]), 5)

    def test_neutral_outcomes_are_excluded_from_direction_metrics(self) -> None:
        forecasts = [
            self._forecast(actual_return=0.0001, p_up=0.90),
            self._forecast(
                timestamp=self.now + timedelta(minutes=1),
                actual_return=0.004,
                p_up=0.80,
            ),
        ]
        report = evaluate_forecasts(forecasts, neutral_threshold_bps=5.0)
        self.assertEqual(report["directional_count"], 1)
        self.assertEqual(report["neutral_count"], 1)

    def test_calibration_table_uses_directional_outcomes_only(self) -> None:
        rows = [
            self._forecast(p_up=0.82, actual_return=0.006),
            self._forecast(
                timestamp=self.now + timedelta(minutes=1),
                p_up=0.78,
                actual_return=-0.005,
            ),
            self._forecast(
                timestamp=self.now + timedelta(minutes=2),
                p_up=0.52,
                actual_return=0.0001,
            ),
        ]
        table = calibration_table(rows, bins=5, neutral_threshold_bps=5.0)
        self.assertEqual(sum(bucket["count"] for bucket in table), 2)

    def test_mixed_horizons_fail_closed(self) -> None:
        rows = [
            self._forecast(horizon_minutes=5),
            self._forecast(
                timestamp=self.now + timedelta(minutes=1),
                horizon_minutes=15,
            ),
        ]
        with self.assertRaisesRegex(ValueError, "exactly one prediction horizon"):
            evaluate_forecasts(rows)

    def test_invalid_probability_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "p_up"):
            self._forecast(p_up=1.2)


class JointReturnResearchEstimateTests(unittest.TestCase):
    def test_conditional_expectation_and_percent_units(self):
        estimate = JointReturnResearchEstimate(p_up=0.6, up_magnitude_bps=30, down_magnitude_bps=10)
        self.assertAlmostEqual(estimate.expected_signed_return_bps, 14.0)
        self.assertAlmostEqual(estimate.expected_signed_return_percent, 0.14)

    def test_asymmetric_magnitudes_can_override_majority_direction(self):
        estimate = JointReturnResearchEstimate(p_up=0.6, up_magnitude_bps=5, down_magnitude_bps=20)
        self.assertAlmostEqual(estimate.expected_signed_return_bps, -5.0)

    def test_invalid_inputs_fail_closed(self):
        for kwargs in ({"p_up": float("nan")}, {"p_up": 1.1}, {"up_magnitude_bps": -1}, {"down_magnitude_bps": float("inf")}, {"horizon_minutes": 5}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                JointReturnResearchEstimate(**({**{"p_up": 0.5, "up_magnitude_bps": 10, "down_magnitude_bps": 10}, **kwargs}))


if __name__ == "__main__":
    unittest.main()
