from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone

from trading.context import (
    ContextObservation,
    ContextSourcePolicy,
    join_point_in_time_context,
)
from trading.dataset import TradingSample


class PointInTimeContextTests(unittest.TestCase):
    def _sample(self, *, symbol: str = "NVDA", minute: int = 10) -> TradingSample:
        timestamp = datetime(2026, 1, 5, 15, minute, tzinfo=timezone.utc)
        return TradingSample(
            symbol=symbol,
            timestamp=timestamp,
            session_date="2026-01-05",
            session="regular",
            features={"price": 100.0},
            targets={"5m": {"direction": "UP", "future_return": 0.01}},
        )

    def test_future_available_context_is_never_joined(self) -> None:
        sample = self._sample(minute=10)
        observations = [
            ContextObservation(
                source="news",
                event_time=sample.timestamp - timedelta(minutes=20),
                available_at=sample.timestamp + timedelta(minutes=1),
                scope="symbol",
                entity="NVDA",
                features={"sentiment": 0.9},
                observation_id="future-publication",
            )
        ]

        result = join_point_in_time_context(
            [sample], observations, expected_sources=("news",)
        )
        enriched = result.samples[0]
        self.assertEqual(enriched.features["ctx__news__available"], 0.0)
        self.assertIsNone(enriched.features["ctx__news__sentiment"])
        self.assertEqual(result.audit[0]["sources"]["news"], [])

    def test_event_time_does_not_override_availability_time(self) -> None:
        sample = self._sample(minute=10)
        observations = [
            ContextObservation(
                source="news",
                event_time=sample.timestamp - timedelta(minutes=30),
                available_at=sample.timestamp - timedelta(minutes=2),
                scope="symbol",
                entity="NVDA",
                features={"sentiment": 0.4},
            )
        ]
        result = join_point_in_time_context([sample], observations)
        enriched = result.samples[0]
        self.assertEqual(enriched.features["ctx__news__sentiment"], 0.4)
        self.assertEqual(enriched.features["ctx__news__age_minutes"], 2.0)

    def test_symbol_scope_overrides_global_feature_from_same_source(self) -> None:
        sample = self._sample()
        observations = [
            ContextObservation(
                source="regime",
                available_at=sample.timestamp - timedelta(minutes=5),
                scope="global",
                features={"risk_on": 0.2, "volatility": 0.7},
                observation_id="global",
            ),
            ContextObservation(
                source="regime",
                available_at=sample.timestamp - timedelta(minutes=6),
                scope="symbol",
                entity="NVDA",
                features={"risk_on": 0.8},
                observation_id="symbol",
            ),
        ]

        result = join_point_in_time_context([sample], observations)
        enriched = result.samples[0]
        self.assertEqual(enriched.features["ctx__regime__risk_on"], 0.8)
        self.assertEqual(enriched.features["ctx__regime__volatility"], 0.7)
        self.assertEqual(len(result.audit[0]["sources"]["regime"]), 2)

    def test_sector_context_requires_explicit_point_in_time_symbol_mapping(self) -> None:
        sample = self._sample(symbol="NVDA")
        observations = [
            ContextObservation(
                source="sector_regime",
                available_at=sample.timestamp - timedelta(minutes=1),
                scope="sector",
                entity="SEMICONDUCTORS",
                features={"relative_strength": 0.6},
            )
        ]

        without_mapping = join_point_in_time_context([sample], observations)
        self.assertEqual(
            without_mapping.samples[0].features["ctx__sector_regime__available"],
            0.0,
        )

        with_mapping = join_point_in_time_context(
            [sample], observations, symbol_sectors={"NVDA": "Semiconductors"}
        )
        self.assertEqual(
            with_mapping.samples[0].features["ctx__sector_regime__relative_strength"],
            0.6,
        )

    def test_freshness_policy_and_valid_until_fail_closed(self) -> None:
        sample = self._sample()
        observations = [
            ContextObservation(
                source="liquidity",
                available_at=sample.timestamp - timedelta(minutes=20),
                scope="symbol",
                entity="NVDA",
                features={"spread_score": 0.3},
            ),
            ContextObservation(
                source="integrity",
                available_at=sample.timestamp - timedelta(minutes=3),
                valid_until=sample.timestamp - timedelta(minutes=1),
                scope="symbol",
                entity="NVDA",
                features={"anomaly_score": 0.9},
            ),
        ]

        result = join_point_in_time_context(
            [sample],
            observations,
            policies={"liquidity": ContextSourcePolicy(max_age_minutes=10)},
        )
        enriched = result.samples[0]
        self.assertEqual(enriched.features["ctx__liquidity__available"], 0.0)
        self.assertIsNone(enriched.features["ctx__liquidity__spread_score"])
        self.assertEqual(enriched.features["ctx__integrity__available"], 0.0)
        self.assertIsNone(enriched.features["ctx__integrity__anomaly_score"])

    def test_latest_available_observation_wins_within_scope(self) -> None:
        sample = self._sample()
        observations = [
            ContextObservation(
                source="news",
                available_at=sample.timestamp - timedelta(minutes=8),
                scope="symbol",
                entity="NVDA",
                features={"sentiment": -0.2},
                observation_id="older",
            ),
            ContextObservation(
                source="news",
                available_at=sample.timestamp - timedelta(minutes=2),
                scope="symbol",
                entity="NVDA",
                features={"sentiment": 0.5},
                observation_id="newer",
            ),
        ]
        result = join_point_in_time_context([sample], observations)
        enriched = result.samples[0]
        self.assertEqual(enriched.features["ctx__news__sentiment"], 0.5)
        self.assertEqual(enriched.features["ctx__news__age_minutes"], 2.0)
        self.assertEqual(
            result.audit[0]["sources"]["news"][0]["observation_id"], "newer"
        )

    def test_schema_is_stable_and_targets_are_preserved(self) -> None:
        first = self._sample(symbol="NVDA", minute=10)
        second = self._sample(symbol="AAPL", minute=11)
        observations = [
            ContextObservation(
                source="news",
                available_at=first.timestamp - timedelta(minutes=1),
                scope="symbol",
                entity="NVDA",
                features={"sentiment": 0.7, "catalyst_strength": 0.8},
            )
        ]

        result = join_point_in_time_context(
            [first, second],
            observations,
            expected_sources=("news", "liquidity"),
        )
        one, two = result.samples

        self.assertIn("ctx__news__sentiment", result.feature_names)
        self.assertIn("ctx__liquidity__available", result.feature_names)
        self.assertEqual(one.features["ctx__news__available"], 1.0)
        self.assertEqual(two.features["ctx__news__available"], 0.0)
        self.assertIsNone(two.features["ctx__news__sentiment"])
        self.assertEqual(two.features["ctx__liquidity__available"], 0.0)
        self.assertEqual(one.targets, first.targets)
        self.assertEqual(two.targets, second.targets)
        self.assertGreater(two.features["context_missing_fraction"], 0.0)

    def test_invalid_temporal_order_is_rejected(self) -> None:
        timestamp = datetime(2026, 1, 5, 15, 10, tzinfo=timezone.utc)
        with self.assertRaises(ValueError):
            ContextObservation(
                source="news",
                event_time=timestamp + timedelta(minutes=1),
                available_at=timestamp,
                features={"sentiment": 0.1},
            )


if __name__ == "__main__":
    unittest.main()
