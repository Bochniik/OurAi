from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone

from trading.dataset import (
    MarketBar,
    build_causal_samples,
    chronological_session_split,
    dataset_manifest,
)


class TradingDatasetTests(unittest.TestCase):
    def _bars(
        self,
        *,
        symbol: str = "NVDA",
        start: datetime | None = None,
        count: int = 90,
        base_price: float = 100.0,
    ) -> list[MarketBar]:
        start = start or datetime(2026, 1, 5, 14, 30, tzinfo=timezone.utc)  # 09:30 ET
        bars = []
        for index in range(count):
            close = base_price + index * 0.10
            bars.append(
                MarketBar(
                    symbol=symbol,
                    timestamp=start + timedelta(minutes=index),
                    open=close - 0.03,
                    high=close + 0.05,
                    low=close - 0.05,
                    close=close,
                    volume=1_000 + index * 10,
                    bid=close - 0.01,
                    ask=close + 0.01,
                    session="regular",
                )
            )
        return bars

    def test_features_are_causal_when_future_price_changes(self) -> None:
        bars = self._bars(count=70)
        timestamp = bars[30].timestamp
        baseline = {
            sample.timestamp: sample for sample in build_causal_samples(bars)
        }[timestamp]

        changed = list(bars)
        future = changed[35]
        changed[35] = MarketBar(
            symbol=future.symbol,
            timestamp=future.timestamp,
            open=109.0,
            high=110.5,
            low=108.5,
            close=110.0,
            volume=future.volume,
            bid=109.99,
            ask=110.01,
            session=future.session,
        )
        mutated = {
            sample.timestamp: sample for sample in build_causal_samples(changed)
        }[timestamp]

        self.assertEqual(dict(baseline.features), dict(mutated.features))
        self.assertNotEqual(
            baseline.targets["5m"]["future_return"],
            mutated.targets["5m"]["future_return"],
        )

    def test_missing_exact_future_bar_does_not_create_fake_horizon(self) -> None:
        bars = self._bars(count=70)
        timestamp = bars[20].timestamp
        missing_timestamp = timestamp + timedelta(minutes=5)
        bars = [bar for bar in bars if bar.timestamp != missing_timestamp]

        sample = {
            item.timestamp: item for item in build_causal_samples(bars)
        }[timestamp]
        self.assertNotIn("5m", sample.targets)
        self.assertIn("15m", sample.targets)

    def test_targets_do_not_cross_named_market_sessions_by_default(self) -> None:
        bars = [
            MarketBar(
                symbol="AAPL",
                timestamp=datetime(2026, 1, 5, 14, 29, tzinfo=timezone.utc),
                open=100,
                high=100.2,
                low=99.9,
                close=100,
                volume=1000,
                session="premarket",
            ),
            MarketBar(
                symbol="AAPL",
                timestamp=datetime(2026, 1, 5, 14, 34, tzinfo=timezone.utc),
                open=100.2,
                high=100.5,
                low=100.1,
                close=100.4,
                volume=2000,
                session="regular",
            ),
        ]
        samples = build_causal_samples(bars, horizons_minutes=(5,))
        self.assertEqual(samples, [])

        crossing = build_causal_samples(
            bars,
            horizons_minutes=(5,),
            allow_session_transition=True,
        )
        self.assertEqual(len(crossing), 1)
        self.assertIn("5m", crossing[0].targets)

    def test_spread_can_expand_neutral_target_threshold(self) -> None:
        start = datetime(2026, 1, 5, 14, 30, tzinfo=timezone.utc)
        bars = [
            MarketBar(
                symbol="JPM",
                timestamp=start,
                open=100,
                high=100.1,
                low=99.9,
                close=100,
                volume=1000,
                bid=99.95,
                ask=100.05,
                session="regular",
            ),
            MarketBar(
                symbol="JPM",
                timestamp=start + timedelta(minutes=5),
                open=100.04,
                high=100.08,
                low=100.01,
                close=100.05,
                volume=1100,
                bid=100.00,
                ask=100.10,
                session="regular",
            ),
        ]
        sample = build_causal_samples(bars, horizons_minutes=(5,))[0]
        self.assertEqual(sample.targets["5m"]["direction"], "NEUTRAL")
        self.assertGreater(sample.targets["5m"]["neutral_threshold_bps"], 10.0)

    def test_chronological_split_keeps_market_dates_disjoint(self) -> None:
        bars: list[MarketBar] = []
        for day in range(5):
            start = datetime(2026, 1, 5 + day, 14, 30, tzinfo=timezone.utc)
            bars.extend(self._bars(start=start, count=70, base_price=100 + day))

        samples = build_causal_samples(bars)
        splits = chronological_session_split(samples, train_fraction=0.60, validation_fraction=0.20)
        date_sets = {
            name: {sample.session_date for sample in rows}
            for name, rows in splits.items()
        }

        self.assertTrue(date_sets["train"].isdisjoint(date_sets["validation"]))
        self.assertTrue(date_sets["train"].isdisjoint(date_sets["test"]))
        self.assertTrue(date_sets["validation"].isdisjoint(date_sets["test"]))
        self.assertLess(max(date_sets["train"]), min(date_sets["validation"]))
        self.assertLess(max(date_sets["validation"]), min(date_sets["test"]))

        manifest = dataset_manifest(samples, splits)
        self.assertEqual(manifest["market_date_count"], 5)
        self.assertIn("5m", manifest["resolved_targets"])
        self.assertEqual(manifest["feature_policy"], "point_in_time_only")


if __name__ == "__main__":
    unittest.main()
