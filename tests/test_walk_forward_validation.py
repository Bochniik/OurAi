from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone

from trading.dataset import TradingSample
from trading.validation import (
    purged_walk_forward_folds,
    sample_label_end,
    walk_forward_manifest,
)


class WalkForwardValidationTests(unittest.TestCase):
    def _sample(
        self,
        day: int,
        minute: int,
        *,
        symbol: str = "NVDA",
        target_minutes: int = 5,
        future_override: datetime | None = None,
    ) -> TradingSample:
        timestamp = datetime(2026, 1, day, 15, minute, tzinfo=timezone.utc)
        future = future_override or timestamp + timedelta(minutes=target_minutes)
        return TradingSample(
            symbol=symbol,
            timestamp=timestamp,
            session_date=timestamp.date().isoformat(),
            session="regular",
            features={"price": 100.0 + day},
            targets={
                f"{target_minutes}m": {
                    "future_timestamp": future.isoformat(),
                    "future_return": 0.01,
                    "direction": "UP",
                }
            },
        )

    def _development_samples(self, days: range = range(5, 13)) -> list[TradingSample]:
        rows: list[TradingSample] = []
        for day in days:
            rows.append(self._sample(day, 0, symbol="NVDA"))
            rows.append(self._sample(day, 5, symbol="AAPL"))
        return rows

    def test_expanding_walk_forward_is_strictly_chronological(self) -> None:
        samples = self._development_samples()
        folds = purged_walk_forward_folds(
            samples,
            min_train_market_dates=3,
            validation_market_dates=2,
            step_market_dates=2,
            embargo_minutes=0,
        )

        self.assertGreaterEqual(len(folds), 2)
        for fold in folds:
            self.assertLess(
                max(sample_label_end(sample) for sample in fold.train),
                fold.validation_start,
            )
            self.assertLess(
                max(sample.timestamp for sample in fold.train),
                min(sample.timestamp for sample in fold.validation),
            )
            self.assertTrue(
                set(fold.train_market_dates).isdisjoint(fold.validation_market_dates)
            )

    def test_target_overlap_is_purged_even_when_feature_time_is_earlier(self) -> None:
        samples = self._development_samples(range(5, 10))
        validation_start = datetime(2026, 1, 8, 15, 0, tzinfo=timezone.utc)

        leaking = self._sample(
            7,
            10,
            symbol="TSLA",
            target_minutes=60,
            future_override=validation_start + timedelta(minutes=1),
        )
        samples.append(leaking)

        folds = purged_walk_forward_folds(
            samples,
            min_train_market_dates=3,
            validation_market_dates=1,
            step_market_dates=1,
            embargo_minutes=0,
        )
        first = folds[0]

        self.assertEqual(first.validation_market_dates, ("2026-01-08",))
        self.assertNotIn(leaking, first.train)
        self.assertGreaterEqual(first.purged_overlap_count, 1)

    def test_embargo_removes_samples_too_close_to_validation(self) -> None:
        samples = self._development_samples(range(5, 9))
        near_boundary = TradingSample(
            symbol="JPM",
            timestamp=datetime(2026, 1, 7, 14, 59, tzinfo=timezone.utc),
            session_date="2026-01-07",
            session="premarket",
            features={"price": 100.0},
            targets={
                "1m": {
                    "future_timestamp": datetime(
                        2026, 1, 7, 15, 0, tzinfo=timezone.utc
                    ).isoformat(),
                    "future_return": 0.0,
                    "direction": "NEUTRAL",
                }
            },
        )
        samples.append(near_boundary)

        folds = purged_walk_forward_folds(
            samples,
            min_train_market_dates=3,
            validation_market_dates=1,
            step_market_dates=1,
            embargo_minutes=24 * 60 + 2,
        )
        first = folds[0]

        self.assertNotIn(near_boundary, first.train)
        self.assertGreaterEqual(first.embargoed_count, 1)

    def test_rolling_window_limits_training_dates(self) -> None:
        samples = self._development_samples(range(5, 15))
        folds = purged_walk_forward_folds(
            samples,
            min_train_market_dates=3,
            validation_market_dates=1,
            step_market_dates=1,
            embargo_minutes=0,
            max_train_market_dates=4,
        )

        self.assertTrue(folds)
        for fold in folds:
            self.assertLessEqual(len(fold.train_market_dates), 4)
            self.assertGreaterEqual(len(fold.train_market_dates), 3)

    def test_manifest_is_deterministic_and_counts_purges(self) -> None:
        samples = self._development_samples(range(5, 11))
        folds = purged_walk_forward_folds(
            samples,
            min_train_market_dates=3,
            validation_market_dates=1,
            step_market_dates=1,
            embargo_minutes=0,
        )
        manifest = walk_forward_manifest(folds)

        self.assertEqual(manifest["fold_count"], len(folds))
        self.assertEqual(
            manifest["policy"], "strictly_chronological_purged_walk_forward"
        )
        self.assertEqual(
            manifest["total_validation_samples_across_folds"],
            sum(len(fold.validation) for fold in folds),
        )

    def test_rejects_too_little_history(self) -> None:
        samples = self._development_samples(range(5, 8))
        with self.assertRaises(ValueError):
            purged_walk_forward_folds(
                samples,
                min_train_market_dates=3,
                validation_market_dates=1,
            )

    def test_sample_label_end_uses_latest_target(self) -> None:
        timestamp = datetime(2026, 1, 5, 15, 0, tzinfo=timezone.utc)
        sample = TradingSample(
            symbol="NVDA",
            timestamp=timestamp,
            session_date="2026-01-05",
            session="regular",
            features={"price": 100.0},
            targets={
                "5m": {"future_timestamp": (timestamp + timedelta(minutes=5)).isoformat()},
                "60m": {"future_timestamp": (timestamp + timedelta(minutes=60)).isoformat()},
            },
        )
        self.assertEqual(sample_label_end(sample), timestamp + timedelta(minutes=60))


if __name__ == "__main__":
    unittest.main()
