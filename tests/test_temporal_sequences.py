from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone

import numpy as np

from trading.dataset import TradingSample
from trading.temporal_sequences import (
    CONTEXT_FEATURES,
    TARGET_FEATURES,
    build_temporal_sequence_corpus,
    fit_sequence_scaler,
    sample_key,
)


def _sample(symbol: str, timestamp: datetime, *, value: float = 1.0) -> TradingSample:
    features = {
        "return_1m": value,
        "return_5m": value + 0.1,
        "return_15m": value + 0.2,
        "return_30m": value + 0.3,
        "distance_from_vwap": value + 0.4,
        "relative_volume_20": value + 0.5,
        "range_position_20": value + 0.6,
        "realized_volatility_20": value + 0.7,
        "spread_bps": value + 0.8,
        "session_progress": value + 0.9,
    }
    return TradingSample(
        symbol=symbol,
        timestamp=timestamp,
        session_date="2025-01-02",
        session="regular",
        features=features,
        targets={"15m": {"direction": "UP", "future_return": 0.001}},
    )


class TemporalSequenceTests(unittest.TestCase):
    def test_exact_60_minute_sequence_and_context_roles(self):
        start = datetime(2025, 1, 2, 14, 31, tzinfo=timezone.utc)
        targets = [_sample("NVDA", start + timedelta(minutes=i), value=10 + i) for i in range(60)]
        contexts = []
        for i in range(60):
            timestamp = start + timedelta(minutes=i)
            contexts.extend(
                [
                    _sample("SPY", timestamp, value=100 + i),
                    _sample("QQQ", timestamp, value=200 + i),
                    _sample("XLK", timestamp, value=300 + i),
                ]
            )

        corpus = build_temporal_sequence_corpus(targets, contexts, sequence_length=60)
        key = sample_key(targets[-1])
        self.assertIn(key, corpus.ref_by_key)
        ref = corpus.ref_by_key[key]
        raw = corpus.raw_window(ref)
        self.assertEqual(raw.shape, (60, len(TARGET_FEATURES) + 3 * len(CONTEXT_FEATURES)))
        self.assertAlmostEqual(float(raw[-1, 0]), 69.0)
        self.assertAlmostEqual(float(raw[-1, len(TARGET_FEATURES)]), 159.0)
        qqq_offset = len(TARGET_FEATURES) + len(CONTEXT_FEATURES)
        sector_offset = qqq_offset + len(CONTEXT_FEATURES)
        self.assertAlmostEqual(float(raw[-1, qqq_offset]), 259.0)
        self.assertAlmostEqual(float(raw[-1, sector_offset]), 359.0)

        scaler = fit_sequence_scaler(corpus, [sample_key(sample) for sample in targets])
        transformed = corpus.transform_window(ref, scaler)
        self.assertEqual(transformed.shape, (60, raw.shape[1] * 2))
        self.assertTrue(np.all(np.isfinite(transformed)))

    def test_missing_same_timestamp_context_is_masked_not_forward_filled(self):
        start = datetime(2025, 1, 2, 14, 31, tzinfo=timezone.utc)
        targets = [_sample("NVDA", start + timedelta(minutes=i), value=10 + i) for i in range(60)]
        contexts = []
        for i in range(60):
            timestamp = start + timedelta(minutes=i)
            contexts.append(_sample("SPY", timestamp, value=100 + i))
            if i != 59:
                contexts.append(_sample("QQQ", timestamp, value=200 + i))
            contexts.append(_sample("XLK", timestamp, value=300 + i))
        # A future QQQ row must not be used as a substitute for the missing same-time row.
        contexts.append(_sample("QQQ", start + timedelta(minutes=60), value=999.0))

        corpus = build_temporal_sequence_corpus(targets, contexts, sequence_length=60)
        ref = corpus.ref_by_key[sample_key(targets[-1])]
        raw = corpus.raw_window(ref)
        qqq_offset = len(TARGET_FEATURES) + len(CONTEXT_FEATURES)
        self.assertTrue(np.isnan(raw[-1, qqq_offset]))

        scaler = fit_sequence_scaler(corpus, [sample_key(sample) for sample in targets])
        transformed = corpus.transform_window(ref, scaler)
        missing_offset = raw.shape[1] + qqq_offset
        self.assertEqual(float(transformed[-1, missing_offset]), 1.0)

    def test_non_contiguous_target_history_is_not_sequence_eligible(self):
        start = datetime(2025, 1, 2, 14, 31, tzinfo=timezone.utc)
        timestamps = [start + timedelta(minutes=i) for i in range(59)]
        timestamps.append(start + timedelta(minutes=60))
        targets = [_sample("NVDA", timestamp, value=10 + i) for i, timestamp in enumerate(timestamps)]
        contexts = []
        for timestamp in timestamps:
            contexts.extend(
                [
                    _sample("SPY", timestamp),
                    _sample("QQQ", timestamp),
                    _sample("XLK", timestamp),
                ]
            )
        corpus = build_temporal_sequence_corpus(targets, contexts, sequence_length=60)
        self.assertNotIn(sample_key(targets[-1]), corpus.ref_by_key)


if __name__ == "__main__":
    unittest.main()
