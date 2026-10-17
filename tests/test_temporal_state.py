from __future__ import annotations

import json
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from trading.dataset import TradingSample
from trading.temporal_state import enrich_temporal_state


class TemporalStateTests(unittest.TestCase):
    def _sample(self, minute: int, price: float, return_1m: float | None) -> TradingSample:
        timestamp = datetime(2025, 1, 2, 14, 31, tzinfo=timezone.utc) + timedelta(minutes=minute)
        return TradingSample(
            symbol="AAPL",
            timestamp=timestamp,
            session_date="2025-01-02",
            session="regular",
            features={
                "price": price,
                "return_1m": return_1m,
                "return_5m": None,
                "return_15m": None,
                "return_30m": None,
                "distance_from_vwap": 0.001 * minute,
                "range_position_20": 0.5,
                "relative_volume_20": 1.0 + minute * 0.1,
                "realized_volatility_20": 0.01,
            },
            targets={},
        )

    def test_exact_lags_and_session_open_return(self) -> None:
        rows = [
            self._sample(0, 100.0, None),
            self._sample(1, 101.0, 0.01),
            self._sample(2, 102.0, 102.0 / 101.0 - 1.0),
            self._sample(3, 103.0, 103.0 / 102.0 - 1.0),
        ]
        enriched = enrich_temporal_state(rows)
        last = enriched[-1].features
        self.assertAlmostEqual(last["temporal_return_2m"], 103.0 / 101.0 - 1.0)
        self.assertAlmostEqual(last["temporal_return_3m"], 0.03)
        self.assertAlmostEqual(last["temporal_return_since_session_open"], 0.03)
        self.assertAlmostEqual(last["temporal_return_1m_lag_1"], 102.0 / 101.0 - 1.0)
        self.assertIsNone(last["temporal_return_1m_lag_5"])

    def test_future_sample_cannot_change_past_features(self) -> None:
        prefix = [
            self._sample(0, 100.0, None),
            self._sample(1, 101.0, 0.01),
            self._sample(2, 102.0, 102.0 / 101.0 - 1.0),
        ]
        first = enrich_temporal_state(prefix + [self._sample(3, 103.0, 0.01)])
        second = enrich_temporal_state(prefix + [self._sample(3, 150.0, 0.50)])
        for index in range(len(prefix)):
            self.assertEqual(first[index].features, second[index].features)

    def test_selective_protocol_keeps_holdouts_closed(self) -> None:
        path = Path(__file__).resolve().parents[1] / "research" / "v4_selective_direction_protocol_v1.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(payload["dataset_id"], "7d5f3b066a30ba878bba")
        self.assertEqual(payload["direction_candidate"]["tail_quantile"], 0.70)
        self.assertEqual(payload["magnitude_gate"]["training_prediction_quantile_for_gate"], 0.70)
        self.assertFalse(payload["research_safety"]["validation_split_accessed"])
        self.assertFalse(payload["research_safety"]["test_split_accessed"])
        self.assertFalse(payload["development_gate"]["pnl_used"])


if __name__ == "__main__":
    unittest.main()
