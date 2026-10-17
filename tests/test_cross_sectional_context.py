from __future__ import annotations

import unittest
from datetime import datetime, timezone

from trading.cross_sectional import enrich_cross_sectional_context
from trading.dataset import TradingSample


class CrossSectionalContextTests(unittest.TestCase):
    def _sample(self, symbol: str, return_1m: float, relative_volume: float) -> TradingSample:
        timestamp = datetime(2025, 1, 2, 14, 31, tzinfo=timezone.utc)
        return TradingSample(
            symbol=symbol,
            timestamp=timestamp,
            session_date="2025-01-02",
            session="regular",
            features={
                "return_1m": return_1m,
                "return_5m": return_1m * 2,
                "return_15m": return_1m * 3,
                "return_30m": return_1m * 4,
                "relative_volume_20": relative_volume,
                "realized_volatility_20": 0.01,
                "distance_from_vwap": return_1m / 2,
                "range_position_20": 0.5,
            },
            targets={},
        )

    def test_peer_aggregates_exclude_current_symbol(self) -> None:
        aapl = self._sample("AAPL", 0.03, 3.0)
        msft = self._sample("MSFT", 0.01, 1.0)
        nvda = self._sample("NVDA", -0.01, 2.0)

        enriched = enrich_cross_sectional_context([aapl, msft, nvda])
        by_symbol = {sample.symbol: sample for sample in enriched}

        self.assertAlmostEqual(by_symbol["AAPL"].features["peer_return_1m_mean"], 0.0)
        self.assertAlmostEqual(by_symbol["AAPL"].features["return_1m_excess_vs_peers"], 0.03)
        self.assertAlmostEqual(by_symbol["AAPL"].features["peer_return_1m_breadth_up"], 0.5)
        self.assertAlmostEqual(by_symbol["AAPL"].features["peer_relative_volume_20_mean"], 1.5)

    def test_different_timestamp_is_never_used_as_peer(self) -> None:
        first = self._sample("AAPL", 0.02, 1.0)
        later = TradingSample(
            symbol="MSFT",
            timestamp=datetime(2025, 1, 2, 14, 32, tzinfo=timezone.utc),
            session_date="2025-01-02",
            session="regular",
            features=dict(first.features),
            targets={},
        )

        enriched = enrich_cross_sectional_context([first, later])
        self.assertIsNone(enriched[0].features["peer_return_1m_mean"])
        self.assertIsNone(enriched[1].features["peer_return_1m_mean"])
        self.assertEqual(
            enriched[0].features["cross_sectional_context_missing_fraction"],
            1.0,
        )


if __name__ == "__main__":
    unittest.main()
