from __future__ import annotations

import unittest
from datetime import datetime, timezone

from trading.orion_adapter import OrionAdapterConfig, normalize_orion_rows


class OrionMarketAdapterTests(unittest.TestCase):
    def test_yahoo_style_rows_use_time_and_volume(self) -> None:
        rows = [
            {
                "time": "2026-01-05T14:30:00Z",
                "open": 100.0,
                "high": 101.0,
                "low": 99.5,
                "close": 100.5,
                "volume": 125000,
            },
            {
                "time": "2026-01-05T14:31:00Z",
                "open": 100.5,
                "high": 101.2,
                "low": 100.2,
                "close": 101.0,
                "volume": 130000,
            },
        ]

        result = normalize_orion_rows(
            rows,
            config=OrionAdapterConfig(default_symbol="nvda", source_name="orion-yahoo"),
        )

        self.assertEqual(len(result.bars), 2)
        self.assertEqual(result.bars[0].symbol, "NVDA")
        self.assertEqual(result.bars[0].timestamp.tzinfo, timezone.utc)
        self.assertEqual(result.manifest["symbols"], ["NVDA"])
        self.assertEqual(result.manifest["aliases_used"]["timestamp"], ["time"])
        self.assertEqual(result.manifest["aliases_used"]["volume"], ["volume"])
        self.assertEqual(result.manifest["volume_semantics"], "source_volume")

    def test_mt5_style_rows_accept_unix_time_and_tick_volume(self) -> None:
        timestamp = datetime(2026, 1, 5, 14, 30, tzinfo=timezone.utc)
        rows = [
            {
                "time": int(timestamp.timestamp()),
                "open": 1.1700,
                "high": 1.1710,
                "low": 1.1690,
                "close": 1.1705,
                "tick_volume": 840,
            }
        ]

        result = normalize_orion_rows(
            rows,
            config=OrionAdapterConfig(default_symbol="EURUSD", source_name="orion-mt5"),
        )

        self.assertEqual(result.bars[0].timestamp, timestamp)
        self.assertEqual(result.bars[0].volume, 840.0)
        self.assertEqual(result.manifest["aliases_used"]["volume"], ["tick_volume"])
        self.assertEqual(result.manifest["volume_semantics"], "tick_volume")

    def test_naive_timestamp_requires_explicit_timezone(self) -> None:
        rows = [
            {
                "symbol": "AAPL",
                "time": "2026-01-05T09:30:00",
                "open": 100,
                "high": 101,
                "low": 99,
                "close": 100.5,
                "volume": 1000,
            }
        ]

        with self.assertRaisesRegex(ValueError, "source_timezone"):
            normalize_orion_rows(rows)

        result = normalize_orion_rows(
            rows,
            config=OrionAdapterConfig(source_timezone="America/New_York"),
        )
        self.assertEqual(result.bars[0].timestamp.hour, 14)
        self.assertEqual(result.bars[0].timestamp.minute, 30)

    def test_duplicate_rows_fail_closed_by_default(self) -> None:
        row = {
            "symbol": "AMD",
            "timestamp": "2026-01-05T14:30:00+00:00",
            "open": 150,
            "high": 151,
            "low": 149,
            "close": 150.5,
            "volume": 1000,
        }

        with self.assertRaisesRegex(ValueError, "duplicate symbol/timestamp"):
            normalize_orion_rows([row, dict(row)])

        result = normalize_orion_rows(
            [row, dict(row)],
            config=OrionAdapterConfig(strict_duplicates=False),
        )
        self.assertEqual(len(result.bars), 1)
        self.assertEqual(result.manifest["duplicates_skipped"], 1)
        self.assertEqual(result.manifest["input_row_count"], 2)

    def test_per_row_symbol_bid_ask_and_session_are_preserved(self) -> None:
        rows = [
            {
                "ticker": "meta",
                "timestamp": "2026-01-05T15:00:00+00:00",
                "open": 600.0,
                "high": 601.0,
                "low": 599.0,
                "close": 600.5,
                "volume": 5000,
                "best_bid": 600.45,
                "best_ask": 600.55,
                "market_session": "regular",
            }
        ]

        result = normalize_orion_rows(rows)
        bar = result.bars[0]
        self.assertEqual(bar.symbol, "META")
        self.assertEqual(bar.bid, 600.45)
        self.assertEqual(bar.ask, 600.55)
        self.assertEqual(bar.session, "regular")
        self.assertEqual(result.manifest["aliases_used"]["symbol"], ["ticker"])
        self.assertEqual(result.manifest["aliases_used"]["bid"], ["best_bid"])
        self.assertEqual(result.manifest["aliases_used"]["ask"], ["best_ask"])

    def test_rows_are_sorted_deterministically(self) -> None:
        rows = [
            {
                "symbol": "NVDA",
                "time": "2026-01-05T14:31:00Z",
                "open": 101,
                "high": 102,
                "low": 100,
                "close": 101.5,
                "volume": 1000,
            },
            {
                "symbol": "AAPL",
                "time": "2026-01-05T14:30:00Z",
                "open": 200,
                "high": 201,
                "low": 199,
                "close": 200.5,
                "volume": 2000,
            },
            {
                "symbol": "NVDA",
                "time": "2026-01-05T14:30:00Z",
                "open": 100,
                "high": 101,
                "low": 99,
                "close": 100.5,
                "volume": 900,
            },
        ]

        result = normalize_orion_rows(rows)
        self.assertEqual(
            [(bar.symbol, bar.timestamp.minute) for bar in result.bars],
            [("AAPL", 30), ("NVDA", 30), ("NVDA", 31)],
        )


if __name__ == "__main__":
    unittest.main()
