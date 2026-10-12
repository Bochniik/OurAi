from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from trading.dataset import MarketBar
from trading.historical_market import (
    AlpacaCredentials,
    HistoricalFetchResult,
    HistoricalMarketConfig,
    build_historical_market_dataset,
    fetch_alpaca_historical_bars,
)


class HistoricalMarketDatasetTests(unittest.TestCase):
    def test_fetches_paginated_alpaca_rows_without_persisting_credentials(self):
        calls: list[str] = []

        def fake_request(url, headers):
            calls.append(url)
            self.assertEqual(headers["APCA-API-KEY-ID"], "local-key")
            self.assertEqual(headers["APCA-API-SECRET-KEY"], "local-secret")
            if "page_token=page-2" in url:
                return {
                    "bars": {
                        "AAPL": [
                            {"t": "2026-07-06T13:32:00Z", "o": 100.2, "h": 100.4, "l": 100.1, "c": 100.3, "v": 1200}
                        ]
                    },
                    "next_page_token": None,
                }
            return {
                "bars": {
                    "AAPL": [
                        {"t": "2026-07-06T13:30:00Z", "o": 100.0, "h": 100.2, "l": 99.9, "c": 100.1, "v": 1000},
                        {"t": "2026-07-06T13:31:00Z", "o": 100.1, "h": 100.3, "l": 100.0, "c": 100.2, "v": 1100},
                    ]
                },
                "next_page_token": "page-2",
            }

        config = HistoricalMarketConfig(
            symbols=("AAPL",),
            start=datetime(2026, 7, 6, 13, 30, tzinfo=timezone.utc),
            end=datetime(2026, 7, 6, 14, 0, tzinfo=timezone.utc),
            sessions=("regular",),
        )
        result = fetch_alpaca_historical_bars(
            config,
            credentials=AlpacaCredentials("local-key", "local-secret"),
            request_json=fake_request,
        )

        self.assertEqual(len(calls), 2)
        self.assertEqual(len(result.bars), 3)
        self.assertEqual(result.manifest["provider"], "alpaca")
        self.assertEqual(result.manifest["feed"], "iex")
        self.assertFalse(result.manifest["partial"])
        self.assertFalse(result.manifest["credentials_persisted"])
        serialized = json.dumps(result.manifest, sort_keys=True)
        self.assertNotIn("local-key", serialized)
        self.assertNotIn("local-secret", serialized)

    def test_rejects_partial_history_by_default(self):
        def always_more(url, headers):
            return {
                "bars": {
                    "AAPL": [
                        {"t": "2026-07-06T13:30:00Z", "o": 100, "h": 101, "l": 99, "c": 100.5, "v": 1000}
                    ]
                },
                "next_page_token": "still-more",
            }

        config = HistoricalMarketConfig(
            symbols=("AAPL",),
            start=datetime(2026, 7, 6, 13, 30, tzinfo=timezone.utc),
            end=datetime(2026, 7, 6, 14, 0, tzinfo=timezone.utc),
            max_pages_per_batch=1,
        )
        with self.assertRaisesRegex(ValueError, "partial history"):
            fetch_alpaca_historical_bars(
                config,
                credentials=AlpacaCredentials("key", "secret"),
                request_json=always_more,
            )

    def test_filters_non_requested_sessions(self):
        def fake_request(url, headers):
            return {
                "bars": {
                    "AAPL": [
                        {"t": "2026-07-06T11:00:00Z", "o": 99, "h": 100, "l": 98.5, "c": 99.5, "v": 500},
                        {"t": "2026-07-06T13:30:00Z", "o": 100, "h": 101, "l": 99.5, "c": 100.5, "v": 1000},
                    ]
                },
                "next_page_token": None,
            }

        config = HistoricalMarketConfig(
            symbols=("AAPL",),
            start=datetime(2026, 7, 6, 10, 0, tzinfo=timezone.utc),
            end=datetime(2026, 7, 6, 14, 0, tzinfo=timezone.utc),
            sessions=("regular",),
        )
        result = fetch_alpaca_historical_bars(
            config,
            credentials=AlpacaCredentials("key", "secret"),
            request_json=fake_request,
        )
        self.assertEqual(len(result.bars), 1)
        self.assertEqual(result.bars[0].session, "regular")

    def test_builds_hashed_chronological_dataset_outputs(self):
        bars: list[MarketBar] = []
        for day_offset in range(4):
            start = datetime(2026, 7, 6 + day_offset, 13, 30, tzinfo=timezone.utc)
            for minute in range(31):
                timestamp = start + timedelta(minutes=minute)
                price = 100.0 + day_offset + minute * 0.02
                bars.append(
                    MarketBar(
                        symbol="AAPL",
                        timestamp=timestamp,
                        open=price,
                        high=price + 0.10,
                        low=price - 0.10,
                        close=price + 0.01,
                        volume=1000 + minute,
                        session="regular",
                    )
                )

        fetched = HistoricalFetchResult(
            bars=tuple(bars),
            manifest={
                "schema_version": 1,
                "provider": "alpaca",
                "feed": "iex",
                "timeframe": "1Min",
                "adjustment": "raw",
                "partial": False,
            },
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest = build_historical_market_dataset(
                fetched,
                temp_dir,
                horizons_minutes=(5, 15),
                train_fraction=0.50,
                validation_fraction=0.25,
            )
            root = Path(temp_dir)
            self.assertTrue((root / "source_bars.csv").is_file())
            self.assertTrue((root / "train.jsonl").is_file())
            self.assertTrue((root / "validation.jsonl").is_file())
            self.assertTrue((root / "test.jsonl").is_file())
            self.assertTrue((root / "manifest.json").is_file())
            self.assertEqual(manifest["source_cache"]["row_count"], len(bars))
            self.assertEqual(len(manifest["source_cache"]["sha256"]), 64)
            self.assertEqual(len(manifest["dataset_id"]), 20)
            self.assertEqual(manifest["dataset"]["market_date_count"], 4)
            self.assertEqual(set(manifest["dataset"]["symbols"]), {"AAPL"})
            split_dates = manifest["dataset"]["splits"]
            self.assertEqual(len(split_dates["train"]["market_dates"]), 2)
            self.assertEqual(len(split_dates["validation"]["market_dates"]), 1)
            self.assertEqual(len(split_dates["test"]["market_dates"]), 1)
            self.assertEqual(len(manifest["manifest_sha256"]), 64)

    def test_v1_rejects_coarser_timeframes(self):
        with self.assertRaisesRegex(ValueError, "1Min only"):
            HistoricalMarketConfig(
                symbols=("AAPL",),
                start=datetime(2026, 7, 6, 13, 30, tzinfo=timezone.utc),
                end=datetime(2026, 7, 6, 14, 0, tzinfo=timezone.utc),
                timeframe="5Min",
            )


if __name__ == "__main__":
    unittest.main()
