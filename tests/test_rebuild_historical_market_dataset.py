from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from scripts.rebuild_historical_market_dataset import rebuild_from_source_dataset
from trading.dataset import MarketBar
from trading.historical_market import write_canonical_bars_csv


class HistoricalMarketRebuildTests(unittest.TestCase):
    def _legacy_source_dataset(self, root: Path) -> None:
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

        source_cache = write_canonical_bars_csv(bars, root / "source_bars.csv")
        manifest = {
            "schema_version": 1,
            "builder": "ourai-historical-market-v1",
            "dataset_id": "legacy-v1-dataset",
            "source": {
                "schema_version": 1,
                "provider": "alpaca",
                "feed": "iex",
                "timeframe": "1Min",
                "adjustment": "raw",
                "partial": False,
            },
            "source_cache": source_cache,
            "configuration": {
                "horizons_minutes": [5, 15],
                "train_fraction": 0.50,
                "validation_fraction": 0.25,
                "minimum_absolute_return_bps": 3.0,
                "spread_multiplier": 1.5,
            },
        }
        (root / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def test_rebuilds_verified_source_cache_without_existing_splits(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "legacy"
            output = root / "v2"
            source.mkdir()
            self._legacy_source_dataset(source)

            manifest = rebuild_from_source_dataset(source, output)

            self.assertEqual(manifest["builder"], "ourai-historical-market-v2")
            self.assertEqual(manifest["configuration"]["bar_availability_delay_minutes"], 1.0)
            self.assertNotEqual(manifest["dataset_id"], "legacy-v1-dataset")
            self.assertTrue((output / "train.jsonl").is_file())
            self.assertTrue((output / "validation.jsonl").is_file())
            self.assertTrue((output / "test.jsonl").is_file())

            first_train = json.loads(
                (output / "train.jsonl").read_text(encoding="utf-8").splitlines()[0]
            )
            self.assertEqual(first_train["timestamp"], "2026-07-06T13:31:00+00:00")
            self.assertEqual(
                first_train["targets"]["5m"]["future_timestamp"],
                "2026-07-06T13:36:00+00:00",
            )

    def test_refuses_rebuild_when_source_cache_hash_changed(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "legacy"
            output = root / "v2"
            source.mkdir()
            self._legacy_source_dataset(source)

            with (source / "source_bars.csv").open("a", encoding="utf-8") as handle:
                handle.write("\n")

            with self.assertRaisesRegex(ValueError, "hash does not match"):
                rebuild_from_source_dataset(source, output)


if __name__ == "__main__":
    unittest.main()
