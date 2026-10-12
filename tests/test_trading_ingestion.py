from __future__ import annotations

import csv
import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from trading.ingestion import (
    CSVIngestionConfig,
    load_market_bars_csv,
    prepare_trading_dataset,
    sha256_file,
)


class TradingIngestionTests(unittest.TestCase):
    def _write_csv(self, path: Path, *, days: int = 5, duplicate: bool = False) -> None:
        rows = []
        for day in range(days):
            start = datetime(2026, 1, 5 + day, 14, 30, tzinfo=timezone.utc)
            for minute in range(70):
                close = 100 + day + minute * 0.05
                rows.append(
                    {
                        "symbol": "NVDA",
                        "timestamp": (start + timedelta(minutes=minute)).isoformat(),
                        "open": f"{close - 0.02:.4f}",
                        "high": f"{close + 0.04:.4f}",
                        "low": f"{close - 0.04:.4f}",
                        "close": f"{close:.4f}",
                        "volume": str(1000 + minute * 10),
                        "bid": f"{close - 0.01:.4f}",
                        "ask": f"{close + 0.01:.4f}",
                        "session": "regular",
                    }
                )
        if duplicate:
            rows.append(dict(rows[0]))

        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def test_ingestion_records_source_hash_and_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "bars.csv"
            self._write_csv(source, days=3)
            bars, manifest = load_market_bars_csv(source)

            self.assertEqual(len(bars), 210)
            self.assertEqual(manifest["row_count"], 210)
            self.assertEqual(manifest["symbols"], ["NVDA"])
            self.assertEqual(manifest["source_sha256"], sha256_file(source))
            self.assertIn("bid", manifest["optional_columns_present"])
            self.assertLess(bars[0].timestamp, bars[-1].timestamp)

    def test_naive_timestamp_requires_explicit_timezone(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "bars.csv"
            source.write_text(
                "symbol,timestamp,open,high,low,close,volume\n"
                "AAPL,2026-01-05T09:30:00,100,101,99,100.5,1000\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "source_timezone"):
                load_market_bars_csv(source)

            bars, _ = load_market_bars_csv(
                source,
                config=CSVIngestionConfig(source_timezone="America/New_York"),
            )
            self.assertEqual(bars[0].timestamp.hour, 14)

    def test_duplicate_symbol_timestamp_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "bars.csv"
            self._write_csv(source, days=1, duplicate=True)
            with self.assertRaisesRegex(ValueError, "duplicate symbol/timestamp"):
                load_market_bars_csv(source)

    def test_prepare_dataset_writes_hashed_chronological_splits(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "bars.csv"
            output = root / "processed"
            self._write_csv(source, days=5)

            manifest = prepare_trading_dataset(
                source,
                output,
                train_fraction=0.60,
                validation_fraction=0.20,
            )

            for name in ("train", "validation", "test"):
                path = output / f"{name}.jsonl"
                self.assertTrue(path.is_file())
                self.assertGreater(manifest["outputs"][name]["sample_count"], 0)
                self.assertEqual(manifest["outputs"][name]["sha256"], sha256_file(path))

            stored_manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            train_dates = set(stored_manifest["dataset"]["splits"]["train"]["market_dates"])
            validation_dates = set(stored_manifest["dataset"]["splits"]["validation"]["market_dates"])
            test_dates = set(stored_manifest["dataset"]["splits"]["test"]["market_dates"])
            self.assertTrue(train_dates.isdisjoint(validation_dates))
            self.assertTrue(train_dates.isdisjoint(test_dates))
            self.assertTrue(validation_dates.isdisjoint(test_dates))
            self.assertEqual(stored_manifest["source"]["source_sha256"], sha256_file(source))


if __name__ == "__main__":
    unittest.main()
