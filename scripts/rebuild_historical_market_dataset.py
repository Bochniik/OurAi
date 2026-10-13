"""Rebuild an OurAI historical dataset from its canonical local bar cache.

This path intentionally does not contact Alpaca and does not read any existing
train/validation/test JSONL files. It verifies the cached source-bar hash from the
source dataset manifest, reuses the original dataset configuration, and rebuilds new
splits with the current historical dataset builder.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

from trading.dataset import MarketBar
from trading.historical_market import HistoricalFetchResult, build_historical_market_dataset
from trading.ingestion import sha256_file


_REQUIRED_COLUMNS = {
    "symbol",
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "session",
}
_REQUIRED_CONFIGURATION = {
    "horizons_minutes",
    "train_fraction",
    "validation_fraction",
    "minimum_absolute_return_bps",
    "spread_multiplier",
}


def _mapping(payload: object, name: str) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping):
        raise ValueError(f"source manifest field {name!r} must be an object")
    return payload


def _read_canonical_bars(path: Path) -> tuple[MarketBar, ...]:
    if not path.is_file():
        raise FileNotFoundError(f"canonical source bars not found: {path}")

    bars: list[MarketBar] = []
    seen: set[tuple[str, datetime]] = set()
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or ())
        missing = sorted(_REQUIRED_COLUMNS - fieldnames)
        if missing:
            raise ValueError(
                "canonical source bars are missing required columns: " + ", ".join(missing)
            )

        for line_number, row in enumerate(reader, start=2):
            try:
                timestamp = datetime.fromisoformat(
                    str(row["timestamp"]).replace("Z", "+00:00")
                )
                if timestamp.tzinfo is None or timestamp.utcoffset() is None:
                    raise ValueError("timestamp must be timezone-aware")
                bar = MarketBar(
                    symbol=row["symbol"],
                    timestamp=timestamp,
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                    volume=float(row["volume"]),
                    session=row.get("session") or None,
                )
            except Exception as exc:
                raise ValueError(
                    f"invalid canonical source bar at CSV line {line_number}"
                ) from exc

            key = (bar.symbol, bar.timestamp)
            if key in seen:
                raise ValueError(
                    f"duplicate canonical source bar: {bar.symbol} {bar.timestamp.isoformat()}"
                )
            seen.add(key)
            bars.append(bar)

    if not bars:
        raise ValueError("canonical source bar cache is empty")
    bars.sort(key=lambda item: (item.symbol, item.timestamp))
    return tuple(bars)


def rebuild_from_source_dataset(
    source_dataset_dir: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Rebuild from a prior dataset's verified source cache without touching its splits."""

    source_root = Path(source_dataset_dir)
    destination = Path(output_dir)
    if source_root.resolve() == destination.resolve():
        raise ValueError("output directory must differ from source dataset directory")
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("output directory must be empty or not yet exist")

    manifest_path = source_root / "manifest.json"
    source_bars_path = source_root / "source_bars.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"source manifest not found: {manifest_path}")

    old_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(old_manifest, Mapping):
        raise ValueError("source manifest must contain a JSON object")

    source_cache = _mapping(old_manifest.get("source_cache"), "source_cache")
    source_contract = _mapping(old_manifest.get("source"), "source")
    configuration = _mapping(old_manifest.get("configuration"), "configuration")
    missing_config = sorted(_REQUIRED_CONFIGURATION - set(configuration))
    if missing_config:
        raise ValueError(
            "source manifest configuration is missing: " + ", ".join(missing_config)
        )

    expected_sha = str(source_cache.get("sha256") or "").strip().lower()
    if len(expected_sha) != 64:
        raise ValueError("source manifest does not contain a valid source_cache.sha256")
    actual_sha = sha256_file(source_bars_path)
    if actual_sha.lower() != expected_sha:
        raise ValueError(
            "source_bars.csv hash does not match source manifest; refusing rebuild"
        )

    bars = _read_canonical_bars(source_bars_path)
    expected_rows = source_cache.get("row_count")
    if expected_rows is not None and int(expected_rows) != len(bars):
        raise ValueError(
            "source_bars.csv row count does not match source manifest; refusing rebuild"
        )

    fetched = HistoricalFetchResult(
        bars=bars,
        manifest=dict(source_contract),
    )
    return build_historical_market_dataset(
        fetched,
        destination,
        horizons_minutes=tuple(int(value) for value in configuration["horizons_minutes"]),
        train_fraction=float(configuration["train_fraction"]),
        validation_fraction=float(configuration["validation_fraction"]),
        minimum_absolute_return_bps=float(configuration["minimum_absolute_return_bps"]),
        spread_multiplier=float(configuration["spread_multiplier"]),
    )


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description=(
            "Rebuild a historical OurAI dataset from an existing verified "
            "source_bars.csv cache without downloading market data again. Existing "
            "train/validation/test JSONL files are not read."
        )
    )
    result.add_argument(
        "--source-dataset-dir",
        required=True,
        help="Existing dataset directory containing manifest.json and source_bars.csv.",
    )
    result.add_argument(
        "--output-dir",
        required=True,
        help="New empty/nonexistent destination directory for the rebuilt dataset.",
    )
    return result


def main() -> int:
    args = parser().parse_args()
    source_root = Path(args.source_dataset_dir)
    old_manifest = json.loads((source_root / "manifest.json").read_text(encoding="utf-8"))
    manifest = rebuild_from_source_dataset(source_root, Path(args.output_dir))
    summary = {
        "source_dataset_id": old_manifest.get("dataset_id"),
        "rebuilt_dataset_id": manifest["dataset_id"],
        "builder": manifest["builder"],
        "output_dir": str(Path(args.output_dir).resolve()),
        "source_rows": manifest["source_cache"]["row_count"],
        "sample_count": manifest["dataset"]["sample_count"],
        "market_date_count": manifest["dataset"]["market_date_count"],
        "splits": manifest["dataset"].get("splits"),
        "bar_availability_delay_minutes": manifest["configuration"].get(
            "bar_availability_delay_minutes"
        ),
        "source_cache_verified": True,
        "existing_split_files_read": False,
        "test_lockbox_evaluated": False,
        "manifest_sha256": manifest["manifest_sha256"],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
