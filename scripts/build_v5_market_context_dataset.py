"""Build the frozen market/sector context dataset for v5 temporal direction research.

This dataset is context-only. SPY/QQQ/sector ETF targets are never used by the
v5 direction experiment; only causal features at the same or earlier availability
timestamp may be consumed. The v4 target validation/test files are not opened here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

from trading.historical_market import HistoricalMarketConfig, fetch_and_build_alpaca_dataset


PROTOCOL_PATH = (
    Path(__file__).resolve().parents[1]
    / "research"
    / "v5_temporal_market_context_protocol_v1.json"
)
SOURCE_START = datetime(2024, 10, 1, 0, 0, tzinfo=timezone.utc)
SOURCE_END = datetime(2025, 10, 1, 0, 0, tzinfo=timezone.utc)
CONTEXT_SYMBOLS = ("SPY", "QQQ", "XLK", "XLC", "XLY", "XLF")
SESSIONS = ("regular",)
HORIZONS_MINUTES = (15,)
TRAIN_FRACTION = 0.70
VALIDATION_FRACTION = 0.15
FEED = "iex"
ADJUSTMENT = "raw"
MAX_PAGES_PER_BATCH = 100
EXPECTED_MARKET_DATES = 249
EXPECTED_SPLIT_DATE_COUNTS = {"train": 174, "validation": 37, "test": 38}

DatasetFetcher = Callable[..., Mapping[str, Any]]


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_protocol() -> dict[str, Any]:
    if not PROTOCOL_PATH.is_file():
        raise FileNotFoundError(f"v5 temporal protocol not found: {PROTOCOL_PATH}")
    payload = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if payload.get("protocol_name") != "ourai-v5-temporal-market-context-direction-v1":
        raise ValueError("unexpected v5 temporal protocol identity")
    context = payload.get("market_context") or {}
    if tuple(context.get("symbols") or ()) != CONTEXT_SYMBOLS:
        raise ValueError("v5 market-context symbol set drifted from preregistration")
    if context.get("targets_from_context_symbols_used") is not False:
        raise ValueError("context targets must remain unused")
    return payload


def build_v5_market_context_dataset(
    output_dir: str | Path,
    *,
    dataset_fetcher: DatasetFetcher = fetch_and_build_alpaca_dataset,
) -> dict[str, Any]:
    protocol = _load_protocol()
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    protocol_copy = destination / "v5_temporal_market_context_protocol.json"
    protocol_copy.write_text(
        json.dumps(protocol, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    config = HistoricalMarketConfig(
        symbols=CONTEXT_SYMBOLS,
        start=SOURCE_START,
        end=SOURCE_END,
        feed=FEED,
        adjustment=ADJUSTMENT,
        sessions=SESSIONS,
        max_pages_per_batch=MAX_PAGES_PER_BATCH,
        allow_partial=False,
    )
    manifest = dataset_fetcher(
        destination,
        market_config=config,
        horizons_minutes=HORIZONS_MINUTES,
        train_fraction=TRAIN_FRACTION,
        validation_fraction=VALIDATION_FRACTION,
    )

    if manifest.get("builder") != "ourai-historical-market-v2":
        raise ValueError("v5 market context requires historical-v2 causal builder")
    source = manifest.get("source") or {}
    if bool(source.get("partial")):
        raise ValueError("partial market-context history is not allowed")

    dataset = manifest.get("dataset") or {}
    market_date_count = int(dataset.get("market_date_count") or 0)
    if market_date_count != EXPECTED_MARKET_DATES:
        raise ValueError(
            f"market-context date count mismatch: {market_date_count} != {EXPECTED_MARKET_DATES}"
        )
    splits = dataset.get("splits") or {}
    observed_counts = {
        name: len((splits.get(name) or {}).get("market_dates") or [])
        for name in EXPECTED_SPLIT_DATE_COUNTS
    }
    if observed_counts != EXPECTED_SPLIT_DATE_COUNTS:
        raise ValueError(
            "market-context split boundaries no longer align with v4 target dates: "
            f"{observed_counts}"
        )

    result = {
        "schema_version": 1,
        "protocol_name": protocol["protocol_name"],
        "protocol_sha256": _canonical_sha256(protocol),
        "base_target_dataset_id": protocol["base_target_dataset"]["dataset_id"],
        "context_dataset_id": manifest.get("dataset_id"),
        "manifest_sha256": manifest.get("manifest_sha256"),
        "builder": manifest.get("builder"),
        "symbols": list(CONTEXT_SYMBOLS),
        "source_start": SOURCE_START.isoformat().replace("+00:00", "Z"),
        "source_end": SOURCE_END.isoformat().replace("+00:00", "Z"),
        "source_partial": source.get("partial"),
        "source_rows": (manifest.get("source_cache") or {}).get("row_count"),
        "sample_count": dataset.get("sample_count"),
        "market_date_count": market_date_count,
        "split_market_date_counts": observed_counts,
        "context_targets_used_by_direction_research": False,
        "v4_validation_accessed": False,
        "v4_test_accessed": False,
    }
    (destination / "v5_market_context_result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Build frozen SPY/QQQ/sector context for v5 temporal direction research."
    )
    result.add_argument(
        "--output-dir",
        default="data/processed/market/v5-temporal-market-context",
        help="Local output directory (gitignored).",
    )
    return result


def main() -> int:
    args = parser().parse_args()
    protocol = _load_protocol()
    print("=== OURAI V5 TEMPORAL MARKET CONTEXT ===")
    print("Protocol:", protocol["protocol_name"])
    print("Protocol SHA256:", _canonical_sha256(protocol))
    print("Context symbols:", ", ".join(CONTEXT_SYMBOLS))
    print("Context targets used: NO")
    print("V4 validation: NOT ACCESSED")
    print("V4 test lockbox: NOT ACCESSED")
    print()
    summary = build_v5_market_context_dataset(args.output_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
