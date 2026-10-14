"""Build the preregistered, non-overlapping OurAI v3 research dataset.

The v2 lockbox has already been opened and must never be reused for model tuning.
This runner therefore freezes a new historical source interval that ends before the
v2 source begins.  The protocol file is written locally *before* any Alpaca request
is made so the date range, universe and split fractions cannot be changed after
seeing the new data.

This is still a retrospective historical lockbox.  A later future-forward period is
required before any production claim or OrionTrader promotion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

from trading.historical_market import (
    HistoricalMarketConfig,
    fetch_and_build_alpaca_dataset,
)


PROTOCOL_NAME = "ourai-v3-nonlinear-research-preregistration-v1"
SOURCE_START = datetime(2025, 10, 1, 0, 0, tzinfo=timezone.utc)
SOURCE_END = datetime(2026, 8, 1, 0, 0, tzinfo=timezone.utc)
PRIOR_V2_SOURCE_START = datetime(2026, 8, 3, 0, 0, tzinfo=timezone.utc)
SYMBOLS = ("NVDA", "AAPL", "TSLA", "JPM", "AMD", "META", "MSFT")
SESSIONS = ("regular",)
HORIZONS_MINUTES = (5, 15, 30, 60)
PRIMARY_RESEARCH_HORIZON_MINUTES = 15
TRAIN_FRACTION = 0.70
VALIDATION_FRACTION = 0.15
FEED = "iex"
ADJUSTMENT = "raw"
MAX_PAGES_PER_BATCH = 100

DatasetFetcher = Callable[..., Mapping[str, Any]]


def _iso_z(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def preregistered_protocol() -> dict[str, Any]:
    if SOURCE_END > PRIOR_V2_SOURCE_START:
        raise RuntimeError("v3 source interval overlaps the already-inspected v2 period")
    payload: dict[str, Any] = {
        "schema_version": 1,
        "protocol_name": PROTOCOL_NAME,
        "source": {
            "provider": "alpaca",
            "feed": FEED,
            "start": _iso_z(SOURCE_START),
            "end": _iso_z(SOURCE_END),
            "timeframe": "1Min",
            "adjustment": ADJUSTMENT,
            "sessions": list(SESSIONS),
            "symbols": list(SYMBOLS),
        },
        "targets": {
            "horizons_minutes": list(HORIZONS_MINUTES),
            "primary_research_horizon_minutes": PRIMARY_RESEARCH_HORIZON_MINUTES,
        },
        "split": {
            "method": "chronological_complete_market_dates",
            "train_fraction": TRAIN_FRACTION,
            "validation_fraction": VALIDATION_FRACTION,
            "test_fraction": 1.0 - TRAIN_FRACTION - VALIDATION_FRACTION,
        },
        "research_safety": {
            "prior_v2_period_may_be_used_for_tuning": False,
            "new_test_may_be_opened_during_model_selection": False,
            "test_open_once_after_challenger_frozen": True,
            "post_test_tuning_on_same_test": False,
            "future_forward_evidence_required_for_production": True,
        },
        "prior_inspected_period": {
            "starts_at": _iso_z(PRIOR_V2_SOURCE_START),
            "overlap_allowed": False,
        },
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    payload["protocol_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return payload


def build_v3_research_dataset(
    output_dir: str | Path,
    *,
    dataset_fetcher: DatasetFetcher = fetch_and_build_alpaca_dataset,
) -> dict[str, Any]:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    protocol = preregistered_protocol()
    protocol_path = destination / "v3_preregistration.json"
    protocol_path.write_text(
        json.dumps(protocol, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    config = HistoricalMarketConfig(
        symbols=SYMBOLS,
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
        raise ValueError("v3 research protocol requires the historical-v2 causal builder")
    source = manifest.get("source") or {}
    if bool(source.get("partial")):
        raise ValueError("v3 dataset must not be built from partial source history")

    summary = {
        "protocol_name": PROTOCOL_NAME,
        "protocol_sha256": protocol["protocol_sha256"],
        "dataset_id": manifest.get("dataset_id"),
        "manifest_sha256": manifest.get("manifest_sha256"),
        "builder": manifest.get("builder"),
        "source_start": _iso_z(SOURCE_START),
        "source_end": _iso_z(SOURCE_END),
        "source_partial": source.get("partial"),
        "source_rows": (manifest.get("source_cache") or {}).get("row_count"),
        "sample_count": (manifest.get("dataset") or {}).get("sample_count"),
        "market_date_count": (manifest.get("dataset") or {}).get("market_date_count"),
        "splits": (manifest.get("dataset") or {}).get("splits"),
        "prior_v2_overlap": False,
        "new_test_status": "UNTOUCHED",
    }
    (destination / "v3_protocol_result.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description=(
            "Build the preregistered OurAI v3 historical dataset from a fixed, "
            "non-overlapping Alpaca interval."
        )
    )
    result.add_argument(
        "--output-dir",
        default="data/processed/market/first-real-dataset-v3",
        help="Local output directory (gitignored).",
    )
    return result


def main() -> int:
    args = parser().parse_args()
    protocol = preregistered_protocol()
    print("=== OURAI V3 PREREGISTRATION ===")
    print("Protocol SHA256:", protocol["protocol_sha256"])
    print("Source:", protocol["source"]["start"], "->", protocol["source"]["end"])
    print("Prior v2 overlap: NO")
    print("New test status: UNTOUCHED")
    print()

    summary = build_v3_research_dataset(args.output_dir)
    print("=== OURAI V3 DATASET ===")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
