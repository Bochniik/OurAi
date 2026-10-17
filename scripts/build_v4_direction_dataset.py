"""Build the preregistered historical dataset for the first nonlinear direction head.

The previously used v2/v3 periods are not reused as a fresh direction lockbox. This
protocol freezes an earlier, non-overlapping one-year Alpaca interval and the exact
15-minute conditional UP-vs-DOWN challenger before any v4 data is fetched.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

from trading.historical_market import HistoricalMarketConfig, fetch_and_build_alpaca_dataset


PROTOCOL_NAME = "ourai-v4-direction-research-preregistration-v1"
SOURCE_START = datetime(2024, 10, 1, 0, 0, tzinfo=timezone.utc)
SOURCE_END = datetime(2025, 10, 1, 0, 0, tzinfo=timezone.utc)
NEXT_KNOWN_RESEARCH_PERIOD_START = datetime(2025, 10, 1, 0, 0, tzinfo=timezone.utc)
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
    if SOURCE_END > NEXT_KNOWN_RESEARCH_PERIOD_START:
        raise RuntimeError("v4 direction source overlaps the previously used v3 period")

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
            "direction_contract": "conditional_up_vs_down",
            "neutral_rows_in_direction_fit_and_metrics": "excluded",
        },
        "split": {
            "method": "chronological_complete_market_dates",
            "train_fraction": TRAIN_FRACTION,
            "validation_fraction": VALIDATION_FRACTION,
            "test_fraction": 1.0 - TRAIN_FRACTION - VALIDATION_FRACTION,
        },
        "frozen_challenger": {
            "family": "HistGradientBoostingClassifier",
            "loss": "log_loss",
            "learning_rate": 0.05,
            "max_iter": 150,
            "max_leaf_nodes": 31,
            "min_samples_leaf": 100,
            "l2_regularization": 1.0,
            "max_bins": 255,
            "early_stopping": False,
            "random_state": 42,
        },
        "linear_incumbent": {
            "family": "shared_logistic_direction_head",
            "probability_l2": 0.001,
            "probability_learning_rate": 0.05,
            "probability_epochs": 120,
            "probability_batch_size": 4096,
            "seed": 42,
        },
        "development_gate": {
            "walk_forward_min_train_market_dates": 55,
            "walk_forward_validation_market_dates": 15,
            "walk_forward_step_market_dates": 15,
            "embargo_minutes": 60.0,
            "expected_fold_count": 8,
            "minimum_brier_fold_wins_vs_linear": 6,
            "minimum_brier_fold_wins_vs_null": 6,
            "require_lower_mean_brier_than_linear": True,
            "require_lower_mean_brier_than_null": True,
            "require_lower_mean_log_loss_than_linear": True,
            "require_lower_mean_log_loss_than_null": True,
            "pnl_used": False,
        },
        "research_safety": {
            "write_preregistration_before_fetch": True,
            "validation_may_be_opened_during_development": False,
            "test_may_be_opened_during_model_selection": False,
            "validation_requires_separate_frozen_gate": True,
            "test_requires_separate_frozen_gate": True,
            "post_test_tuning_on_same_test": False,
            "previous_v2_v3_periods_are_fresh_direction_lockboxes": False,
            "future_forward_evidence_required_for_production": True,
        },
        "non_overlap": {
            "next_known_research_period_starts_at": _iso_z(NEXT_KNOWN_RESEARCH_PERIOD_START),
            "overlap_allowed": False,
        },
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    payload["protocol_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return payload


def build_v4_direction_dataset(
    output_dir: str | Path,
    *,
    dataset_fetcher: DatasetFetcher = fetch_and_build_alpaca_dataset,
) -> dict[str, Any]:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    protocol = preregistered_protocol()
    protocol_path = destination / "v4_direction_preregistration.json"
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
        raise ValueError("v4 direction protocol requires the historical-v2 causal builder")
    source = manifest.get("source") or {}
    if bool(source.get("partial")):
        raise ValueError("v4 direction dataset must not use partial source history")

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
        "overlap_with_v3": False,
        "validation_status": "UNTOUCHED",
        "test_status": "UNTOUCHED",
    }
    (destination / "v4_direction_protocol_result.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Build the preregistered fresh historical dataset for OurAI direction research."
    )
    result.add_argument(
        "--output-dir",
        default="data/processed/market/first-real-direction-v4",
        help="Local output directory (gitignored).",
    )
    return result


def main() -> int:
    args = parser().parse_args()
    protocol = preregistered_protocol()
    print("=== OURAI V4 DIRECTION PREREGISTRATION ===")
    print("Protocol SHA256:", protocol["protocol_sha256"])
    print("Source:", protocol["source"]["start"], "->", protocol["source"]["end"])
    print("Overlap with v3: NO")
    print("Validation status: UNTOUCHED")
    print("Test status: UNTOUCHED")
    print()

    summary = build_v4_direction_dataset(args.output_dir)
    print("=== OURAI V4 DIRECTION DATASET ===")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
