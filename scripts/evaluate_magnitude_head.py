"""Evaluate the first absolute-return magnitude challenger on training data only.

This script is intentionally narrow and preregistered for the current research step:

* input is one historical-v2 dataset directory;
* only ``manifest.json`` and ``train.jsonl`` are read;
* 15-minute expanding purged walk-forward folds are fixed at 18/4/4 market dates;
* ridge regression predicts ``|future_return|``;
* the null predicts the training-fold median absolute return, which is the proper
  constant comparator for MAE;
* promotion requires lower mean MAE and MAE wins on at least 3 of 4 folds;
* PnL is not computed and is not a promotion criterion.

Validation and test split files are never opened by this script.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from statistics import fmean
from typing import Any, Mapping

from trading.baseline import BaselineConfig
from trading.dataset import TradingSample
from trading.magnitude import (
    evaluate_magnitude_predictions,
    fit_unified_linear_magnitude_baseline,
    forecast_magnitudes,
)
from trading.validation import purged_walk_forward_folds, walk_forward_manifest


HORIZON_MINUTES = 15
MIN_TRAIN_MARKET_DATES = 18
VALIDATION_MARKET_DATES = 4
STEP_MARKET_DATES = 4
EMBARGO_MINUTES = 60.0
PROMOTION_MIN_FOLD_WINS = 3
EXPECTED_BUILDER = "ourai-historical-market-v2"
EXPECTED_AVAILABILITY_DELAY_MINUTES = 1.0

MODEL_CONFIG = BaselineConfig(
    horizon_minutes=HORIZON_MINUTES,
    probability_l2=0.001,
    probability_learning_rate=0.05,
    probability_epochs=120,
    probability_batch_size=4096,
    regression_l2=1.0,
    minimum_training_samples=40,
    seed=42,
)


def _parse_timestamp(value: object) -> datetime:
    timestamp = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ValueError("sample timestamp must be timezone-aware")
    return timestamp


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"manifest field {name!r} must be an object")
    return value


def _load_train(path: Path) -> list[TradingSample]:
    if path.name != "train.jsonl":
        raise ValueError("magnitude experiment accepts train.jsonl only")
    rows: list[TradingSample] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                rows.append(
                    TradingSample(
                        symbol=row["symbol"],
                        timestamp=_parse_timestamp(row["timestamp"]),
                        session_date=row["session_date"],
                        session=row["session"],
                        features=row.get("features") or {},
                        targets=row.get("targets") or {},
                    )
                )
            except Exception as exc:
                raise ValueError(f"invalid train row at line {line_number}") from exc
    if not rows:
        raise ValueError("train.jsonl is empty")
    return rows


def _validate_dataset_manifest(manifest: Mapping[str, Any]) -> None:
    if manifest.get("builder") != EXPECTED_BUILDER:
        raise ValueError(
            f"expected builder {EXPECTED_BUILDER!r}, got {manifest.get('builder')!r}"
        )
    configuration = _mapping(manifest.get("configuration"), "configuration")
    delay = float(configuration.get("bar_availability_delay_minutes", -1.0))
    if delay != EXPECTED_AVAILABILITY_DELAY_MINUTES:
        raise ValueError(
            "expected 1-minute bar availability delay for historical-v2 dataset"
        )


def run_experiment(dataset_dir: str | Path, output_path: str | Path) -> dict[str, Any]:
    root = Path(dataset_dir)
    manifest_path = root / "manifest.json"
    train_path = root / "train.jsonl"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    if not train_path.is_file():
        raise FileNotFoundError(f"train split not found: {train_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, Mapping):
        raise ValueError("manifest must contain a JSON object")
    _validate_dataset_manifest(manifest)

    samples = _load_train(train_path)
    folds = purged_walk_forward_folds(
        samples,
        min_train_market_dates=MIN_TRAIN_MARKET_DATES,
        validation_market_dates=VALIDATION_MARKET_DATES,
        step_market_dates=STEP_MARKET_DATES,
        embargo_minutes=EMBARGO_MINUTES,
        max_train_market_dates=None,
    )

    fold_results: list[dict[str, Any]] = []
    for fold in folds:
        model, fit_summary = fit_unified_linear_magnitude_baseline(
            fold.train,
            config=MODEL_CONFIG,
        )
        predictions = forecast_magnitudes(
            model,
            fold.validation,
            include_actual=True,
        )
        metrics = evaluate_magnitude_predictions(
            predictions,
            null_magnitude_bps=fit_summary.training_median_magnitude_bps,
        )
        fold_results.append(
            {
                "fold": fold.index,
                "train_market_dates": list(fold.train_market_dates),
                "validation_market_dates": list(fold.validation_market_dates),
                "train_samples": len(fold.train),
                "validation_samples": len(fold.validation),
                "purged_overlap_count": fold.purged_overlap_count,
                "embargoed_count": fold.embargoed_count,
                "fit_summary": fit_summary.to_dict(),
                "metrics": metrics,
            }
        )

    model_maes = [row["metrics"]["model_mae_bps"] for row in fold_results]
    null_maes = [row["metrics"]["null_mae_bps"] for row in fold_results]
    fold_wins = sum(bool(row["metrics"]["model_wins_mae"]) for row in fold_results)
    mean_model_mae = fmean(model_maes)
    mean_null_mae = fmean(null_maes)
    supported = (
        fold_wins >= PROMOTION_MIN_FOLD_WINS
        and mean_model_mae < mean_null_mae
    )

    payload = {
        "schema_version": 1,
        "experiment": "absolute_return_linear_magnitude_v1",
        "dataset_id": manifest.get("dataset_id"),
        "dataset_builder": manifest.get("builder"),
        "horizon_minutes": HORIZON_MINUTES,
        "source": "train_split_only",
        "original_validation_used": False,
        "test_lockbox_accessed": False,
        "model": {
            "family": "ridge_absolute_return",
            "config": {
                "regression_l2": MODEL_CONFIG.regression_l2,
                "minimum_training_samples": MODEL_CONFIG.minimum_training_samples,
                "negative_prediction_policy": "clip_to_zero",
            },
        },
        "null": {
            "family": "training_fold_median_absolute_return",
            "reason": "median_is_optimal_constant_for_mae",
        },
        "walk_forward": {
            "min_train_market_dates": MIN_TRAIN_MARKET_DATES,
            "validation_market_dates": VALIDATION_MARKET_DATES,
            "step_market_dates": STEP_MARKET_DATES,
            "embargo_minutes": EMBARGO_MINUTES,
            "manifest": walk_forward_manifest(folds),
        },
        "promotion_rule": {
            "metric": "mae_bps",
            "minimum_fold_wins": PROMOTION_MIN_FOLD_WINS,
            "require_lower_mean_mae": True,
            "pnl_used": False,
        },
        "fold_results": fold_results,
        "summary": {
            "fold_count": len(fold_results),
            "fold_wins": fold_wins,
            "mean_model_mae_bps": mean_model_mae,
            "mean_null_mae_bps": mean_null_mae,
            "mean_mae_delta_bps": mean_model_mae - mean_null_mae,
            "magnitude_challenger_supported": supported,
        },
    }

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description=(
            "Run the preregistered train-only 15m absolute-return magnitude challenger. "
            "Validation/test split files are not read."
        )
    )
    result.add_argument(
        "--dataset-dir",
        required=True,
        help="Historical-v2 dataset directory containing manifest.json and train.jsonl.",
    )
    result.add_argument(
        "--output",
        default="runs/first-real-15m-v2/magnitude_head_linear_v1.json",
        help="Research report output path.",
    )
    return result


def main() -> int:
    args = parser().parse_args()
    payload = run_experiment(args.dataset_dir, args.output)

    for row in payload["fold_results"]:
        metrics = row["metrics"]
        dates = row["validation_market_dates"]
        print(
            f"FOLD {row['fold']} {dates[0]} -> {dates[-1]} "
            f"model_MAE={metrics['model_mae_bps']:.6f} "
            f"null_MAE={metrics['null_mae_bps']:.6f} "
            f"delta={metrics['mae_delta_bps']:+.6f} "
            f"win={metrics['model_wins_mae']}"
        )

    summary = payload["summary"]
    print()
    print("=== MAGNITUDE HEAD SUMMARY ===")
    print("Dataset ID:", payload["dataset_id"])
    print("Fold wins:", f"{summary['fold_wins']}/{summary['fold_count']}")
    print("Mean model MAE bps:", summary["mean_model_mae_bps"])
    print("Mean null MAE bps:", summary["mean_null_mae_bps"])
    print("Mean delta bps:", summary["mean_mae_delta_bps"])
    print("MAGNITUDE CHALLENGER SUPPORTED:", summary["magnitude_challenger_supported"])
    print("Report:", args.output)
    print("Original validation: NOT ACCESSED")
    print("Test lockbox: NOT ACCESSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
