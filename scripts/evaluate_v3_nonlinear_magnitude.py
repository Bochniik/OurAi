"""Preregistered train-only development experiment for the OurAI v3 nonlinear magnitude challenger.

The v3 test lockbox is never opened here. Development uses only ``train.jsonl``
and six fixed purged walk-forward folds. The nonlinear histogram-gradient model must
beat both the linear magnitude incumbent and a train-fold median null before it may
advance to the separate validation split.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import fmean
from typing import Any, Mapping

from scripts.build_v3_research_dataset import preregistered_protocol
from scripts.evaluate_magnitude_head import _load_train
from trading.baseline import BaselineConfig
from trading.magnitude import (
    evaluate_magnitude_predictions,
    fit_unified_linear_magnitude_baseline,
    forecast_magnitudes,
)
from trading.nonlinear import (
    HistGradientMagnitudeConfig,
    fit_hist_gradient_magnitude,
    forecast_hist_gradient_magnitudes,
)
from trading.validation import purged_walk_forward_folds, walk_forward_manifest


HORIZON_MINUTES = 15
MIN_TRAIN_MARKET_DATES = 55
VALIDATION_MARKET_DATES = 15
STEP_MARKET_DATES = 15
EMBARGO_MINUTES = 60.0
EXPECTED_FOLD_COUNT = 6
MINIMUM_FOLD_WINS_VS_LINEAR = 4
MINIMUM_FOLD_WINS_VS_NULL = 4
EXPECTED_BUILDER = "ourai-historical-market-v2"

NONLINEAR_CONFIG = HistGradientMagnitudeConfig(
    horizon_minutes=HORIZON_MINUTES,
    learning_rate=0.05,
    max_iter=150,
    max_leaf_nodes=31,
    min_samples_leaf=100,
    l2_regularization=1.0,
    max_bins=255,
    random_state=42,
    minimum_training_samples=100,
)
LINEAR_CONFIG = BaselineConfig(
    horizon_minutes=HORIZON_MINUTES,
    regression_l2=1.0,
    minimum_training_samples=40,
    seed=42,
)


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _validate_v3_contract(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest_path = root / "manifest.json"
    protocol_path = root / "v3_preregistration.json"
    result_path = root / "v3_protocol_result.json"
    for path in (manifest_path, protocol_path, result_path):
        if not path.is_file():
            raise FileNotFoundError(f"required v3 artifact not found: {path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, Mapping) or not isinstance(protocol, Mapping) or not isinstance(result, Mapping):
        raise ValueError("v3 dataset artifacts must contain JSON objects")

    expected = preregistered_protocol()
    if protocol != expected:
        raise ValueError("v3 preregistration does not match the committed research protocol")
    if result.get("protocol_sha256") != expected["protocol_sha256"]:
        raise ValueError("v3 protocol result hash does not match preregistration")
    if result.get("new_test_status") != "UNTOUCHED":
        raise ValueError("v3 test lockbox is not marked UNTOUCHED")
    if result.get("prior_v2_overlap") is not False:
        raise ValueError("v3 dataset overlaps the prior inspected v2 period")
    if manifest.get("builder") != EXPECTED_BUILDER:
        raise ValueError("v3 nonlinear experiment requires historical-v2 causal builder")
    if result.get("dataset_id") != manifest.get("dataset_id"):
        raise ValueError("v3 protocol result dataset_id does not match manifest")
    source = _mapping(manifest.get("source"), "manifest.source")
    if bool(source.get("partial")):
        raise ValueError("partial source history is not allowed")
    return dict(manifest), dict(result)


def run_experiment(dataset_dir: str | Path, output_path: str | Path) -> dict[str, Any]:
    root = Path(dataset_dir)
    manifest, protocol_result = _validate_v3_contract(root)
    train_path = root / "train.jsonl"
    if not train_path.is_file():
        raise FileNotFoundError(f"train split not found: {train_path}")

    # Important: validation.jsonl and test.jsonl are intentionally never opened here.
    samples = _load_train(train_path)
    folds = purged_walk_forward_folds(
        samples,
        min_train_market_dates=MIN_TRAIN_MARKET_DATES,
        validation_market_dates=VALIDATION_MARKET_DATES,
        step_market_dates=STEP_MARKET_DATES,
        embargo_minutes=EMBARGO_MINUTES,
        max_train_market_dates=None,
    )
    if len(folds) != EXPECTED_FOLD_COUNT:
        raise ValueError(
            f"preregistered v3 nonlinear experiment requires exactly {EXPECTED_FOLD_COUNT} folds, got {len(folds)}"
        )

    fold_results: list[dict[str, Any]] = []
    for fold in folds:
        nonlinear_model, nonlinear_fit = fit_hist_gradient_magnitude(
            fold.train,
            config=NONLINEAR_CONFIG,
        )
        linear_model, linear_fit = fit_unified_linear_magnitude_baseline(
            fold.train,
            config=LINEAR_CONFIG,
        )
        nonlinear_predictions = forecast_hist_gradient_magnitudes(
            nonlinear_model,
            fold.validation,
            include_actual=True,
        )
        linear_predictions = forecast_magnitudes(
            linear_model,
            fold.validation,
            include_actual=True,
        )
        null_bps = linear_fit.training_median_magnitude_bps
        nonlinear_metrics = evaluate_magnitude_predictions(
            nonlinear_predictions,
            null_magnitude_bps=null_bps,
        )
        linear_metrics = evaluate_magnitude_predictions(
            linear_predictions,
            null_magnitude_bps=null_bps,
        )
        nonlinear_mae = float(nonlinear_metrics["model_mae_bps"])
        linear_mae = float(linear_metrics["model_mae_bps"])
        null_mae = float(nonlinear_metrics["null_mae_bps"])
        fold_results.append(
            {
                "fold": fold.index,
                "train_market_dates": list(fold.train_market_dates),
                "validation_market_dates": list(fold.validation_market_dates),
                "train_samples": len(fold.train),
                "validation_samples": len(fold.validation),
                "purged_overlap_count": fold.purged_overlap_count,
                "embargoed_count": fold.embargoed_count,
                "nonlinear_fit_summary": nonlinear_fit.to_dict(),
                "nonlinear_mae_bps": nonlinear_mae,
                "linear_mae_bps": linear_mae,
                "null_mae_bps": null_mae,
                "delta_vs_linear_bps": nonlinear_mae - linear_mae,
                "delta_vs_null_bps": nonlinear_mae - null_mae,
                "wins_vs_linear": nonlinear_mae < linear_mae,
                "wins_vs_null": nonlinear_mae < null_mae,
            }
        )

    nonlinear_maes = [row["nonlinear_mae_bps"] for row in fold_results]
    linear_maes = [row["linear_mae_bps"] for row in fold_results]
    null_maes = [row["null_mae_bps"] for row in fold_results]
    wins_vs_linear = sum(bool(row["wins_vs_linear"]) for row in fold_results)
    wins_vs_null = sum(bool(row["wins_vs_null"]) for row in fold_results)
    mean_nonlinear = fmean(nonlinear_maes)
    mean_linear = fmean(linear_maes)
    mean_null = fmean(null_maes)
    supported = bool(
        wins_vs_linear >= MINIMUM_FOLD_WINS_VS_LINEAR
        and wins_vs_null >= MINIMUM_FOLD_WINS_VS_NULL
        and mean_nonlinear < mean_linear
        and mean_nonlinear < mean_null
    )

    payload = {
        "schema_version": 1,
        "experiment": "v3_hist_gradient_absolute_return_magnitude_v1",
        "dataset_id": manifest.get("dataset_id"),
        "dataset_builder": manifest.get("builder"),
        "protocol_name": protocol_result.get("protocol_name"),
        "protocol_sha256": protocol_result.get("protocol_sha256"),
        "horizon_minutes": HORIZON_MINUTES,
        "source": "train_split_only",
        "original_validation_used": False,
        "test_lockbox_accessed": False,
        "nonlinear_model": {
            "family": "HistGradientBoostingRegressor",
            "loss": "absolute_error",
            "config": {
                "learning_rate": NONLINEAR_CONFIG.learning_rate,
                "max_iter": NONLINEAR_CONFIG.max_iter,
                "max_leaf_nodes": NONLINEAR_CONFIG.max_leaf_nodes,
                "min_samples_leaf": NONLINEAR_CONFIG.min_samples_leaf,
                "l2_regularization": NONLINEAR_CONFIG.l2_regularization,
                "max_bins": NONLINEAR_CONFIG.max_bins,
                "early_stopping": False,
                "random_state": NONLINEAR_CONFIG.random_state,
            },
        },
        "incumbent": {
            "family": "ridge_absolute_return",
            "regression_l2": LINEAR_CONFIG.regression_l2,
        },
        "walk_forward": {
            "min_train_market_dates": MIN_TRAIN_MARKET_DATES,
            "validation_market_dates": VALIDATION_MARKET_DATES,
            "step_market_dates": STEP_MARKET_DATES,
            "embargo_minutes": EMBARGO_MINUTES,
            "expected_fold_count": EXPECTED_FOLD_COUNT,
            "manifest": walk_forward_manifest(folds),
        },
        "promotion_rule": {
            "minimum_fold_wins_vs_linear": MINIMUM_FOLD_WINS_VS_LINEAR,
            "minimum_fold_wins_vs_null": MINIMUM_FOLD_WINS_VS_NULL,
            "require_lower_mean_mae_than_linear": True,
            "require_lower_mean_mae_than_null": True,
            "validation_split_used_for_tuning": False,
            "pnl_used": False,
        },
        "fold_results": fold_results,
        "summary": {
            "fold_count": len(fold_results),
            "fold_wins_vs_linear": wins_vs_linear,
            "fold_wins_vs_null": wins_vs_null,
            "mean_nonlinear_mae_bps": mean_nonlinear,
            "mean_linear_mae_bps": mean_linear,
            "mean_null_mae_bps": mean_null,
            "mean_delta_vs_linear_bps": mean_nonlinear - mean_linear,
            "mean_delta_vs_null_bps": mean_nonlinear - mean_null,
            "nonlinear_challenger_supported": supported,
        },
    }

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description=(
            "Run the preregistered train-only OurAI v3 nonlinear 15m magnitude challenger. "
            "Validation and test files are never opened."
        )
    )
    result.add_argument("--dataset-dir", required=True)
    result.add_argument(
        "--output",
        default="runs/first-real-15m-v3/nonlinear_magnitude_development_v1.json",
    )
    return result


def main() -> int:
    args = parser().parse_args()
    payload = run_experiment(args.dataset_dir, args.output)
    for row in payload["fold_results"]:
        dates = row["validation_market_dates"]
        print(
            f"FOLD {row['fold']} {dates[0]} -> {dates[-1]} "
            f"nonlinear={row['nonlinear_mae_bps']:.6f} "
            f"linear={row['linear_mae_bps']:.6f} "
            f"null={row['null_mae_bps']:.6f} "
            f"d_linear={row['delta_vs_linear_bps']:+.6f} "
            f"d_null={row['delta_vs_null_bps']:+.6f}"
        )

    summary = payload["summary"]
    print()
    print("=== V3 NONLINEAR MAGNITUDE DEVELOPMENT ===")
    print("Dataset ID:", payload["dataset_id"])
    print("Fold wins vs linear:", f"{summary['fold_wins_vs_linear']}/{summary['fold_count']}")
    print("Fold wins vs null:", f"{summary['fold_wins_vs_null']}/{summary['fold_count']}")
    print("Mean nonlinear MAE bps:", summary["mean_nonlinear_mae_bps"])
    print("Mean linear MAE bps:", summary["mean_linear_mae_bps"])
    print("Mean null MAE bps:", summary["mean_null_mae_bps"])
    print("Mean delta vs linear bps:", summary["mean_delta_vs_linear_bps"])
    print("Mean delta vs null bps:", summary["mean_delta_vs_null_bps"])
    print("NONLINEAR CHALLENGER SUPPORTED:", summary["nonlinear_challenger_supported"])
    print("Report:", args.output)
    print("Original validation: NOT ACCESSED")
    print("V3 test lockbox: NOT ACCESSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
