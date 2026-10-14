"""One-shot validation gate for the frozen OurAI v3 nonlinear 15m magnitude challenger.

The real v3 test lockbox is never opened here. The gate first verifies that the
train-only development report passed the preregistered six-fold promotion rule,
then fits the exact frozen nonlinear challenger and linear incumbent on train only
and evaluates them once on validation.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

from scripts.evaluate_magnitude_head import _load_train
from scripts.evaluate_v3_nonlinear_magnitude import (
    EXPECTED_FOLD_COUNT,
    LINEAR_CONFIG,
    MINIMUM_FOLD_WINS_VS_LINEAR,
    MINIMUM_FOLD_WINS_VS_NULL,
    NONLINEAR_CONFIG,
    _validate_v3_contract,
)
from scripts.validate_magnitude_head import _load_validation
from trading.dataset import TradingSample
from trading.magnitude import (
    evaluate_magnitude_predictions,
    fit_unified_linear_magnitude_baseline,
    forecast_magnitudes,
)
from trading.nonlinear import (
    fit_hist_gradient_magnitude,
    forecast_hist_gradient_magnitudes,
)

EXPECTED_DEVELOPMENT_EXPERIMENT = "v3_hist_gradient_absolute_return_magnitude_v1"


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _expected_nonlinear_config() -> dict[str, Any]:
    return {
        "learning_rate": NONLINEAR_CONFIG.learning_rate,
        "max_iter": NONLINEAR_CONFIG.max_iter,
        "max_leaf_nodes": NONLINEAR_CONFIG.max_leaf_nodes,
        "min_samples_leaf": NONLINEAR_CONFIG.min_samples_leaf,
        "l2_regularization": NONLINEAR_CONFIG.l2_regularization,
        "max_bins": NONLINEAR_CONFIG.max_bins,
        "early_stopping": False,
        "random_state": NONLINEAR_CONFIG.random_state,
    }


def _validate_development_report(
    report: Mapping[str, Any],
    *,
    dataset_id: object,
    protocol_name: object,
    protocol_sha256: object,
) -> None:
    if report.get("experiment") != EXPECTED_DEVELOPMENT_EXPERIMENT:
        raise ValueError("unexpected v3 nonlinear development experiment identity")
    if report.get("dataset_id") != dataset_id:
        raise ValueError("development report dataset_id does not match v3 dataset")
    if report.get("protocol_name") != protocol_name:
        raise ValueError("development report protocol_name does not match v3 protocol")
    if report.get("protocol_sha256") != protocol_sha256:
        raise ValueError("development report protocol hash does not match v3 protocol")
    if report.get("original_validation_used") is not False:
        raise ValueError("development report must not have used validation")
    if report.get("test_lockbox_accessed") is not False:
        raise ValueError("development report must not have accessed v3 test lockbox")

    nonlinear = _mapping(report.get("nonlinear_model"), "nonlinear_model")
    if nonlinear.get("family") != "HistGradientBoostingRegressor":
        raise ValueError("unexpected nonlinear model family")
    if nonlinear.get("loss") != "absolute_error":
        raise ValueError("unexpected nonlinear loss")
    if dict(_mapping(nonlinear.get("config"), "nonlinear_model.config")) != _expected_nonlinear_config():
        raise ValueError("development report does not match frozen nonlinear configuration")

    incumbent = _mapping(report.get("incumbent"), "incumbent")
    if incumbent.get("family") != "ridge_absolute_return":
        raise ValueError("unexpected linear incumbent family")
    if float(incumbent.get("regression_l2", -1.0)) != LINEAR_CONFIG.regression_l2:
        raise ValueError("unexpected linear incumbent configuration")

    walk_forward = _mapping(report.get("walk_forward"), "walk_forward")
    if int(walk_forward.get("expected_fold_count", -1)) != EXPECTED_FOLD_COUNT:
        raise ValueError("unexpected development fold count")

    promotion = _mapping(report.get("promotion_rule"), "promotion_rule")
    if int(promotion.get("minimum_fold_wins_vs_linear", -1)) != MINIMUM_FOLD_WINS_VS_LINEAR:
        raise ValueError("unexpected development win rule vs linear")
    if int(promotion.get("minimum_fold_wins_vs_null", -1)) != MINIMUM_FOLD_WINS_VS_NULL:
        raise ValueError("unexpected development win rule vs null")
    if promotion.get("require_lower_mean_mae_than_linear") is not True:
        raise ValueError("development report must require lower mean MAE than linear")
    if promotion.get("require_lower_mean_mae_than_null") is not True:
        raise ValueError("development report must require lower mean MAE than null")
    if promotion.get("validation_split_used_for_tuning") is not False:
        raise ValueError("development report must keep validation outside tuning")
    if promotion.get("pnl_used") is not False:
        raise ValueError("development promotion must not use PnL")

    summary = _mapping(report.get("summary"), "summary")
    if int(summary.get("fold_count", -1)) != EXPECTED_FOLD_COUNT:
        raise ValueError("development report does not contain six folds")
    if int(summary.get("fold_wins_vs_linear", -1)) < MINIMUM_FOLD_WINS_VS_LINEAR:
        raise ValueError("nonlinear challenger failed fold-win gate vs linear")
    if int(summary.get("fold_wins_vs_null", -1)) < MINIMUM_FOLD_WINS_VS_NULL:
        raise ValueError("nonlinear challenger failed fold-win gate vs null")
    if not float(summary.get("mean_nonlinear_mae_bps", float("inf"))) < float(
        summary.get("mean_linear_mae_bps", float("-inf"))
    ):
        raise ValueError("nonlinear challenger failed mean MAE gate vs linear")
    if not float(summary.get("mean_nonlinear_mae_bps", float("inf"))) < float(
        summary.get("mean_null_mae_bps", float("-inf"))
    ):
        raise ValueError("nonlinear challenger failed mean MAE gate vs null")
    if summary.get("nonlinear_challenger_supported") is not True:
        raise ValueError("development nonlinear challenger was not supported")


def _evaluate_pair(
    nonlinear_model: object,
    linear_model: object,
    samples: list[TradingSample],
    *,
    null_magnitude_bps: float,
) -> dict[str, Any]:
    nonlinear_predictions = forecast_hist_gradient_magnitudes(
        nonlinear_model, samples, include_actual=True
    )
    linear_predictions = forecast_magnitudes(linear_model, samples, include_actual=True)
    nonlinear_metrics = evaluate_magnitude_predictions(
        nonlinear_predictions, null_magnitude_bps=null_magnitude_bps
    )
    linear_metrics = evaluate_magnitude_predictions(
        linear_predictions, null_magnitude_bps=null_magnitude_bps
    )
    nonlinear_mae = float(nonlinear_metrics["model_mae_bps"])
    linear_mae = float(linear_metrics["model_mae_bps"])
    null_mae = float(nonlinear_metrics["null_mae_bps"])
    return {
        "evaluated_count": int(nonlinear_metrics["evaluated_count"]),
        "nonlinear_mae_bps": nonlinear_mae,
        "linear_mae_bps": linear_mae,
        "null_mae_bps": null_mae,
        "delta_vs_linear_bps": nonlinear_mae - linear_mae,
        "delta_vs_null_bps": nonlinear_mae - null_mae,
        "wins_vs_linear": nonlinear_mae < linear_mae,
        "wins_vs_null": nonlinear_mae < null_mae,
    }


def _per_market_date_metrics(
    nonlinear_model: object,
    linear_model: object,
    samples: list[TradingSample],
    *,
    null_magnitude_bps: float,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[TradingSample]] = defaultdict(list)
    for sample in samples:
        grouped[sample.session_date].append(sample)
    output: list[dict[str, Any]] = []
    for market_date in sorted(grouped):
        row = _evaluate_pair(
            nonlinear_model,
            linear_model,
            grouped[market_date],
            null_magnitude_bps=null_magnitude_bps,
        )
        row["market_date"] = market_date
        output.append(row)
    return output


def run_validation_gate(
    dataset_dir: str | Path,
    development_report_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    root = Path(dataset_dir)
    manifest, protocol_result = _validate_v3_contract(root)
    train_path = root / "train.jsonl"
    validation_path = root / "validation.jsonl"
    if not train_path.is_file():
        raise FileNotFoundError(f"train split not found: {train_path}")
    if not validation_path.is_file():
        raise FileNotFoundError(f"validation split not found: {validation_path}")

    development_report = json.loads(
        Path(development_report_path).read_text(encoding="utf-8")
    )
    if not isinstance(development_report, Mapping):
        raise ValueError("development report must contain a JSON object")
    _validate_development_report(
        development_report,
        dataset_id=manifest.get("dataset_id"),
        protocol_name=protocol_result.get("protocol_name"),
        protocol_sha256=protocol_result.get("protocol_sha256"),
    )

    # Validation is opened only after the frozen development gate is verified.
    # test.jsonl is intentionally never referenced or opened in this module.
    training_samples = _load_train(train_path)
    validation_samples = _load_validation(validation_path)

    nonlinear_model, nonlinear_fit = fit_hist_gradient_magnitude(
        training_samples, config=NONLINEAR_CONFIG
    )
    linear_model, linear_fit = fit_unified_linear_magnitude_baseline(
        training_samples, config=LINEAR_CONFIG
    )
    null_bps = linear_fit.training_median_magnitude_bps
    overall = _evaluate_pair(
        nonlinear_model,
        linear_model,
        validation_samples,
        null_magnitude_bps=null_bps,
    )
    per_date = _per_market_date_metrics(
        nonlinear_model,
        linear_model,
        validation_samples,
        null_magnitude_bps=null_bps,
    )

    date_count = len(per_date)
    wins_vs_linear = sum(bool(row["wins_vs_linear"]) for row in per_date)
    wins_vs_null = sum(bool(row["wins_vs_null"]) for row in per_date)
    strict_majority_vs_linear = wins_vs_linear > date_count / 2.0
    strict_majority_vs_null = wins_vs_null > date_count / 2.0
    supported = bool(
        overall["wins_vs_linear"]
        and overall["wins_vs_null"]
        and strict_majority_vs_linear
        and strict_majority_vs_null
    )

    relative_vs_linear = None
    if overall["linear_mae_bps"] > 0:
        relative_vs_linear = (
            overall["linear_mae_bps"] - overall["nonlinear_mae_bps"]
        ) / overall["linear_mae_bps"]
    relative_vs_null = None
    if overall["null_mae_bps"] > 0:
        relative_vs_null = (
            overall["null_mae_bps"] - overall["nonlinear_mae_bps"]
        ) / overall["null_mae_bps"]

    payload = {
        "schema_version": 1,
        "experiment": "v3_hist_gradient_absolute_return_magnitude_validation_v1",
        "dataset_id": manifest.get("dataset_id"),
        "dataset_builder": manifest.get("builder"),
        "protocol_name": protocol_result.get("protocol_name"),
        "protocol_sha256": protocol_result.get("protocol_sha256"),
        "development_report": str(development_report_path),
        "horizon_minutes": NONLINEAR_CONFIG.horizon_minutes,
        "frozen_nonlinear_model": {
            "family": "HistGradientBoostingRegressor",
            "loss": "absolute_error",
            "config": _expected_nonlinear_config(),
        },
        "incumbent": {
            "family": "ridge_absolute_return",
            "regression_l2": LINEAR_CONFIG.regression_l2,
        },
        "validation_rule": {
            "overall_nonlinear_mae_must_beat_linear": True,
            "overall_nonlinear_mae_must_beat_train_median_null": True,
            "require_strict_majority_market_date_wins_vs_linear": True,
            "require_strict_majority_market_date_wins_vs_null": True,
            "tuning_after_validation": False,
            "pnl_used": False,
        },
        "fit_summary": {
            "nonlinear": nonlinear_fit.to_dict(),
            "linear": linear_fit.to_dict(),
        },
        "validation": {
            "overall": overall,
            "market_dates": per_date,
            "market_date_count": date_count,
            "market_date_wins_vs_linear": wins_vs_linear,
            "market_date_wins_vs_null": wins_vs_null,
            "relative_mae_improvement_vs_linear": relative_vs_linear,
            "relative_mae_improvement_vs_null": relative_vs_null,
        },
        "validation_supported": supported,
        "original_validation_used": True,
        "test_lockbox_accessed": False,
    }

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return payload


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description=(
            "Run the one-shot frozen v3 nonlinear 15m magnitude validation gate. "
            "The v3 test lockbox is never opened."
        )
    )
    result.add_argument("--dataset-dir", required=True)
    result.add_argument("--development-report", required=True)
    result.add_argument(
        "--output",
        default="runs/first-real-15m-v3/nonlinear_magnitude_validation_v1.json",
    )
    return result


def main() -> int:
    args = parser().parse_args()
    payload = run_validation_gate(
        args.dataset_dir, args.development_report, args.output
    )
    validation = payload["validation"]
    overall = validation["overall"]
    print("=== V3 NONLINEAR MAGNITUDE VALIDATION ===")
    print("Dataset ID:", payload["dataset_id"])
    print("Validation samples:", overall["evaluated_count"])
    print("Nonlinear MAE bps:", overall["nonlinear_mae_bps"])
    print("Linear MAE bps:", overall["linear_mae_bps"])
    print("Null MAE bps:", overall["null_mae_bps"])
    print("Delta vs linear bps:", overall["delta_vs_linear_bps"])
    print("Delta vs null bps:", overall["delta_vs_null_bps"])
    print(
        "Market-date wins vs linear:",
        f"{validation['market_date_wins_vs_linear']}/{validation['market_date_count']}",
    )
    print(
        "Market-date wins vs null:",
        f"{validation['market_date_wins_vs_null']}/{validation['market_date_count']}",
    )
    print(
        "Relative MAE improvement vs linear:",
        validation["relative_mae_improvement_vs_linear"],
    )
    print(
        "Relative MAE improvement vs null:",
        validation["relative_mae_improvement_vs_null"],
    )
    print("VALIDATION SUPPORTED:", payload["validation_supported"])
    print("Report:", args.output)
    print("V3 test lockbox: NOT ACCESSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
