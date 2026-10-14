"""One-shot final test lockbox for the frozen OurAI v3 nonlinear 15m magnitude challenger.

The v3 test split is opened only after both the preregistered train-only development
report and the frozen validation report have passed. The exact nonlinear and linear
configurations are reused without tuning. Final fitting uses train + validation,
then the untouched test split is evaluated exactly once.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from scripts.evaluate_magnitude_head import _load_train
from scripts.evaluate_v3_nonlinear_magnitude import (
    LINEAR_CONFIG,
    NONLINEAR_CONFIG,
    _validate_v3_contract,
)
from scripts.test_magnitude_lockbox import _load_test
from scripts.validate_magnitude_head import _load_validation
from scripts.validate_v3_nonlinear_magnitude import (
    EXPECTED_DEVELOPMENT_EXPERIMENT,
    _evaluate_pair,
    _expected_nonlinear_config,
    _per_market_date_metrics,
    _validate_development_report,
)
from trading.magnitude import fit_unified_linear_magnitude_baseline
from trading.nonlinear import fit_hist_gradient_magnitude

EXPECTED_VALIDATION_EXPERIMENT = "v3_hist_gradient_absolute_return_magnitude_validation_v1"


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _validate_validation_report(
    report: Mapping[str, Any],
    *,
    dataset_id: object,
    protocol_name: object,
    protocol_sha256: object,
    development_report_path: str | Path,
) -> None:
    if report.get("experiment") != EXPECTED_VALIDATION_EXPERIMENT:
        raise ValueError("unexpected v3 nonlinear validation experiment identity")
    if report.get("dataset_id") != dataset_id:
        raise ValueError("validation report dataset_id does not match v3 dataset")
    if report.get("protocol_name") != protocol_name:
        raise ValueError("validation report protocol_name does not match v3 protocol")
    if report.get("protocol_sha256") != protocol_sha256:
        raise ValueError("validation report protocol hash does not match v3 protocol")
    if report.get("development_report") != str(development_report_path):
        raise ValueError("validation report references a different development report")
    if report.get("validation_supported") is not True:
        raise ValueError("v3 nonlinear challenger failed validation gate")
    if report.get("original_validation_used") is not True:
        raise ValueError("validation report must have opened validation exactly once")
    if report.get("test_lockbox_accessed") is not False:
        raise ValueError("validation report must not have accessed the v3 test lockbox")

    frozen = _mapping(report.get("frozen_nonlinear_model"), "frozen_nonlinear_model")
    if frozen.get("family") != "HistGradientBoostingRegressor":
        raise ValueError("unexpected frozen nonlinear model family")
    if frozen.get("loss") != "absolute_error":
        raise ValueError("unexpected frozen nonlinear loss")
    if dict(_mapping(frozen.get("config"), "frozen_nonlinear_model.config")) != _expected_nonlinear_config():
        raise ValueError("validation report changed the frozen nonlinear configuration")

    incumbent = _mapping(report.get("incumbent"), "incumbent")
    if incumbent.get("family") != "ridge_absolute_return":
        raise ValueError("unexpected validation linear incumbent family")
    if float(incumbent.get("regression_l2", -1.0)) != LINEAR_CONFIG.regression_l2:
        raise ValueError("validation report changed the frozen linear configuration")

    rule = _mapping(report.get("validation_rule"), "validation_rule")
    expected_true = (
        "overall_nonlinear_mae_must_beat_linear",
        "overall_nonlinear_mae_must_beat_train_median_null",
        "require_strict_majority_market_date_wins_vs_linear",
        "require_strict_majority_market_date_wins_vs_null",
    )
    for key in expected_true:
        if rule.get(key) is not True:
            raise ValueError(f"validation report used an unexpected rule for {key}")
    if rule.get("tuning_after_validation") is not False:
        raise ValueError("validation report must prohibit post-validation tuning")
    if rule.get("pnl_used") is not False:
        raise ValueError("validation report must not use PnL")


def run_test_lockbox_gate(
    dataset_dir: str | Path,
    development_report_path: str | Path,
    validation_report_path: str | Path,
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

    development_report = json.loads(Path(development_report_path).read_text(encoding="utf-8"))
    if not isinstance(development_report, Mapping):
        raise ValueError("development report must contain a JSON object")
    _validate_development_report(
        development_report,
        dataset_id=manifest.get("dataset_id"),
        protocol_name=protocol_result.get("protocol_name"),
        protocol_sha256=protocol_result.get("protocol_sha256"),
    )
    if development_report.get("experiment") != EXPECTED_DEVELOPMENT_EXPERIMENT:
        raise ValueError("unexpected development experiment identity")

    validation_report = json.loads(Path(validation_report_path).read_text(encoding="utf-8"))
    if not isinstance(validation_report, Mapping):
        raise ValueError("validation report must contain a JSON object")
    _validate_validation_report(
        validation_report,
        dataset_id=manifest.get("dataset_id"),
        protocol_name=protocol_result.get("protocol_name"),
        protocol_sha256=protocol_result.get("protocol_sha256"),
        development_report_path=development_report_path,
    )

    # Only after both frozen pre-test gates pass do we even require/open test.jsonl.
    test_path = root / "test.jsonl"
    if not test_path.is_file():
        raise FileNotFoundError(f"test split not found: {test_path}")

    train_samples = _load_train(train_path)
    validation_samples = _load_validation(validation_path)
    final_training_samples = [*train_samples, *validation_samples]

    nonlinear_model, nonlinear_fit = fit_hist_gradient_magnitude(
        final_training_samples, config=NONLINEAR_CONFIG
    )
    linear_model, linear_fit = fit_unified_linear_magnitude_baseline(
        final_training_samples, config=LINEAR_CONFIG
    )
    null_bps = linear_fit.training_median_magnitude_bps

    test_samples = _load_test(test_path)
    overall = _evaluate_pair(
        nonlinear_model,
        linear_model,
        test_samples,
        null_magnitude_bps=null_bps,
    )
    per_date = _per_market_date_metrics(
        nonlinear_model,
        linear_model,
        test_samples,
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
        "experiment": "v3_hist_gradient_absolute_return_magnitude_test_lockbox_v1",
        "dataset_id": manifest.get("dataset_id"),
        "dataset_builder": manifest.get("builder"),
        "protocol_name": protocol_result.get("protocol_name"),
        "protocol_sha256": protocol_result.get("protocol_sha256"),
        "development_report": str(development_report_path),
        "validation_report": str(validation_report_path),
        "horizon_minutes": NONLINEAR_CONFIG.horizon_minutes,
        "final_training_source": "train_plus_validation",
        "frozen_nonlinear_model": {
            "family": "HistGradientBoostingRegressor",
            "loss": "absolute_error",
            "config": _expected_nonlinear_config(),
        },
        "incumbent": {
            "family": "ridge_absolute_return",
            "regression_l2": LINEAR_CONFIG.regression_l2,
        },
        "test_rule": {
            "overall_nonlinear_mae_must_beat_linear": True,
            "overall_nonlinear_mae_must_beat_final_training_median_null": True,
            "require_strict_majority_market_date_wins_vs_linear": True,
            "require_strict_majority_market_date_wins_vs_null": True,
            "tuning_after_test": False,
            "pnl_used": False,
        },
        "fit_summary": {
            "nonlinear": nonlinear_fit.to_dict(),
            "linear": linear_fit.to_dict(),
        },
        "test": {
            "overall": overall,
            "market_dates": per_date,
            "market_date_count": date_count,
            "market_date_wins_vs_linear": wins_vs_linear,
            "market_date_wins_vs_null": wins_vs_null,
            "relative_mae_improvement_vs_linear": relative_vs_linear,
            "relative_mae_improvement_vs_null": relative_vs_null,
        },
        "test_supported": supported,
        "test_lockbox_accessed": True,
        "post_test_tuning_allowed": False,
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
            "Open the frozen OurAI v3 nonlinear 15m magnitude test lockbox exactly once "
            "after both development and validation gates pass."
        )
    )
    result.add_argument("--dataset-dir", required=True)
    result.add_argument("--development-report", required=True)
    result.add_argument("--validation-report", required=True)
    result.add_argument(
        "--output",
        default="runs/first-real-15m-v3/nonlinear_magnitude_test_lockbox_v1.json",
    )
    return result


def main() -> int:
    args = parser().parse_args()
    payload = run_test_lockbox_gate(
        args.dataset_dir,
        args.development_report,
        args.validation_report,
        args.output,
    )
    test = payload["test"]
    overall = test["overall"]
    print("=== V3 NONLINEAR MAGNITUDE TEST LOCKBOX ===")
    print("Dataset ID:", payload["dataset_id"])
    print("Test samples:", overall["evaluated_count"])
    print("Nonlinear MAE bps:", overall["nonlinear_mae_bps"])
    print("Linear MAE bps:", overall["linear_mae_bps"])
    print("Null MAE bps:", overall["null_mae_bps"])
    print("Delta vs linear bps:", overall["delta_vs_linear_bps"])
    print("Delta vs null bps:", overall["delta_vs_null_bps"])
    print(
        "Market-date wins vs linear:",
        f"{test['market_date_wins_vs_linear']}/{test['market_date_count']}",
    )
    print(
        "Market-date wins vs null:",
        f"{test['market_date_wins_vs_null']}/{test['market_date_count']}",
    )
    print(
        "Relative MAE improvement vs linear:",
        test["relative_mae_improvement_vs_linear"],
    )
    print(
        "Relative MAE improvement vs null:",
        test["relative_mae_improvement_vs_null"],
    )
    print("TEST LOCKBOX SUPPORTED:", payload["test_supported"])
    print("Report:", args.output)
    print("Post-test tuning allowed: NO")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
