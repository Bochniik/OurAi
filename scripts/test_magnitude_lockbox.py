"""One-shot final test-lockbox gate for the frozen 15m magnitude challenger.

The gate is intentionally fixed before opening the real test split:

* require the successful train-only development report;
* require the successful frozen validation report;
* reuse the exact frozen magnitude model configuration;
* fit the final model on train + validation only;
* evaluate exactly once on test.jsonl;
* compare against the train+validation median-absolute-return null for MAE;
* require lower overall MAE and a strict majority of test market-date MAE wins;
* do not use PnL as an acceptance criterion.

No model or threshold tuning is permitted after the test lockbox is opened.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

from scripts.evaluate_magnitude_head import (
    EXPECTED_BUILDER,
    EXPECTED_AVAILABILITY_DELAY_MINUTES,
    MODEL_CONFIG,
    _load_train,
    _validate_dataset_manifest,
)
from scripts.validate_magnitude_head import (
    _load_validation,
    _validate_development_report,
)
from trading.dataset import TradingSample
from trading.magnitude import (
    evaluate_magnitude_predictions,
    fit_unified_linear_magnitude_baseline,
    forecast_magnitudes,
)

EXPECTED_VALIDATION_EXPERIMENT = "absolute_return_linear_magnitude_validation_v1"


def _parse_timestamp(value: object) -> datetime:
    timestamp = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ValueError("sample timestamp must be timezone-aware")
    return timestamp


def _load_test(path: Path) -> list[TradingSample]:
    if path.name != "test.jsonl":
        raise ValueError("test lockbox gate accepts test.jsonl only")
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
                raise ValueError(f"invalid test row at line {line_number}") from exc
    if not rows:
        raise ValueError("test.jsonl is empty")
    return rows


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _validate_validation_report(
    report: Mapping[str, Any],
    *,
    dataset_id: object,
    development_report_path: str | Path,
) -> None:
    if report.get("experiment") != EXPECTED_VALIDATION_EXPERIMENT:
        raise ValueError("unexpected validation experiment identity")
    if report.get("dataset_id") != dataset_id:
        raise ValueError("validation report dataset_id does not match dataset")
    if report.get("dataset_builder") != EXPECTED_BUILDER:
        raise ValueError("validation report was not produced from historical-v2")
    if report.get("validation_supported") is not True:
        raise ValueError("validation magnitude challenger was not supported")
    if report.get("original_validation_used") is not True:
        raise ValueError("validation report must have opened validation")
    if report.get("test_lockbox_accessed") is not False:
        raise ValueError("validation report must not have accessed test lockbox")

    frozen_model = _mapping(report.get("frozen_model"), "frozen_model")
    if frozen_model.get("family") != "ridge_absolute_return":
        raise ValueError("unexpected frozen validation model family")
    if float(frozen_model.get("regression_l2", -1.0)) != MODEL_CONFIG.regression_l2:
        raise ValueError("validation report changed frozen regression_l2")
    if int(frozen_model.get("minimum_training_samples", -1)) != MODEL_CONFIG.minimum_training_samples:
        raise ValueError("validation report changed frozen minimum_training_samples")
    if frozen_model.get("negative_prediction_policy") != "clip_to_zero":
        raise ValueError("validation report changed negative prediction policy")

    rule = _mapping(report.get("validation_rule"), "validation_rule")
    if rule.get("overall_model_mae_must_beat_train_median_null") is not True:
        raise ValueError("validation report used an unexpected MAE rule")
    if rule.get("require_strict_majority_market_date_mae_wins") is not True:
        raise ValueError("validation report used an unexpected date-win rule")
    if rule.get("tuning_after_validation") is not False:
        raise ValueError("validation report does not freeze post-validation tuning")
    if rule.get("pnl_used") is not False:
        raise ValueError("validation report must not use PnL")

    if str(report.get("development_report")) != str(development_report_path):
        raise ValueError("validation report references a different development report")


def _per_market_date_metrics(
    model: object,
    samples: list[TradingSample],
    *,
    null_magnitude_bps: float,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[TradingSample]] = defaultdict(list)
    for sample in samples:
        grouped[sample.session_date].append(sample)

    results: list[dict[str, Any]] = []
    for session_date in sorted(grouped):
        predictions = forecast_magnitudes(
            model,
            grouped[session_date],
            include_actual=True,
        )
        metrics = evaluate_magnitude_predictions(
            predictions,
            null_magnitude_bps=null_magnitude_bps,
        )
        results.append(
            {
                "market_date": session_date,
                "evaluated_count": metrics["evaluated_count"],
                "model_mae_bps": metrics["model_mae_bps"],
                "null_mae_bps": metrics["null_mae_bps"],
                "mae_delta_bps": metrics["mae_delta_bps"],
                "model_wins_mae": metrics["model_wins_mae"],
            }
        )
    return results


def run_test_lockbox_gate(
    dataset_dir: str | Path,
    development_report_path: str | Path,
    validation_report_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    root = Path(dataset_dir)
    manifest_path = root / "manifest.json"
    train_path = root / "train.jsonl"
    validation_path = root / "validation.jsonl"
    test_path = root / "test.jsonl"

    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    if not train_path.is_file():
        raise FileNotFoundError(f"train split not found: {train_path}")
    if not validation_path.is_file():
        raise FileNotFoundError(f"validation split not found: {validation_path}")
    if not test_path.is_file():
        raise FileNotFoundError(f"test split not found: {test_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, Mapping):
        raise ValueError("manifest must contain a JSON object")
    _validate_dataset_manifest(manifest)

    development_report = json.loads(Path(development_report_path).read_text(encoding="utf-8"))
    if not isinstance(development_report, Mapping):
        raise ValueError("development report must contain a JSON object")
    _validate_development_report(development_report, dataset_id=manifest.get("dataset_id"))

    validation_report = json.loads(Path(validation_report_path).read_text(encoding="utf-8"))
    if not isinstance(validation_report, Mapping):
        raise ValueError("validation report must contain a JSON object")
    _validate_validation_report(
        validation_report,
        dataset_id=manifest.get("dataset_id"),
        development_report_path=development_report_path,
    )

    # Only after both frozen pre-test gates are proven do we open the test lockbox.
    train_samples = _load_train(train_path)
    validation_samples = _load_validation(validation_path)
    final_training_samples = [*train_samples, *validation_samples]

    model, fit_summary = fit_unified_linear_magnitude_baseline(
        final_training_samples,
        config=MODEL_CONFIG,
    )

    test_samples = _load_test(test_path)
    predictions = forecast_magnitudes(model, test_samples, include_actual=True)
    overall = evaluate_magnitude_predictions(
        predictions,
        null_magnitude_bps=fit_summary.training_median_magnitude_bps,
    )
    per_date = _per_market_date_metrics(
        model,
        test_samples,
        null_magnitude_bps=fit_summary.training_median_magnitude_bps,
    )

    date_wins = sum(bool(row["model_wins_mae"]) for row in per_date)
    date_count = len(per_date)
    strict_majority_date_wins = date_wins > (date_count / 2.0)
    supported = bool(overall["model_wins_mae"] and strict_majority_date_wins)
    relative_improvement = None
    if overall["null_mae_bps"] > 0:
        relative_improvement = (
            (overall["null_mae_bps"] - overall["model_mae_bps"])
            / overall["null_mae_bps"]
        )

    payload = {
        "schema_version": 1,
        "experiment": "absolute_return_linear_magnitude_test_lockbox_v1",
        "dataset_id": manifest.get("dataset_id"),
        "dataset_builder": manifest.get("builder"),
        "bar_availability_delay_minutes": EXPECTED_AVAILABILITY_DELAY_MINUTES,
        "horizon_minutes": MODEL_CONFIG.horizon_minutes,
        "development_report": str(development_report_path),
        "validation_report": str(validation_report_path),
        "final_training_source": "train_plus_validation",
        "frozen_model": {
            "family": "ridge_absolute_return",
            "regression_l2": MODEL_CONFIG.regression_l2,
            "minimum_training_samples": MODEL_CONFIG.minimum_training_samples,
            "negative_prediction_policy": "clip_to_zero",
        },
        "test_rule": {
            "overall_model_mae_must_beat_final_training_median_null": True,
            "require_strict_majority_market_date_mae_wins": True,
            "tuning_after_test": False,
            "pnl_used": False,
        },
        "fit_summary": fit_summary.to_dict(),
        "test": {
            "overall": overall,
            "market_dates": per_date,
            "market_date_win_count": date_wins,
            "market_date_count": date_count,
            "relative_mae_improvement": relative_improvement,
        },
        "test_supported": supported,
        "test_lockbox_accessed": True,
        "post_test_tuning_allowed": False,
    }

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Open the 15m magnitude test lockbox exactly once after both frozen pre-test gates pass."
    )
    result.add_argument("--dataset-dir", required=True)
    result.add_argument("--development-report", required=True)
    result.add_argument("--validation-report", required=True)
    result.add_argument(
        "--output",
        default="runs/first-real-15m-v2/magnitude_test_lockbox_v1.json",
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

    print("=== MAGNITUDE TEST LOCKBOX ===")
    print("Dataset ID:", payload["dataset_id"])
    print("Test samples:", overall["evaluated_count"])
    print("Model MAE bps:", overall["model_mae_bps"])
    print("Null MAE bps:", overall["null_mae_bps"])
    print("MAE delta bps:", overall["mae_delta_bps"])
    print("Market-date wins:", f"{test['market_date_win_count']}/{test['market_date_count']}")
    print("Relative MAE improvement:", test["relative_mae_improvement"])
    print("TEST LOCKBOX SUPPORTED:", payload["test_supported"])
    print("Report:", args.output)
    print("Post-test tuning allowed: NO")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
