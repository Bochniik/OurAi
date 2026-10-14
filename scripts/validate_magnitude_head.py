"""One-shot validation gate for the frozen 15m magnitude challenger.

This runner is intentionally stricter than a generic evaluator:

* it requires a successful preregistered train-only development report;
* it reuses the exact frozen model configuration from ``evaluate_magnitude_head``;
* it fits only on ``train.jsonl``;
* it evaluates exactly once on ``validation.jsonl``;
* it never opens ``test.jsonl``;
* validation support requires lower overall MAE than the fixed train-median null
  and a strict majority of validation market dates with lower MAE.

The rule above is part of the code before the user's real validation result is read.
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
from trading.dataset import TradingSample
from trading.magnitude import (
    evaluate_magnitude_predictions,
    fit_unified_linear_magnitude_baseline,
    forecast_magnitudes,
)

EXPECTED_DEVELOPMENT_EXPERIMENT = "absolute_return_linear_magnitude_v1"
EXPECTED_DEVELOPMENT_FOLD_COUNT = 4
EXPECTED_DEVELOPMENT_MIN_FOLD_WINS = 3


def _parse_timestamp(value: object) -> datetime:
    timestamp = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ValueError("sample timestamp must be timezone-aware")
    return timestamp


def _load_validation(path: Path) -> list[TradingSample]:
    if path.name != "validation.jsonl":
        raise ValueError("validation gate accepts validation.jsonl only")
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
                raise ValueError(
                    f"invalid validation row at line {line_number}"
                ) from exc
    if not rows:
        raise ValueError("validation.jsonl is empty")
    return rows


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _validate_development_report(
    report: Mapping[str, Any],
    *,
    dataset_id: object,
) -> None:
    if report.get("experiment") != EXPECTED_DEVELOPMENT_EXPERIMENT:
        raise ValueError("unexpected development experiment identity")
    if report.get("dataset_id") != dataset_id:
        raise ValueError("development report dataset_id does not match dataset")
    if report.get("dataset_builder") != EXPECTED_BUILDER:
        raise ValueError("development report was not produced from historical-v2")
    if report.get("original_validation_used") is not False:
        raise ValueError("development report must not have used original validation")
    if report.get("test_lockbox_accessed") is not False:
        raise ValueError("development report must not have accessed test lockbox")

    promotion = _mapping(report.get("promotion_rule"), "promotion_rule")
    if int(promotion.get("required_fold_count", -1)) != EXPECTED_DEVELOPMENT_FOLD_COUNT:
        raise ValueError("unexpected development fold count")
    if int(promotion.get("minimum_fold_wins", -1)) != EXPECTED_DEVELOPMENT_MIN_FOLD_WINS:
        raise ValueError("unexpected development promotion rule")
    if promotion.get("require_lower_mean_mae") is not True:
        raise ValueError("development report must require lower mean MAE")

    summary = _mapping(report.get("summary"), "summary")
    if int(summary.get("fold_count", -1)) != EXPECTED_DEVELOPMENT_FOLD_COUNT:
        raise ValueError("development report does not contain four folds")
    if int(summary.get("fold_wins", -1)) < EXPECTED_DEVELOPMENT_MIN_FOLD_WINS:
        raise ValueError("development magnitude challenger did not pass fold-win gate")
    if summary.get("magnitude_challenger_supported") is not True:
        raise ValueError("development magnitude challenger was not supported")


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


def run_validation_gate(
    dataset_dir: str | Path,
    development_report_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    root = Path(dataset_dir)
    manifest_path = root / "manifest.json"
    train_path = root / "train.jsonl"
    validation_path = root / "validation.jsonl"

    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    if not train_path.is_file():
        raise FileNotFoundError(f"train split not found: {train_path}")
    if not validation_path.is_file():
        raise FileNotFoundError(f"validation split not found: {validation_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, Mapping):
        raise ValueError("manifest must contain a JSON object")
    _validate_dataset_manifest(manifest)

    development_report = json.loads(
        Path(development_report_path).read_text(encoding="utf-8")
    )
    if not isinstance(development_report, Mapping):
        raise ValueError("development report must contain a JSON object")
    _validate_development_report(
        development_report,
        dataset_id=manifest.get("dataset_id"),
    )

    # Only after the frozen development gate has been proven do we open validation.
    training_samples = _load_train(train_path)
    validation_samples = _load_validation(validation_path)

    model, fit_summary = fit_unified_linear_magnitude_baseline(
        training_samples,
        config=MODEL_CONFIG,
    )
    predictions = forecast_magnitudes(
        model,
        validation_samples,
        include_actual=True,
    )
    overall = evaluate_magnitude_predictions(
        predictions,
        null_magnitude_bps=fit_summary.training_median_magnitude_bps,
    )
    per_date = _per_market_date_metrics(
        model,
        validation_samples,
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
        "experiment": "absolute_return_linear_magnitude_validation_v1",
        "dataset_id": manifest.get("dataset_id"),
        "dataset_builder": manifest.get("builder"),
        "bar_availability_delay_minutes": EXPECTED_AVAILABILITY_DELAY_MINUTES,
        "horizon_minutes": MODEL_CONFIG.horizon_minutes,
        "development_report": str(development_report_path),
        "frozen_model": {
            "family": "ridge_absolute_return",
            "regression_l2": MODEL_CONFIG.regression_l2,
            "minimum_training_samples": MODEL_CONFIG.minimum_training_samples,
            "negative_prediction_policy": "clip_to_zero",
        },
        "validation_rule": {
            "overall_model_mae_must_beat_train_median_null": True,
            "require_strict_majority_market_date_mae_wins": True,
            "tuning_after_validation": False,
            "pnl_used": False,
        },
        "fit_summary": fit_summary.to_dict(),
        "validation": {
            "overall": overall,
            "market_dates": per_date,
            "market_date_win_count": date_wins,
            "market_date_count": date_count,
            "relative_mae_improvement": relative_improvement,
        },
        "validation_supported": supported,
        "original_validation_used": True,
        "test_lockbox_accessed": False,
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
            "Run the one-shot frozen 15m magnitude validation gate. "
            "The test lockbox is never opened."
        )
    )
    result.add_argument(
        "--dataset-dir",
        required=True,
        help="Historical-v2 dataset directory containing train/validation splits.",
    )
    result.add_argument(
        "--development-report",
        required=True,
        help="Successful preregistered train-only magnitude report.",
    )
    result.add_argument(
        "--output",
        default="runs/first-real-15m-v2/magnitude_validation_v1.json",
        help="Validation report output path.",
    )
    return result


def main() -> int:
    args = parser().parse_args()
    payload = run_validation_gate(
        args.dataset_dir,
        args.development_report,
        args.output,
    )
    validation = payload["validation"]
    overall = validation["overall"]

    print("=== MAGNITUDE VALIDATION GATE ===")
    print("Dataset ID:", payload["dataset_id"])
    print("Validation samples:", overall["evaluated_count"])
    print("Model MAE bps:", overall["model_mae_bps"])
    print("Null MAE bps:", overall["null_mae_bps"])
    print("MAE delta bps:", overall["mae_delta_bps"])
    print("Market-date wins:", f"{validation['market_date_win_count']}/{validation['market_date_count']}")
    print("Relative MAE improvement:", validation["relative_mae_improvement"])
    print("VALIDATION SUPPORTED:", payload["validation_supported"])
    print("Report:", args.output)
    print("Test lockbox: NOT ACCESSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
