"""Train-only v4 direction exploration with causal cross-sectional market context.

This script does not open v4 validation.jsonl or test.jsonl. It reuses the canonical
v4 train split, enriches each sample only with same-timestamp peer features, then
compares two context-aware candidates against the already-recorded original linear
incumbent and training-prior null on the same seven purged walk-forward folds.

The output is development evidence only. It does not promote a model to validation;
a separate frozen gate is required after a candidate is selected.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import fmean
from typing import Any

from scripts.evaluate_magnitude_head import _load_train
from scripts.evaluate_v4_nonlinear_direction import (
    EMBARGO_MINUTES,
    EXPECTED_DATASET_ID,
    EXPECTED_FOLD_COUNT,
    EXPECTED_TRAIN_MARKET_DATES,
    HORIZON_MINUTES,
    LINEAR_CONFIG,
    MIN_TRAIN_MARKET_DATES,
    NONLINEAR_CONFIG,
    STEP_MARKET_DATES,
    VALIDATION_MARKET_DATES,
    _validate_v4_contract,
)
from trading.baseline import fit_unified_linear_baseline
from trading.cross_sectional import enrich_cross_sectional_context
from trading.direction import (
    evaluate_direction_predictions,
    fit_hist_gradient_direction,
    forecast_hist_gradient_directions,
    forecast_linear_directions,
)
from trading.validation import purged_walk_forward_folds


def _load_base_report(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("dataset_id") != EXPECTED_DATASET_ID:
        raise ValueError("base direction report dataset_id mismatch")
    if payload.get("validation_used") is not False:
        raise ValueError("base direction report must be train-only")
    if payload.get("test_lockbox_accessed") is not False:
        raise ValueError("base direction report must not have accessed test")
    folds = payload.get("fold_results")
    if not isinstance(folds, list) or len(folds) != EXPECTED_FOLD_COUNT:
        raise ValueError("base direction report fold count mismatch")
    return payload


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    return fmean(float(row[key]) for row in rows)


def run_exploration(
    dataset_dir: str | Path,
    base_report_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    root = Path(dataset_dir)
    manifest, protocol_result, amendment = _validate_v4_contract(root)
    base_report = _load_base_report(base_report_path)

    train_path = root / "train.jsonl"
    if not train_path.is_file():
        raise FileNotFoundError(f"train split not found: {train_path}")

    # Validation and test files are intentionally never opened here.
    original_samples = _load_train(train_path)
    train_dates = sorted({sample.session_date for sample in original_samples})
    if len(train_dates) != EXPECTED_TRAIN_MARKET_DATES:
        raise ValueError("unexpected v4 train market-date count")

    context_samples = enrich_cross_sectional_context(original_samples)
    original_folds = purged_walk_forward_folds(
        original_samples,
        min_train_market_dates=MIN_TRAIN_MARKET_DATES,
        validation_market_dates=VALIDATION_MARKET_DATES,
        step_market_dates=STEP_MARKET_DATES,
        embargo_minutes=EMBARGO_MINUTES,
        max_train_market_dates=None,
    )
    context_folds = purged_walk_forward_folds(
        context_samples,
        min_train_market_dates=MIN_TRAIN_MARKET_DATES,
        validation_market_dates=VALIDATION_MARKET_DATES,
        step_market_dates=STEP_MARKET_DATES,
        embargo_minutes=EMBARGO_MINUTES,
        max_train_market_dates=None,
    )
    if len(original_folds) != EXPECTED_FOLD_COUNT or len(context_folds) != EXPECTED_FOLD_COUNT:
        raise ValueError("context exploration fold count mismatch")

    base_by_fold = {int(row["fold"]): row for row in base_report["fold_results"]}
    rows: list[dict[str, Any]] = []

    for original_fold, context_fold in zip(original_folds, context_folds):
        if original_fold.validation_market_dates != context_fold.validation_market_dates:
            raise AssertionError("context enrichment changed fold boundaries")

        context_linear, linear_fit = fit_unified_linear_baseline(
            context_fold.train,
            config=LINEAR_CONFIG,
        )
        context_nonlinear, nonlinear_fit = fit_hist_gradient_direction(
            context_fold.train,
            config=NONLINEAR_CONFIG,
        )
        null_p_up = nonlinear_fit.training_up_rate

        linear_predictions = forecast_linear_directions(
            context_linear,
            context_fold.validation,
            include_actual=True,
        )
        nonlinear_predictions = forecast_hist_gradient_directions(
            context_nonlinear,
            context_fold.validation,
            include_actual=True,
        )
        linear_metrics = evaluate_direction_predictions(linear_predictions, null_p_up=null_p_up)
        nonlinear_metrics = evaluate_direction_predictions(nonlinear_predictions, null_p_up=null_p_up)

        base = base_by_fold[context_fold.index]
        base_null = float(base["null_brier"])
        if abs(float(linear_metrics["null_brier_score"]) - base_null) > 1e-12:
            raise AssertionError("context fold null no longer matches canonical base fold")

        rows.append(
            {
                "fold": context_fold.index,
                "validation_market_dates": list(context_fold.validation_market_dates),
                "context_linear_brier": float(linear_metrics["brier_score"]),
                "context_linear_log_loss": float(linear_metrics["log_loss"]),
                "context_nonlinear_brier": float(nonlinear_metrics["brier_score"]),
                "context_nonlinear_log_loss": float(nonlinear_metrics["log_loss"]),
                "original_linear_brier": float(base["linear_brier"]),
                "original_linear_log_loss": float(base["linear_log_loss"]),
                "null_brier": base_null,
                "null_log_loss": float(base["null_log_loss"]),
                "context_linear_feature_count": linear_fit.feature_count,
                "context_nonlinear_feature_count": nonlinear_fit.feature_count,
                "context_linear_wins_vs_original": float(linear_metrics["brier_score"]) < float(base["linear_brier"]),
                "context_linear_wins_vs_null": float(linear_metrics["brier_score"]) < base_null,
                "context_nonlinear_wins_vs_original": float(nonlinear_metrics["brier_score"]) < float(base["linear_brier"]),
                "context_nonlinear_wins_vs_null": float(nonlinear_metrics["brier_score"]) < base_null,
                "context_nonlinear_wins_vs_context_linear": float(nonlinear_metrics["brier_score"]) < float(linear_metrics["brier_score"]),
            }
        )

    models = {
        "context_linear": {
            "mean_brier": _mean(rows, "context_linear_brier"),
            "mean_log_loss": _mean(rows, "context_linear_log_loss"),
            "brier_wins_vs_original_linear": sum(bool(row["context_linear_wins_vs_original"]) for row in rows),
            "brier_wins_vs_null": sum(bool(row["context_linear_wins_vs_null"]) for row in rows),
        },
        "context_nonlinear": {
            "mean_brier": _mean(rows, "context_nonlinear_brier"),
            "mean_log_loss": _mean(rows, "context_nonlinear_log_loss"),
            "brier_wins_vs_original_linear": sum(bool(row["context_nonlinear_wins_vs_original"]) for row in rows),
            "brier_wins_vs_null": sum(bool(row["context_nonlinear_wins_vs_null"]) for row in rows),
            "brier_wins_vs_context_linear": sum(bool(row["context_nonlinear_wins_vs_context_linear"]) for row in rows),
        },
        "original_linear": {
            "mean_brier": _mean(rows, "original_linear_brier"),
            "mean_log_loss": _mean(rows, "original_linear_log_loss"),
        },
        "null": {
            "mean_brier": _mean(rows, "null_brier"),
            "mean_log_loss": _mean(rows, "null_log_loss"),
        },
    }
    preferred = min(models, key=lambda name: float(models[name]["mean_brier"]))

    payload = {
        "schema_version": 1,
        "experiment": "v4_cross_sectional_direction_train_only_exploration_v1",
        "dataset_id": manifest.get("dataset_id"),
        "protocol_name": protocol_result.get("protocol_name"),
        "development_amendment": amendment.get("amendment_name"),
        "horizon_minutes": HORIZON_MINUTES,
        "source": "train_split_only",
        "validation_used": False,
        "test_lockbox_accessed": False,
        "pnl_used": False,
        "feature_contract": {
            "peer_definition": "other_symbols_at_exact_same_availability_timestamp",
            "current_symbol_excluded_from_peer_aggregates": True,
            "future_or_target_values_used": False,
            "return_horizons_minutes": [1, 5, 15, 30],
        },
        "fold_results": rows,
        "summary": {
            "fold_count": len(rows),
            "models": models,
            "preferred_model_by_train_only_mean_brier": preferred,
            "eligible_for_validation_without_separate_freeze": False,
        },
    }

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Explore causal cross-sectional direction context on v4 train only.")
    result.add_argument("--dataset-dir", required=True)
    result.add_argument(
        "--base-report",
        default="runs/first-real-direction-v4/nonlinear_direction_development_v1.json",
    )
    result.add_argument(
        "--output",
        default="runs/first-real-direction-v4/context_direction_exploration_v1.json",
    )
    return result


def main() -> int:
    args = parser().parse_args()
    payload = run_exploration(args.dataset_dir, args.base_report, args.output)
    for row in payload["fold_results"]:
        dates = row["validation_market_dates"]
        print(
            f"FOLD {row['fold']} {dates[0]} -> {dates[-1]} "
            f"context_linear={row['context_linear_brier']:.6f} "
            f"context_nonlinear={row['context_nonlinear_brier']:.6f} "
            f"original_linear={row['original_linear_brier']:.6f} "
            f"null={row['null_brier']:.6f}"
        )

    summary = payload["summary"]
    models = summary["models"]
    print()
    print("=== V4 CROSS-SECTIONAL DIRECTION EXPLORATION ===")
    print("Dataset ID:", payload["dataset_id"])
    for name in ("context_linear", "context_nonlinear", "original_linear", "null"):
        model = models[name]
        print(name, "mean Brier=", model["mean_brier"], "mean log loss=", model["mean_log_loss"])
    print("Context linear wins vs original:", f"{models['context_linear']['brier_wins_vs_original_linear']}/{summary['fold_count']}")
    print("Context linear wins vs null:", f"{models['context_linear']['brier_wins_vs_null']}/{summary['fold_count']}")
    print("Context nonlinear wins vs original:", f"{models['context_nonlinear']['brier_wins_vs_original_linear']}/{summary['fold_count']}")
    print("Context nonlinear wins vs null:", f"{models['context_nonlinear']['brier_wins_vs_null']}/{summary['fold_count']}")
    print("Preferred by train-only mean Brier:", summary["preferred_model_by_train_only_mean_brier"])
    print("V4 validation: NOT ACCESSED")
    print("V4 test lockbox: NOT ACCESSED")
    print("Eligible for validation now: NO - separate freeze required")
    print("Report:", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
