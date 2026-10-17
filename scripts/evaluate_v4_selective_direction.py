"""Train-only selective 15m direction development for v4.

The previous every-minute direction challengers failed to beat a train-prior null.
This experiment therefore separates two questions:

1. Is a materially larger 15-minute move likely? (nonlinear magnitude gate)
2. Conditional on an actionable state, is the move more likely UP or DOWN?

The direction candidate uses causal temporal state plus same-timestamp peer context and
is trained only on the upper 30% of train-only realized absolute 15m moves. At
inference, selection is based only on the magnitude model's prediction, never on the
future return. v4 validation.jsonl and test.jsonl are not opened.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable

import numpy as np

from scripts.evaluate_magnitude_head import _load_train
from scripts.evaluate_v4_nonlinear_direction import (
    EMBARGO_MINUTES,
    EXPECTED_DATASET_ID,
    EXPECTED_FOLD_COUNT,
    EXPECTED_TRAIN_MARKET_DATES,
    HORIZON_MINUTES,
    LINEAR_CONFIG,
    MIN_TRAIN_MARKET_DATES,
    STEP_MARKET_DATES,
    VALIDATION_MARKET_DATES,
    _validate_v4_contract,
)
from trading.baseline import fit_unified_linear_baseline
from trading.cross_sectional import enrich_cross_sectional_context
from trading.dataset import TradingSample
from trading.direction import evaluate_direction_predictions, forecast_linear_directions
from trading.nonlinear import (
    HistGradientMagnitudeConfig,
    fit_hist_gradient_magnitude,
    forecast_hist_gradient_magnitudes,
)
from trading.temporal_state import enrich_temporal_state
from trading.validation import purged_walk_forward_folds


PROTOCOL_PATH = (
    Path(__file__).resolve().parents[1]
    / "research"
    / "v4_selective_direction_protocol_v1.json"
)
TAIL_QUANTILE = 0.70
GATE_PREDICTION_QUANTILE = 0.70
MINIMUM_WINS = 5
MINIMUM_MEAN_COVERAGE = 0.10

MAGNITUDE_CONFIG = HistGradientMagnitudeConfig(
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


def _target(sample: TradingSample) -> dict[str, Any] | None:
    value = sample.targets.get(f"{HORIZON_MINUTES}m")
    return dict(value) if isinstance(value, dict) else None


def _future_return(sample: TradingSample) -> float | None:
    target = _target(sample)
    if target is None:
        return None
    try:
        value = float(target.get("future_return"))
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def _direction(sample: TradingSample) -> str | None:
    target = _target(sample)
    if target is None:
        return None
    direction = str(target.get("direction") or "").strip().upper()
    return direction if direction in {"UP", "DOWN", "NEUTRAL"} else None


def _resolved_magnitudes_bps(samples: Iterable[TradingSample]) -> list[float]:
    values: list[float] = []
    for sample in samples:
        value = _future_return(sample)
        if value is not None:
            values.append(abs(value) * 10_000.0)
    if not values:
        raise ValueError("no resolved 15m magnitudes available")
    return values


def _tail_directional_samples(
    samples: Iterable[TradingSample],
    threshold_bps: float,
) -> list[TradingSample]:
    result: list[TradingSample] = []
    for sample in samples:
        value = _future_return(sample)
        direction = _direction(sample)
        if value is None or direction not in {"UP", "DOWN"}:
            continue
        if abs(value) * 10_000.0 >= threshold_bps:
            result.append(sample)
    return result


def _up_rate(samples: Iterable[TradingSample]) -> float:
    labels = [_direction(sample) for sample in samples]
    directional = [label for label in labels if label in {"UP", "DOWN"}]
    if not directional:
        raise ValueError("no directional rows for tail null")
    value = sum(label == "UP" for label in directional) / len(directional)
    if not 0.0 < value < 1.0:
        raise ValueError("tail null requires both UP and DOWN rows")
    return float(value)


def _mean_actual_magnitude_bps(samples: Iterable[TradingSample]) -> float:
    values = _resolved_magnitudes_bps(samples)
    return fmean(values)


def _canonical_sha256(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_protocol() -> dict[str, Any]:
    if not PROTOCOL_PATH.is_file():
        raise FileNotFoundError(f"selective direction protocol not found: {PROTOCOL_PATH}")
    payload = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if payload.get("protocol_name") != "ourai-v4-selective-direction-development-v1":
        raise ValueError("unexpected selective direction protocol")
    if payload.get("dataset_id") != EXPECTED_DATASET_ID:
        raise ValueError("selective direction protocol dataset mismatch")
    if int(payload.get("horizon_minutes")) != HORIZON_MINUTES:
        raise ValueError("selective direction protocol horizon mismatch")
    if float(payload["direction_candidate"]["tail_quantile"]) != TAIL_QUANTILE:
        raise ValueError("tail quantile drifted from selective protocol")
    if float(payload["magnitude_gate"]["training_prediction_quantile_for_gate"]) != GATE_PREDICTION_QUANTILE:
        raise ValueError("magnitude gate quantile drifted from selective protocol")
    gate = payload["development_gate"]
    if int(gate["minimum_brier_fold_wins_vs_original_linear"]) != MINIMUM_WINS:
        raise ValueError("selective development win threshold drifted")
    if int(gate["minimum_brier_fold_wins_vs_tail_null"]) != MINIMUM_WINS:
        raise ValueError("selective null win threshold drifted")
    if float(gate["minimum_mean_selected_coverage"]) != MINIMUM_MEAN_COVERAGE:
        raise ValueError("selective minimum coverage drifted")
    if gate.get("pnl_used") is not False:
        raise ValueError("PnL must not be used by selective development gate")
    return payload


def _load_prior_report(path: str | Path, *, expected_experiment_prefix: str) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("dataset_id") != EXPECTED_DATASET_ID:
        raise ValueError("prior direction report dataset mismatch")
    if not str(payload.get("experiment") or "").startswith(expected_experiment_prefix):
        raise ValueError("unexpected prior direction report")
    if payload.get("validation_used") is not False:
        raise ValueError("prior direction report must not use v4 validation")
    if payload.get("test_lockbox_accessed") is not False:
        raise ValueError("prior direction report must not access v4 test")
    return payload


def run_experiment(
    dataset_dir: str | Path,
    base_report_path: str | Path,
    context_report_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    root = Path(dataset_dir)
    manifest, protocol_result, amendment = _validate_v4_contract(root)
    protocol = _load_protocol()
    base_report = _load_prior_report(
        base_report_path,
        expected_experiment_prefix="v4_hist_gradient_direction_15m_v1",
    )
    context_report = _load_prior_report(
        context_report_path,
        expected_experiment_prefix="v4_cross_sectional_direction_train_only_exploration_v1",
    )
    if context_report.get("summary", {}).get("preferred_model_by_train_only_mean_brier") != "null":
        raise ValueError("selective experiment expects the recorded context exploration to prefer null")

    train_path = root / "train.jsonl"
    if not train_path.is_file():
        raise FileNotFoundError(f"train split not found: {train_path}")

    # Fresh v4 validation.jsonl and test.jsonl are intentionally never opened here.
    original_samples = _load_train(train_path)
    train_dates = sorted({sample.session_date for sample in original_samples})
    if len(train_dates) != EXPECTED_TRAIN_MARKET_DATES:
        raise ValueError("unexpected v4 train market-date count")

    state_samples = enrich_cross_sectional_context(
        enrich_temporal_state(original_samples)
    )
    original_folds = purged_walk_forward_folds(
        original_samples,
        min_train_market_dates=MIN_TRAIN_MARKET_DATES,
        validation_market_dates=VALIDATION_MARKET_DATES,
        step_market_dates=STEP_MARKET_DATES,
        embargo_minutes=EMBARGO_MINUTES,
        max_train_market_dates=None,
    )
    state_folds = purged_walk_forward_folds(
        state_samples,
        min_train_market_dates=MIN_TRAIN_MARKET_DATES,
        validation_market_dates=VALIDATION_MARKET_DATES,
        step_market_dates=STEP_MARKET_DATES,
        embargo_minutes=EMBARGO_MINUTES,
        max_train_market_dates=None,
    )
    if len(original_folds) != EXPECTED_FOLD_COUNT or len(state_folds) != EXPECTED_FOLD_COUNT:
        raise ValueError("selective direction fold count mismatch")

    fold_rows: list[dict[str, Any]] = []
    for original_fold, state_fold in zip(original_folds, state_folds):
        if original_fold.validation_market_dates != state_fold.validation_market_dates:
            raise AssertionError("state enrichment changed walk-forward boundaries")

        train_magnitudes = _resolved_magnitudes_bps(state_fold.train)
        actual_tail_threshold_bps = float(np.quantile(train_magnitudes, TAIL_QUANTILE))
        tail_train = _tail_directional_samples(
            state_fold.train,
            actual_tail_threshold_bps,
        )
        tail_null_p_up = _up_rate(tail_train)

        magnitude_model, magnitude_fit = fit_hist_gradient_magnitude(
            state_fold.train,
            config=MAGNITUDE_CONFIG,
        )
        train_magnitude_predictions = forecast_hist_gradient_magnitudes(
            magnitude_model,
            state_fold.train,
            include_actual=False,
        )
        train_predicted_bps = [
            float(prediction.expected_magnitude_bps)
            for prediction in train_magnitude_predictions
        ]
        predicted_gate_threshold_bps = float(
            np.quantile(train_predicted_bps, GATE_PREDICTION_QUANTILE)
        )

        selective_model, selective_fit = fit_unified_linear_baseline(
            tail_train,
            config=LINEAR_CONFIG,
        )
        original_model, original_fit = fit_unified_linear_baseline(
            original_fold.train,
            config=LINEAR_CONFIG,
        )

        validation_magnitude_predictions = forecast_hist_gradient_magnitudes(
            magnitude_model,
            state_fold.validation,
            include_actual=False,
        )
        selected_validation = [
            sample
            for sample, prediction in zip(
                state_fold.validation,
                validation_magnitude_predictions,
            )
            if float(prediction.expected_magnitude_bps) >= predicted_gate_threshold_bps
        ]
        if not selected_validation:
            raise ValueError("magnitude gate selected no validation rows")

        selective_predictions = forecast_linear_directions(
            selective_model,
            selected_validation,
            include_actual=True,
        )
        original_predictions = forecast_linear_directions(
            original_model,
            selected_validation,
            include_actual=True,
        )
        selective_metrics = evaluate_direction_predictions(
            selective_predictions,
            null_p_up=tail_null_p_up,
        )
        original_metrics = evaluate_direction_predictions(
            original_predictions,
            null_p_up=tail_null_p_up,
        )

        selected_coverage = len(selected_validation) / len(state_fold.validation)
        selected_mean_actual_magnitude = _mean_actual_magnitude_bps(selected_validation)
        overall_mean_actual_magnitude = _mean_actual_magnitude_bps(state_fold.validation)

        fold_rows.append(
            {
                "fold": state_fold.index,
                "validation_market_dates": list(state_fold.validation_market_dates),
                "actual_tail_threshold_bps": actual_tail_threshold_bps,
                "predicted_gate_threshold_bps": predicted_gate_threshold_bps,
                "tail_training_samples": len(tail_train),
                "tail_null_p_up": tail_null_p_up,
                "magnitude_training_samples": magnitude_fit.training_sample_count,
                "selective_feature_count": selective_fit.feature_count,
                "original_feature_count": original_fit.feature_count,
                "selected_samples": len(selected_validation),
                "validation_samples": len(state_fold.validation),
                "selected_coverage": selected_coverage,
                "selected_directional_samples": int(selective_metrics["evaluated_count"]),
                "selected_mean_actual_magnitude_bps": selected_mean_actual_magnitude,
                "overall_mean_actual_magnitude_bps": overall_mean_actual_magnitude,
                "magnitude_enrichment_ratio": selected_mean_actual_magnitude / overall_mean_actual_magnitude,
                "selective_brier": float(selective_metrics["brier_score"]),
                "original_linear_brier_on_selected": float(original_metrics["brier_score"]),
                "tail_null_brier": float(selective_metrics["null_brier_score"]),
                "selective_log_loss": float(selective_metrics["log_loss"]),
                "original_linear_log_loss_on_selected": float(original_metrics["log_loss"]),
                "tail_null_log_loss": float(selective_metrics["null_log_loss"]),
                "selective_balanced_accuracy": selective_metrics["balanced_accuracy"],
                "original_balanced_accuracy_on_selected": original_metrics["balanced_accuracy"],
                "wins_brier_vs_original": float(selective_metrics["brier_score"]) < float(original_metrics["brier_score"]),
                "wins_brier_vs_null": float(selective_metrics["brier_score"]) < float(selective_metrics["null_brier_score"]),
            }
        )

    mean_selective_brier = fmean(row["selective_brier"] for row in fold_rows)
    mean_original_brier = fmean(row["original_linear_brier_on_selected"] for row in fold_rows)
    mean_null_brier = fmean(row["tail_null_brier"] for row in fold_rows)
    mean_selective_log_loss = fmean(row["selective_log_loss"] for row in fold_rows)
    mean_original_log_loss = fmean(row["original_linear_log_loss_on_selected"] for row in fold_rows)
    mean_null_log_loss = fmean(row["tail_null_log_loss"] for row in fold_rows)
    wins_original = sum(bool(row["wins_brier_vs_original"]) for row in fold_rows)
    wins_null = sum(bool(row["wins_brier_vs_null"]) for row in fold_rows)
    mean_coverage = fmean(row["selected_coverage"] for row in fold_rows)
    mean_enrichment = fmean(row["magnitude_enrichment_ratio"] for row in fold_rows)

    supported = bool(
        wins_original >= MINIMUM_WINS
        and wins_null >= MINIMUM_WINS
        and mean_selective_brier < mean_original_brier
        and mean_selective_brier < mean_null_brier
        and mean_selective_log_loss < mean_original_log_loss
        and mean_selective_log_loss < mean_null_log_loss
        and mean_coverage >= MINIMUM_MEAN_COVERAGE
    )

    payload = {
        "schema_version": 1,
        "experiment": "v4_selective_temporal_direction_15m_v1",
        "dataset_id": manifest.get("dataset_id"),
        "protocol_name": protocol["protocol_name"],
        "protocol_sha256": _canonical_sha256(protocol),
        "base_dataset_protocol": protocol_result.get("protocol_name"),
        "development_amendment": amendment.get("amendment_name"),
        "horizon_minutes": HORIZON_MINUTES,
        "source": "train_split_only",
        "validation_used": False,
        "test_lockbox_accessed": False,
        "pnl_used": False,
        "lineage": {
            "base_direction_experiment": base_report.get("experiment"),
            "context_experiment": context_report.get("experiment"),
        },
        "fold_results": fold_rows,
        "summary": {
            "fold_count": len(fold_rows),
            "brier_fold_wins_vs_original_linear": wins_original,
            "brier_fold_wins_vs_tail_null": wins_null,
            "mean_selective_brier": mean_selective_brier,
            "mean_original_linear_brier_on_selected": mean_original_brier,
            "mean_tail_null_brier": mean_null_brier,
            "mean_selective_log_loss": mean_selective_log_loss,
            "mean_original_linear_log_loss_on_selected": mean_original_log_loss,
            "mean_tail_null_log_loss": mean_null_log_loss,
            "mean_selected_coverage": mean_coverage,
            "mean_magnitude_enrichment_ratio": mean_enrichment,
            "selective_direction_supported": supported,
            "eligible_for_validation_without_separate_freeze": False,
        },
    }

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Evaluate the preregistered train-only v4 selective temporal direction candidate."
    )
    result.add_argument("--dataset-dir", required=True)
    result.add_argument(
        "--base-report",
        default="runs/first-real-direction-v4/nonlinear_direction_development_v1.json",
    )
    result.add_argument(
        "--context-report",
        default="runs/first-real-direction-v4/context_direction_exploration_v1.json",
    )
    result.add_argument(
        "--output",
        default="runs/first-real-direction-v4/selective_direction_development_v1.json",
    )
    return result


def main() -> int:
    args = parser().parse_args()
    payload = run_experiment(
        args.dataset_dir,
        args.base_report,
        args.context_report,
        args.output,
    )
    for row in payload["fold_results"]:
        dates = row["validation_market_dates"]
        print(
            f"FOLD {row['fold']} {dates[0]} -> {dates[-1]} "
            f"coverage={row['selected_coverage']:.3f} "
            f"magnitude_x={row['magnitude_enrichment_ratio']:.3f} "
            f"selective={row['selective_brier']:.6f} "
            f"original={row['original_linear_brier_on_selected']:.6f} "
            f"null={row['tail_null_brier']:.6f}"
        )

    summary = payload["summary"]
    print()
    print("=== V4 SELECTIVE TEMPORAL DIRECTION DEVELOPMENT ===")
    print("Dataset ID:", payload["dataset_id"])
    print("Brier fold wins vs original linear:", f"{summary['brier_fold_wins_vs_original_linear']}/{summary['fold_count']}")
    print("Brier fold wins vs tail null:", f"{summary['brier_fold_wins_vs_tail_null']}/{summary['fold_count']}")
    print("Mean selected coverage:", summary["mean_selected_coverage"])
    print("Mean magnitude enrichment ratio:", summary["mean_magnitude_enrichment_ratio"])
    print("Mean selective Brier:", summary["mean_selective_brier"])
    print("Mean original Brier on selected:", summary["mean_original_linear_brier_on_selected"])
    print("Mean tail null Brier:", summary["mean_tail_null_brier"])
    print("Mean selective log loss:", summary["mean_selective_log_loss"])
    print("Mean original log loss on selected:", summary["mean_original_linear_log_loss_on_selected"])
    print("Mean tail null log loss:", summary["mean_tail_null_log_loss"])
    print("SELECTIVE DIRECTION SUPPORTED:", summary["selective_direction_supported"])
    print("V4 validation: NOT ACCESSED")
    print("V4 test lockbox: NOT ACCESSED")
    print("Eligible for validation now: NO - separate freeze required")
    print("Report:", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
