"""Read-only TRAIN-period development comparison for the joint-return challenger.

Uses only the *canonical v4 TRAIN split* (plus read-only metadata verification).
Its final 20 TRAIN market dates are a chronological internal development window;
earlier TRAIN dates generate purged OOF calibration and fit incumbent baselines.
Never reads reserved validation/test or V3 forward data. No trading or promotion.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from math import isfinite
from pathlib import Path
from statistics import mean, median
from typing import Any, Sequence

from scripts.evaluate_magnitude_head import _load_train
from scripts.evaluate_v4_nonlinear_direction import _validate_v4_contract
from trading.baseline import BaselineConfig, fit_unified_linear_baseline
from trading.dataset import TradingSample
from trading.joint_oof import (
    JointOOFConfig,
    _feature_only,
    _resolved_horizon_target,
    generate_joint_train_oof,
)
from trading.joint_return_calibration import (
    JointCalibrationConfig,
    JointReturnDistribution,
    _is_utc_anchor,
    fit_joint_return_calibrator,
)
from trading.magnitude import fit_unified_linear_magnitude_baseline
from trading.validation import purged_walk_forward_folds, sample_label_end


@dataclass(frozen=True)
class JointDevelopmentConfig:
    horizon_minutes: int = 15
    evaluation_market_dates: int = 20
    embargo_minutes: float = 60.0

    def __post_init__(self) -> None:
        if type(self.horizon_minutes) is not int or self.horizon_minutes <= 0:
            raise ValueError("horizon_minutes must be a positive integer")
        if type(self.evaluation_market_dates) is not int or self.evaluation_market_dates < 1:
            raise ValueError("evaluation_market_dates must be positive")
        if not isfinite(self.embargo_minutes) or self.embargo_minutes < self.horizon_minutes:
            raise ValueError("embargo must be finite and >= the forecast horizon")


def _metric_mae(pairs: Sequence[tuple[float, float]]) -> float:
    if not pairs:
        raise ValueError("cannot evaluate empty prediction/outcome pairs")
    return float(mean(abs(pred - actual) * 10_000.0 for pred, actual in pairs))


def _three_class(realized: float, band: float) -> tuple[float, float, float]:
    return (float(realized > band), float(realized < -band), float(abs(realized) <= band))


def _brier(probabilities: Sequence[float], actual: float, band: float) -> float:
    return sum((prob - expected) ** 2 for prob, expected in
               zip(probabilities, _three_class(actual, band)))


def evaluate_train_only_joint(
    train_samples: Sequence[TradingSample],
    *,
    source_split: str,
    config: JointDevelopmentConfig | None = None,
    oof_config: JointOOFConfig | None = None,
    calibration_config: JointCalibrationConfig | None = None,
    direction_config: BaselineConfig | None = None,
    magnitude_config: BaselineConfig | None = None,
) -> dict[str, Any]:
    """Hold out the final TRAIN dates *before* any OOF/calibration fitting.

    This is exploratory TRAIN development, not an untouched test result.
    All choices default to fixed, declared settings; caller changes must be
    reported and must never be selected on reserved validation/test outcomes.
    """

    if source_split != "train":
        raise ValueError("only source_split='train' is permitted")
    config = config or JointDevelopmentConfig()
    oof_config = oof_config or JointOOFConfig(horizon_minutes=config.horizon_minutes)
    calibration_config = calibration_config or JointCalibrationConfig(
        horizon_minutes=config.horizon_minutes,
    )
    direction_config = direction_config or BaselineConfig(
        horizon_minutes=config.horizon_minutes,
    )
    magnitude_config = magnitude_config or BaselineConfig(
        horizon_minutes=config.horizon_minutes,
    )
    if any(value != config.horizon_minutes for value in (
        oof_config.horizon_minutes, calibration_config.horizon_minutes,
        direction_config.horizon_minutes, magnitude_config.horizon_minutes,
    )):
        raise ValueError("all forecasting and calibration horizons must match")
    if oof_config.embargo_minutes < config.horizon_minutes:
        raise ValueError("OOF embargo shorter than forecasting horizon")
    if not train_samples:
        raise ValueError("empty TRAIN split")

    ordered_dates = sorted({row.session_date for row in train_samples})
    min_required = (oof_config.min_train_market_dates +
                    oof_config.validation_market_dates +
                    config.evaluation_market_dates)
    if len(ordered_dates) < min_required:
        raise ValueError(f"not enough TRAIN market dates: need >= {min_required}")

    # Exactly one final internal development block: chronology determined
    # independent of forecasts, labels, or outcomes.
    development = purged_walk_forward_folds(
        train_samples,
        min_train_market_dates=len(ordered_dates) - config.evaluation_market_dates,
        validation_market_dates=config.evaluation_market_dates,
        step_market_dates=config.evaluation_market_dates,
        embargo_minutes=config.embargo_minutes,
    )
    if len(development) != 1:
        raise AssertionError("expected exactly one final TRAIN development block")
    fold = development[0]
    if max(sample_label_end(row) for row in fold.train) >= fold.validation_start:
        raise AssertionError("training outcomes overlap held-out development block")
    if set(fold.train_market_dates) & set(fold.validation_market_dates):
        raise AssertionError("training dates and development dates overlap")

    oof = generate_joint_train_oof(
        fold.train, source_split="train", config=oof_config,
        direction_config=direction_config, magnitude_config=magnitude_config,
    )
    calibrator = fit_joint_return_calibrator(
        oof.observations, config=calibration_config
    )

    # Refit incumbents *only* on earlier TRAIN history.
    signed_model, _ = fit_unified_linear_baseline(fold.train, config=direction_config)
    abs_model, _ = fit_unified_linear_magnitude_baseline(fold.train,
                                                       config=magnitude_config)

    band = calibration_config.neutral_band_bps / 10_000.0
    # Fixed train-only constants; median is the optimal constant for MAE.
    train_returns = [
        actual for row in fold.train
        if _is_utc_anchor(row.timestamp, config.horizon_minutes)
        if (actual := _resolved_horizon_target(row, config.horizon_minutes)) is not None
    ]
    if not train_returns:
        raise ValueError("no resolved, anchored TRAIN outcomes for null baselines")
    median_signed = float(median(train_returns))
    median_abs = float(median(abs(value) for value in train_returns))
    prior = tuple(
        mean(_three_class(actual, band)[class_index] for actual in train_returns)
        for class_index in range(3)
    )

    evaluation_rows: list[tuple[TradingSample, float]] = []
    for row in fold.validation:
        if not _is_utc_anchor(row.timestamp, config.horizon_minutes):
            continue
        actual = _resolved_horizon_target(row, config.horizon_minutes)
        if actual is not None:
            evaluation_rows.append((row, actual))
    if not evaluation_rows:
        raise ValueError("no resolved, anchored development outcomes")

    signed_baselines_all: list[tuple[float, float]] = []
    abs_baselines_all: list[tuple[float, float]] = []
    supported: list[tuple[JointReturnDistribution, float, float, float]] = []
    for row, actual in evaluation_rows:
        # Critical: discard targets before either predictor sees the row.
        features = _feature_only(row)
        signed = signed_model.predict_sample(features, include_actual=False)
        magnitude = abs_model.predict_sample(features, include_actual=False)
        signed_baselines_all.append((signed.expected_return, actual))
        abs_baselines_all.append((magnitude.expected_magnitude, abs(actual)))
        distribution = calibrator.predict(
            raw_p_up=signed.p_up,
            predicted_abs_return=magnitude.expected_magnitude,
            horizon_minutes=config.horizon_minutes,
        )
        if distribution is not None:
            supported.append((
                distribution, actual, signed.expected_return,
                magnitude.expected_magnitude,
            ))

    covered = len(supported)
    total = len(evaluation_rows)
    result: dict[str, Any] = {
        "schema_version": 1,
        "experiment": "joint_return_train_chronological_development_v1",
        "source": "v4_train_split_only",
        "independent_test": False,
        "original_validation_used": False,
        "test_lockbox_accessed": False,
        "v3_forward_used": False,
        "production_promotion_allowed": False,
        "config": {
            "development": asdict(config),
            "oof": asdict(oof_config),
            "calibration": asdict(calibration_config),
            "direction": asdict(direction_config),
            "magnitude": asdict(magnitude_config),
        },
        "train_market_dates": list(fold.train_market_dates),
        "development_market_dates": list(fold.validation_market_dates),
        "train_rows_after_purge": len(fold.train),
        "development_rows": len(fold.validation),
        "purged_overlap_count": fold.purged_overlap_count,
        "embargoed_count": fold.embargoed_count,
        "oof_count": len(oof.observations),
        "oof_fold_count": oof.manifest["walk_forward"]["fold_count"],
        "supported_cells": len(calibrator.cells),
        "eligible_nonoverlap_development_count": total,
        "joint_forecast_count": covered,
        "abstained_count": total - covered,
        "coverage_fraction": covered / total,
        "baseline_all_development": {
            "signed_linear_mae_bps": _metric_mae(signed_baselines_all),
            "magnitude_linear_mae_bps": _metric_mae(abs_baselines_all),
            "train_median_signed_mae_bps": _metric_mae(
                [(median_signed, actual) for _, actual in signed_baselines_all]
            ),
            "train_median_magnitude_mae_bps": _metric_mae(
                [(median_abs, actual) for _, actual in abs_baselines_all]
            ),
        },
        "covered_only_metrics": None,
        "conclusion": "INSUFFICIENT_SUPPORT" if not covered else "RESEARCH_ONLY_NO_PROMOTION",
    }

    if covered:
        result["covered_only_metrics"] = {
            "joint_signed_mae_bps": _metric_mae([
                (dist.expected_signed_return, actual)
                for dist, actual, _, _ in supported
            ]),
            "signed_linear_mae_bps": _metric_mae([
                (signed, actual) for _, actual, signed, _ in supported
            ]),
            "train_median_signed_mae_bps": _metric_mae([
                (median_signed, actual) for _, actual, _, _ in supported
            ]),
            "joint_magnitude_mae_bps": _metric_mae([
                (dist.expected_abs_return, abs(actual))
                for dist, actual, _, _ in supported
            ]),
            # MAE should score the conditional median of absolute returns,
            # whereas the existing joint estimate is their mean.
            "joint_mae_optimal_magnitude_mae_bps": _metric_mae([
                (dist.mae_optimal_abs_return, abs(actual))
                for dist, actual, _, _ in supported
            ]),
            "magnitude_linear_mae_bps": _metric_mae([
                (magnitude, abs(actual)) for _, actual, _, magnitude in supported
            ]),
            "train_median_magnitude_mae_bps": _metric_mae([
                (median_abs, abs(actual)) for _, actual, _, _ in supported
            ]),
            "joint_three_class_brier": float(mean(
                _brier((dist.p_up, dist.p_down, dist.p_neutral), actual, band)
                for dist, actual, _, _ in supported
            )),
            "train_prior_three_class_brier": float(mean(
                _brier(prior, actual, band)
                for _, actual, _, _ in supported
            )),
            "q10_q90_empirical_coverage": float(mean(
                float(dist.q10_return <= actual <= dist.q90_return)
                for dist, actual, _, _ in supported
            )),
            "mean_q10_q90_width_bps": float(mean(
                (dist.q90_return - dist.q10_return) * 10_000.0
                for dist, _, _, _ in supported
            )),
            "nominal_interval_coverage": 0.8,
        }
    return result


def run_experiment(
    dataset_dir: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    root = Path(dataset_dir)
    # Existing v4 contract verifies identity, preregistration, and lockbox
    # UNTOUCHED flags. It never opens validation.jsonl or test.jsonl.
    manifest, protocol, _ = _validate_v4_contract(root)
    train_path = root / "train.jsonl"
    if not train_path.is_file():
        raise FileNotFoundError(f"train split not found: {train_path}")
    samples = _load_train(train_path)
    report = evaluate_train_only_joint(samples, source_split="train")
    report["dataset_id"] = manifest["dataset_id"]
    report["protocol_name"] = protocol["protocol_name"]
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="TRAIN-only chronological research comparison: no test/forward access."
    )
    parser.add_argument(
        "--dataset-dir",
        default="data/processed/market/first-real-direction-v4",
    )
    parser.add_argument(
        "--output",
        default="runs/joint-return-development-v1/report.json",
    )
    args = parser.parse_args()
    report = run_experiment(args.dataset_dir, args.output)
    print("=== OURAI JOINT RETURN TRAIN-ONLY DEVELOPMENT ===")
    print("Dataset ID:", report["dataset_id"])
    print("OOF calibration observations:", report["oof_count"])
    print("Evaluation anchors:", report["eligible_nonoverlap_development_count"])
    print("Joint supported:", report["joint_forecast_count"])
    print("Abstained:", report["abstained_count"])
    print("Coverage:", round(report["coverage_fraction"], 4))
    print("Covered-only scores:", json.dumps(report["covered_only_metrics"], indent=2))
    print("NOTE: conditional median |return| is MAE-optimal; old mean kept as expectation.")
    print("All-row baselines:", json.dumps(report["baseline_all_development"], indent=2))
    print("Conclusion:", report["conclusion"], "(NO PRODUCTION PROMOTION)")
    print("V4 validation/test and V3 forward: NOT OPENED")
    print("Report:", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
