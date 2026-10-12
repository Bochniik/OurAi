"""Train-prior null benchmarks for validation-only probabilistic trading research.

The UP prior is estimated *only* from the training split, excluding neutral
targets, then held fixed for validation. Never fit it on validation or test.
The comparison is descriptive and does not authorize trading.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any, Iterable, Mapping

from .dataset import TradingSample
from .evaluation import ForecastPrediction, calibration_table, evaluate_forecasts


@dataclass(frozen=True)
class TrainingDirectionPrior:
    horizon_minutes: int
    up_count: int
    down_count: int
    neutral_count: int

    @property
    def p_up(self) -> float:
        directional = self.up_count + self.down_count
        if directional <= 0:
            raise ValueError("training prior requires directional examples")
        return self.up_count / directional

    def to_dict(self) -> dict[str, int | float]:
        return {
            "horizon_minutes": self.horizon_minutes,
            "up_count": self.up_count,
            "down_count": self.down_count,
            "neutral_count": self.neutral_count,
            "p_up": self.p_up,
        }


def training_direction_prior(
    training_samples: Iterable[TradingSample | Mapping[str, Any]],
    *,
    horizon_minutes: int = 15,
) -> TrainingDirectionPrior:
    """Fit the constant UP probability using resolved *training* labels only."""
    if horizon_minutes <= 0:
        raise ValueError("horizon_minutes must be positive")
    counts = {"UP": 0, "DOWN": 0, "NEUTRAL": 0}
    for sample in training_samples:
        targets = sample.targets if isinstance(sample, TradingSample) else sample.get("targets", {})
        target = targets.get(f"{horizon_minutes}m") if isinstance(targets, Mapping) else None
        if not isinstance(target, Mapping):
            continue
        direction = str(target.get("direction") or "").upper()
        if direction in counts:
            counts[direction] += 1
    prior = TrainingDirectionPrior(
        horizon_minutes=horizon_minutes,
        up_count=counts["UP"],
        down_count=counts["DOWN"],
        neutral_count=counts["NEUTRAL"],
    )
    # A single-class training prior is a valid benchmark; no smoothing or
    # validation-based adjustment is performed.
    _ = prior.p_up
    return prior


def _ece(forecasts: list[ForecastPrediction], neutral_threshold_bps: float) -> float | None:
    table = calibration_table(
        forecasts,
        bins=10,
        neutral_threshold_bps=neutral_threshold_bps,
    )
    total = sum(bucket["count"] for bucket in table)
    if not total:
        return None
    return sum(
        bucket["count"] * abs(bucket["mean_p_up"] - bucket["actual_up_rate"])
        for bucket in table
        if bucket["count"]
    ) / total


def compare_to_training_prior(
    validation_forecasts: Iterable[ForecastPrediction],
    *,
    prior: TrainingDirectionPrior,
    neutral_threshold_bps: float = 3.0,
) -> dict[str, Any]:
    """Compare validation forecasts with a frozen train-prior null model.

    Negative deltas for Brier/log-loss/MAE/ECE mean the candidate is better.
    The null model always abstains because its expected return is zero.
    No paths or test-split data are accepted or opened.
    """
    rows = list(validation_forecasts)
    if not rows:
        raise ValueError("validation forecasts must not be empty")
    if any(row.horizon_minutes != prior.horizon_minutes for row in rows):
        raise ValueError("validation and training-prior horizons must match")
    if not isfinite(neutral_threshold_bps) or neutral_threshold_bps < 0:
        raise ValueError("neutral_threshold_bps must be finite and non-negative")

    null_rows = [
        ForecastPrediction(
            symbol=row.symbol,
            timestamp=row.timestamp,
            horizon_minutes=row.horizon_minutes,
            p_up=prior.p_up,
            expected_return=0.0,
            actual_return=row.actual_return,
            spread_bps=row.spread_bps,
        )
        for row in rows
    ]
    candidate = evaluate_forecasts(rows, neutral_threshold_bps=neutral_threshold_bps)
    null = evaluate_forecasts(null_rows, neutral_threshold_bps=neutral_threshold_bps)
    candidate_ece = _ece(rows, neutral_threshold_bps)
    null_ece = _ece(null_rows, neutral_threshold_bps)

    def delta(left: float | None, right: float | None) -> float | None:
        return None if left is None or right is None else left - right

    return {
        "schema_version": 1,
        "purpose": "validation_only_null_comparison",
        "horizon_minutes": prior.horizon_minutes,
        "train_prior": prior.to_dict(),
        "validation_count": candidate["evaluated_count"],
        "directional_count": candidate["directional_count"],
        "candidate": {
            "brier_score": candidate["probability"]["brier_score"],
            "log_loss": candidate["probability"]["log_loss"],
            "direction_accuracy": candidate["probability"]["direction_accuracy"],
            "ece_10_bins": candidate_ece,
            "return_mae_bps": candidate["return"]["mae_bps"],
        },
        "null": {
            "brier_score": null["probability"]["brier_score"],
            "log_loss": null["probability"]["log_loss"],
            "direction_accuracy": null["probability"]["direction_accuracy"],
            "ece_10_bins": null_ece,
            "return_mae_bps": null["return"]["mae_bps"],
        },
        "candidate_minus_null": {
            "brier_score": delta(candidate["probability"]["brier_score"], null["probability"]["brier_score"]),
            "log_loss": delta(candidate["probability"]["log_loss"], null["probability"]["log_loss"]),
            "direction_accuracy": delta(candidate["probability"]["direction_accuracy"], null["probability"]["direction_accuracy"]),
            "ece_10_bins": delta(candidate_ece, null_ece),
            "return_mae_bps": delta(candidate["return"]["mae_bps"], null["return"]["mae_bps"]),
        },
        "notes": [
            "Training prior is fixed before validation; no validation labels are used to fit it.",
            "Neutral outcomes are excluded from binary probability metrics.",
            "Intraday overlapping outcomes are correlated; this is not independent evidence of edge.",
            "No test lockbox is loaded or accessed.",
        ],
    }
