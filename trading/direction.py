"""Probabilistic UP-vs-DOWN direction models for leakage-safe OurAI research.

The direction head is intentionally conditional on a resolved directional move. Rows
whose target is ``NEUTRAL`` are excluded from classifier fitting and probability
metrics; the separate magnitude/decision layer remains responsible for abstention.

This module contains no trading policy and no PnL-based promotion logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, log
from statistics import fmean
from typing import Any, Iterable, Mapping

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from .baseline import (
    BaselineFeatureSchema,
    UnifiedLinearBaseline,
    _discover_schema,
    _fit_numeric_scaler,
    _normalize_sample,
    _target_for,
    _transform_rows,
)
from .dataset import TradingSample


@dataclass(frozen=True)
class DirectionPrediction:
    symbol: str
    timestamp: Any
    horizon_minutes: int
    p_up: float
    actual_direction: int | None = None

    def __post_init__(self) -> None:
        probability = float(self.p_up)
        if not isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError("p_up must be finite and in [0, 1]")
        object.__setattr__(self, "p_up", probability)
        if self.actual_direction is not None and int(self.actual_direction) not in (0, 1):
            raise ValueError("actual_direction must be 0, 1, or None")
        if self.actual_direction is not None:
            object.__setattr__(self, "actual_direction", int(self.actual_direction))

    @property
    def p_down(self) -> float:
        return 1.0 - self.p_up


@dataclass(frozen=True)
class HistGradientDirectionConfig:
    horizon_minutes: int = 15
    learning_rate: float = 0.05
    max_iter: int = 150
    max_leaf_nodes: int = 31
    min_samples_leaf: int = 100
    l2_regularization: float = 1.0
    max_bins: int = 255
    random_state: int = 42
    minimum_training_samples: int = 100

    def __post_init__(self) -> None:
        if self.horizon_minutes <= 0:
            raise ValueError("horizon_minutes must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.max_iter <= 0 or self.max_leaf_nodes < 2:
            raise ValueError("invalid boosting iteration/tree configuration")
        if self.min_samples_leaf <= 0:
            raise ValueError("min_samples_leaf must be positive")
        if self.l2_regularization < 0:
            raise ValueError("l2_regularization must be non-negative")
        if not 2 <= self.max_bins <= 255:
            raise ValueError("max_bins must be in [2, 255]")
        if self.minimum_training_samples < 20:
            raise ValueError("minimum_training_samples must be at least 20")


@dataclass(frozen=True)
class HistGradientDirectionFitSummary:
    training_directional_sample_count: int
    excluded_neutral_sample_count: int
    feature_count: int
    symbols: tuple[str, ...]
    sessions: tuple[str, ...]
    horizon_minutes: int
    training_up_rate: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "training_directional_sample_count": self.training_directional_sample_count,
            "excluded_neutral_sample_count": self.excluded_neutral_sample_count,
            "feature_count": self.feature_count,
            "symbols": list(self.symbols),
            "sessions": list(self.sessions),
            "horizon_minutes": self.horizon_minutes,
            "training_up_rate": self.training_up_rate,
        }


@dataclass
class UnifiedHistGradientDirection:
    config: HistGradientDirectionConfig
    schema: BaselineFeatureSchema
    numeric_means: np.ndarray
    numeric_scales: np.ndarray
    estimator: HistGradientBoostingClassifier

    def predict_sample(
        self,
        sample: TradingSample | Mapping[str, Any],
        *,
        include_actual: bool = True,
    ) -> DirectionPrediction:
        row = _normalize_sample(sample)
        matrix = _transform_rows(
            [row],
            schema=self.schema,
            numeric_means=self.numeric_means,
            numeric_scales=self.numeric_scales,
        )
        probabilities = self.estimator.predict_proba(matrix)[0]
        classes = list(self.estimator.classes_)
        if 1 not in classes:
            raise ValueError("direction classifier is missing UP class")
        p_up = float(probabilities[classes.index(1)])

        actual = None
        if include_actual:
            target = _target_for(row, self.config.horizon_minutes)
            actual = _direction_label(target)
        return DirectionPrediction(
            symbol=row["symbol"],
            timestamp=row["timestamp"],
            horizon_minutes=self.config.horizon_minutes,
            p_up=p_up,
            actual_direction=actual,
        )


def _direction_label(target: Mapping[str, Any] | None) -> int | None:
    if not isinstance(target, Mapping):
        return None
    direction = str(target.get("direction") or "").strip().upper()
    if direction == "UP":
        return 1
    if direction == "DOWN":
        return 0
    return None


def fit_hist_gradient_direction(
    samples: Iterable[TradingSample | Mapping[str, Any]],
    *,
    config: HistGradientDirectionConfig | None = None,
) -> tuple[UnifiedHistGradientDirection, HistGradientDirectionFitSummary]:
    """Fit the fixed nonlinear conditional UP-vs-DOWN classifier on training data."""

    config = config or HistGradientDirectionConfig()
    normalized = [_normalize_sample(sample) for sample in samples]
    usable: list[dict[str, Any]] = []
    labels: list[int] = []
    neutral_count = 0

    for row in normalized:
        target = _target_for(row, config.horizon_minutes)
        if target is None:
            continue
        label = _direction_label(target)
        if label is None:
            if str(target.get("direction") or "").strip().upper() == "NEUTRAL":
                neutral_count += 1
            continue
        usable.append(row)
        labels.append(label)

    if len(usable) < config.minimum_training_samples:
        raise ValueError(
            "not enough directional training samples for nonlinear direction model: "
            f"{len(usable)} < {config.minimum_training_samples}"
        )
    if len(set(labels)) < 2:
        raise ValueError("direction model requires both UP and DOWN training examples")

    schema = _discover_schema(usable)
    means, scales = _fit_numeric_scaler(usable, schema)
    matrix = _transform_rows(
        usable,
        schema=schema,
        numeric_means=means,
        numeric_scales=scales,
    )
    outcomes = np.asarray(labels, dtype=np.int8)

    estimator = HistGradientBoostingClassifier(
        loss="log_loss",
        learning_rate=config.learning_rate,
        max_iter=config.max_iter,
        max_leaf_nodes=config.max_leaf_nodes,
        min_samples_leaf=config.min_samples_leaf,
        l2_regularization=config.l2_regularization,
        max_bins=config.max_bins,
        early_stopping=False,
        random_state=config.random_state,
    )
    estimator.fit(matrix, outcomes)

    model = UnifiedHistGradientDirection(
        config=config,
        schema=schema,
        numeric_means=np.asarray(means, dtype=np.float64),
        numeric_scales=np.asarray(scales, dtype=np.float64),
        estimator=estimator,
    )
    summary = HistGradientDirectionFitSummary(
        training_directional_sample_count=len(usable),
        excluded_neutral_sample_count=neutral_count,
        feature_count=len(schema.expanded_feature_names),
        symbols=schema.symbols,
        sessions=schema.sessions,
        horizon_minutes=config.horizon_minutes,
        training_up_rate=float(np.mean(outcomes)),
    )
    return model, summary


def forecast_hist_gradient_directions(
    model: UnifiedHistGradientDirection,
    samples: Iterable[TradingSample | Mapping[str, Any]],
    *,
    include_actual: bool = True,
) -> list[DirectionPrediction]:
    predictions: list[DirectionPrediction] = []
    for sample in samples:
        row = _normalize_sample(sample)
        target = _target_for(row, model.config.horizon_minutes)
        if include_actual and _direction_label(target) is None:
            continue
        predictions.append(model.predict_sample(row, include_actual=include_actual))
    return predictions


def forecast_linear_directions(
    model: UnifiedLinearBaseline,
    samples: Iterable[TradingSample | Mapping[str, Any]],
    *,
    include_actual: bool = True,
) -> list[DirectionPrediction]:
    """Expose the existing logistic incumbent through the same direction contract."""

    predictions: list[DirectionPrediction] = []
    for sample in samples:
        row = _normalize_sample(sample)
        target = _target_for(row, model.config.horizon_minutes)
        actual = _direction_label(target)
        if include_actual and actual is None:
            continue
        forecast = model.predict_sample(row, include_actual=False)
        predictions.append(
            DirectionPrediction(
                symbol=row["symbol"],
                timestamp=row["timestamp"],
                horizon_minutes=model.config.horizon_minutes,
                p_up=forecast.p_up,
                actual_direction=actual if include_actual else None,
            )
        )
    return predictions


def evaluate_direction_predictions(
    predictions: Iterable[DirectionPrediction],
    *,
    null_p_up: float,
) -> dict[str, Any]:
    """Evaluate probability quality against a fixed training-only UP-rate null."""

    prior = float(null_p_up)
    if not isfinite(prior) or not 0.0 < prior < 1.0:
        raise ValueError("null_p_up must be finite and strictly between 0 and 1")
    rows = [row for row in predictions if row.actual_direction is not None]
    if not rows:
        raise ValueError("at least one resolved directional prediction is required")

    epsilon = 1e-12
    outcomes = [int(row.actual_direction) for row in rows]
    brier = fmean((row.p_up - outcome) ** 2 for row, outcome in zip(rows, outcomes))
    null_brier = fmean((prior - outcome) ** 2 for outcome in outcomes)

    def binary_log_loss(probability: float, outcome: int) -> float:
        p = min(1.0 - epsilon, max(epsilon, probability))
        return -(outcome * log(p) + (1 - outcome) * log(1.0 - p))

    log_loss = fmean(binary_log_loss(row.p_up, outcome) for row, outcome in zip(rows, outcomes))
    null_log_loss = fmean(binary_log_loss(prior, outcome) for outcome in outcomes)
    accuracy = fmean(int((row.p_up >= 0.5) == bool(outcome)) for row, outcome in zip(rows, outcomes))

    positives = [(row, outcome) for row, outcome in zip(rows, outcomes) if outcome == 1]
    negatives = [(row, outcome) for row, outcome in zip(rows, outcomes) if outcome == 0]
    balanced_accuracy = None
    if positives and negatives:
        true_positive_rate = fmean(int(row.p_up >= 0.5) for row, _ in positives)
        true_negative_rate = fmean(int(row.p_up < 0.5) for row, _ in negatives)
        balanced_accuracy = (true_positive_rate + true_negative_rate) / 2.0

    return {
        "schema_version": 1,
        "evaluated_count": len(rows),
        "brier_score": brier,
        "null_brier_score": null_brier,
        "brier_delta_vs_null": brier - null_brier,
        "log_loss": log_loss,
        "null_log_loss": null_log_loss,
        "log_loss_delta_vs_null": log_loss - null_log_loss,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "mean_p_up": fmean(row.p_up for row in rows),
        "actual_up_rate": fmean(outcomes),
        "null_p_up": prior,
    }


@dataclass(frozen=True)
class JointReturnResearchEstimate:
    """Research-only signed-return estimate from conditional direction magnitudes.

    The magnitude inputs MUST be conditional estimates trained/evaluated for
    their respective directions. Never pass V3 unconditional absolute magnitude
    as both conditional inputs: that would assert an unvalidated independence.
    """

    p_up: float
    up_magnitude_bps: float
    down_magnitude_bps: float
    horizon_minutes: int = 15

    def __post_init__(self) -> None:
        if not isfinite(self.p_up) or not 0.0 <= self.p_up <= 1.0:
            raise ValueError("p_up must be a finite probability")
        for name in ("up_magnitude_bps", "down_magnitude_bps"):
            value = getattr(self, name)
            if not isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.horizon_minutes != 15:
            raise ValueError("joint research contract requires 15-minute horizon")

    @property
    def expected_signed_return_bps(self) -> float:
        return self.p_up * self.up_magnitude_bps - (1.0 - self.p_up) * self.down_magnitude_bps

    @property
    def expected_signed_return_percent(self) -> float:
        return self.expected_signed_return_bps / 100.0
