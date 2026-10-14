"""Nonlinear tabular challengers for leakage-safe OurAI trading research.

The first v3 challenger predicts 15-minute absolute return magnitude with
scikit-learn's histogram gradient boosting regressor. It deliberately reuses the
same causal tabular encoding as the linear baseline so model-family changes are not
confounded with a different feature contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from statistics import fmean
from typing import Any, Iterable, Mapping

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from .baseline import (
    BaselineFeatureSchema,
    _discover_schema,
    _fit_numeric_scaler,
    _normalize_sample,
    _optional_finite_float,
    _target_for,
    _transform_rows,
)
from .dataset import TradingSample
from .magnitude import MagnitudePrediction


@dataclass(frozen=True)
class HistGradientMagnitudeConfig:
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
class HistGradientMagnitudeFitSummary:
    training_sample_count: int
    feature_count: int
    symbols: tuple[str, ...]
    sessions: tuple[str, ...]
    horizon_minutes: int
    training_median_magnitude_bps: float
    training_mean_magnitude_bps: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "training_sample_count": self.training_sample_count,
            "feature_count": self.feature_count,
            "symbols": list(self.symbols),
            "sessions": list(self.sessions),
            "horizon_minutes": self.horizon_minutes,
            "training_median_magnitude_bps": self.training_median_magnitude_bps,
            "training_mean_magnitude_bps": self.training_mean_magnitude_bps,
        }


@dataclass
class UnifiedHistGradientMagnitude:
    config: HistGradientMagnitudeConfig
    schema: BaselineFeatureSchema
    numeric_means: np.ndarray
    numeric_scales: np.ndarray
    estimator: HistGradientBoostingRegressor

    def predict_sample(
        self,
        sample: TradingSample | Mapping[str, Any],
        *,
        include_actual: bool = True,
    ) -> MagnitudePrediction:
        row = _normalize_sample(sample)
        matrix = _transform_rows(
            [row],
            schema=self.schema,
            numeric_means=self.numeric_means,
            numeric_scales=self.numeric_scales,
        )
        expected_bps = max(0.0, float(self.estimator.predict(matrix)[0]))

        actual_magnitude = None
        target = _target_for(row, self.config.horizon_minutes)
        if include_actual and target is not None:
            future_return = _optional_finite_float(target.get("future_return"))
            if future_return is not None:
                actual_magnitude = abs(future_return)

        return MagnitudePrediction(
            symbol=row["symbol"],
            timestamp=row["timestamp"],
            horizon_minutes=self.config.horizon_minutes,
            expected_magnitude=expected_bps / 10_000.0,
            actual_magnitude=actual_magnitude,
        )


def fit_hist_gradient_magnitude(
    samples: Iterable[TradingSample | Mapping[str, Any]],
    *,
    config: HistGradientMagnitudeConfig | None = None,
) -> tuple[UnifiedHistGradientMagnitude, HistGradientMagnitudeFitSummary]:
    """Fit a fixed histogram-gradient-boosting magnitude model on training data."""

    config = config or HistGradientMagnitudeConfig()
    normalized = [_normalize_sample(sample) for sample in samples]
    usable: list[dict[str, Any]] = []
    targets_bps: list[float] = []

    for row in normalized:
        target = _target_for(row, config.horizon_minutes)
        if target is None:
            continue
        future_return = _optional_finite_float(target.get("future_return"))
        if future_return is None:
            continue
        usable.append(row)
        targets_bps.append(abs(future_return) * 10_000.0)

    if len(usable) < config.minimum_training_samples:
        raise ValueError(
            "not enough resolved training samples for nonlinear magnitude model: "
            f"{len(usable)} < {config.minimum_training_samples}"
        )

    schema = _discover_schema(usable)
    means, scales = _fit_numeric_scaler(usable, schema)
    matrix = _transform_rows(
        usable,
        schema=schema,
        numeric_means=means,
        numeric_scales=scales,
    )
    targets = np.asarray(targets_bps, dtype=np.float64)

    estimator = HistGradientBoostingRegressor(
        loss="absolute_error",
        learning_rate=config.learning_rate,
        max_iter=config.max_iter,
        max_leaf_nodes=config.max_leaf_nodes,
        min_samples_leaf=config.min_samples_leaf,
        l2_regularization=config.l2_regularization,
        max_bins=config.max_bins,
        early_stopping=False,
        random_state=config.random_state,
    )
    estimator.fit(matrix, targets)

    model = UnifiedHistGradientMagnitude(
        config=config,
        schema=schema,
        numeric_means=np.asarray(means, dtype=np.float64),
        numeric_scales=np.asarray(scales, dtype=np.float64),
        estimator=estimator,
    )
    summary = HistGradientMagnitudeFitSummary(
        training_sample_count=len(usable),
        feature_count=len(schema.expanded_feature_names),
        symbols=schema.symbols,
        sessions=schema.sessions,
        horizon_minutes=config.horizon_minutes,
        training_median_magnitude_bps=float(np.median(targets)),
        training_mean_magnitude_bps=float(np.mean(targets)),
    )
    return model, summary


def forecast_hist_gradient_magnitudes(
    model: UnifiedHistGradientMagnitude,
    samples: Iterable[TradingSample | Mapping[str, Any]],
    *,
    include_actual: bool = True,
) -> list[MagnitudePrediction]:
    forecasts: list[MagnitudePrediction] = []
    for sample in samples:
        row = _normalize_sample(sample)
        if include_actual and _target_for(row, model.config.horizon_minutes) is None:
            continue
        forecasts.append(model.predict_sample(row, include_actual=include_actual))
    return forecasts


def mean_absolute_error_bps(predictions: Iterable[MagnitudePrediction]) -> float:
    errors = [
        abs(row.expected_magnitude_bps - float(row.actual_magnitude_bps))
        for row in predictions
        if row.actual_magnitude_bps is not None
    ]
    if not errors:
        raise ValueError("at least one resolved prediction is required")
    value = fmean(errors)
    if not isfinite(value):
        raise ValueError("non-finite MAE")
    return value
