"""Magnitude-only baseline for leakage-safe OurAI trading research.

The direction head and movement-size head are deliberately separated. This module
predicts the absolute future return ``|r|`` only; it does not choose LONG/SHORT and
it does not consume validation/test data during fitting.

The first challenger is intentionally simple: reuse the deterministic tabular
encoding from :mod:`trading.baseline`, fit ridge regression on absolute return, and
clip negative predictions to zero because a magnitude cannot be negative.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite, sqrt
from statistics import fmean
from typing import Any, Iterable, Mapping

import numpy as np

from .baseline import (
    BaselineConfig,
    BaselineFeatureSchema,
    _discover_schema,
    _fit_numeric_scaler,
    _fit_ridge,
    _normalize_sample,
    _optional_finite_float,
    _target_for,
    _transform_rows,
)
from .dataset import TradingSample


@dataclass(frozen=True)
class MagnitudePrediction:
    symbol: str
    timestamp: Any
    horizon_minutes: int
    expected_magnitude: float
    actual_magnitude: float | None = None

    def __post_init__(self) -> None:
        expected = float(self.expected_magnitude)
        if not isfinite(expected) or expected < 0:
            raise ValueError("expected_magnitude must be finite and non-negative")
        object.__setattr__(self, "expected_magnitude", expected)

        if self.actual_magnitude is not None:
            actual = float(self.actual_magnitude)
            if not isfinite(actual) or actual < 0:
                raise ValueError("actual_magnitude must be finite and non-negative")
            object.__setattr__(self, "actual_magnitude", actual)

    @property
    def expected_magnitude_bps(self) -> float:
        return self.expected_magnitude * 10_000.0

    @property
    def actual_magnitude_bps(self) -> float | None:
        if self.actual_magnitude is None:
            return None
        return self.actual_magnitude * 10_000.0


@dataclass(frozen=True)
class MagnitudeFitSummary:
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


@dataclass(frozen=True)
class UnifiedLinearMagnitudeBaseline:
    config: BaselineConfig
    schema: BaselineFeatureSchema
    numeric_means: tuple[float, ...]
    numeric_scales: tuple[float, ...]
    magnitude_weights_bps: tuple[float, ...]
    training_median_magnitude_bps: float

    def __post_init__(self) -> None:
        width = len(self.schema.expanded_feature_names)
        if len(self.numeric_means) != len(self.schema.numeric_features):
            raise ValueError("numeric_means does not match feature schema")
        if len(self.numeric_scales) != len(self.schema.numeric_features):
            raise ValueError("numeric_scales does not match feature schema")
        if len(self.magnitude_weights_bps) != width:
            raise ValueError("magnitude_weights_bps does not match feature schema")
        if any((not isfinite(value) or value <= 0) for value in self.numeric_scales):
            raise ValueError("numeric_scales must be positive and finite")
        median = float(self.training_median_magnitude_bps)
        if not isfinite(median) or median < 0:
            raise ValueError("training_median_magnitude_bps must be non-negative")
        object.__setattr__(self, "training_median_magnitude_bps", median)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "model_family": "unified_linear_magnitude_baseline",
            "config": asdict(self.config),
            "feature_schema": {
                "numeric_features": list(self.schema.numeric_features),
                "symbols": list(self.schema.symbols),
                "sessions": list(self.schema.sessions),
                "expanded_feature_names": list(self.schema.expanded_feature_names),
            },
            "numeric_means": list(self.numeric_means),
            "numeric_scales": list(self.numeric_scales),
            "magnitude_weights_bps": list(self.magnitude_weights_bps),
            "training_median_magnitude_bps": self.training_median_magnitude_bps,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "UnifiedLinearMagnitudeBaseline":
        if payload.get("model_family") != "unified_linear_magnitude_baseline":
            raise ValueError("unsupported model_family")
        feature_schema = payload["feature_schema"]
        return cls(
            config=BaselineConfig(**dict(payload["config"])),
            schema=BaselineFeatureSchema(
                numeric_features=tuple(feature_schema["numeric_features"]),
                symbols=tuple(feature_schema["symbols"]),
                sessions=tuple(feature_schema["sessions"]),
            ),
            numeric_means=tuple(float(value) for value in payload["numeric_means"]),
            numeric_scales=tuple(float(value) for value in payload["numeric_scales"]),
            magnitude_weights_bps=tuple(
                float(value) for value in payload["magnitude_weights_bps"]
            ),
            training_median_magnitude_bps=float(
                payload["training_median_magnitude_bps"]
            ),
        )

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
            numeric_means=np.asarray(self.numeric_means, dtype=np.float64),
            numeric_scales=np.asarray(self.numeric_scales, dtype=np.float64),
        )
        weights = np.asarray(self.magnitude_weights_bps, dtype=np.float64)
        expected_bps = max(0.0, float(matrix[0] @ weights))

        target = _target_for(row, self.config.horizon_minutes)
        actual_magnitude = None
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


def fit_unified_linear_magnitude_baseline(
    samples: Iterable[TradingSample | Mapping[str, Any]],
    *,
    config: BaselineConfig | None = None,
) -> tuple[UnifiedLinearMagnitudeBaseline, MagnitudeFitSummary]:
    """Fit ridge regression on absolute future returns using training samples only."""

    config = config or BaselineConfig()
    normalized = [_normalize_sample(sample) for sample in samples]
    usable: list[dict[str, Any]] = []
    magnitudes: list[float] = []

    for row in normalized:
        target = _target_for(row, config.horizon_minutes)
        if target is None:
            continue
        future_return = _optional_finite_float(target.get("future_return"))
        if future_return is None:
            continue
        usable.append(row)
        magnitudes.append(abs(future_return))

    if len(usable) < config.minimum_training_samples:
        raise ValueError(
            "not enough resolved training samples for magnitude baseline: "
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
    magnitude_array = np.asarray(magnitudes, dtype=np.float64)
    weights = _fit_ridge(matrix, magnitude_array, l2=config.regression_l2)
    magnitude_bps = magnitude_array * 10_000.0
    training_median_bps = float(np.median(magnitude_bps))
    training_mean_bps = float(np.mean(magnitude_bps))

    model = UnifiedLinearMagnitudeBaseline(
        config=config,
        schema=schema,
        numeric_means=tuple(float(value) for value in means),
        numeric_scales=tuple(float(value) for value in scales),
        magnitude_weights_bps=tuple(float(value) for value in weights),
        training_median_magnitude_bps=training_median_bps,
    )
    summary = MagnitudeFitSummary(
        training_sample_count=len(usable),
        feature_count=len(schema.expanded_feature_names),
        symbols=schema.symbols,
        sessions=schema.sessions,
        horizon_minutes=config.horizon_minutes,
        training_median_magnitude_bps=training_median_bps,
        training_mean_magnitude_bps=training_mean_bps,
    )
    return model, summary


def forecast_magnitudes(
    model: UnifiedLinearMagnitudeBaseline,
    samples: Iterable[TradingSample | Mapping[str, Any]],
    *,
    include_actual: bool = True,
) -> list[MagnitudePrediction]:
    forecasts: list[MagnitudePrediction] = []
    for sample in samples:
        row = _normalize_sample(sample)
        if _target_for(row, model.config.horizon_minutes) is None and include_actual:
            continue
        forecasts.append(model.predict_sample(row, include_actual=include_actual))
    return forecasts


def evaluate_magnitude_predictions(
    predictions: Iterable[MagnitudePrediction],
    *,
    null_magnitude_bps: float,
) -> dict[str, Any]:
    """Compare magnitude forecasts with a fixed train-only constant null.

    For MAE, the scientifically appropriate constant training-only comparator is the
    median absolute return from the training fold. The caller must supply that value;
    this function never derives it from evaluation outcomes.
    """

    null_bps = float(null_magnitude_bps)
    if not isfinite(null_bps) or null_bps < 0:
        raise ValueError("null_magnitude_bps must be finite and non-negative")

    evaluated = [row for row in predictions if row.actual_magnitude is not None]
    if not evaluated:
        raise ValueError("at least one prediction with actual magnitude is required")

    actual_bps = [float(row.actual_magnitude_bps) for row in evaluated]
    predicted_bps = [row.expected_magnitude_bps for row in evaluated]
    model_errors = [abs(predicted - actual) for predicted, actual in zip(predicted_bps, actual_bps)]
    null_errors = [abs(null_bps - actual) for actual in actual_bps]
    model_sq_errors = [
        (predicted - actual) ** 2 for predicted, actual in zip(predicted_bps, actual_bps)
    ]
    null_sq_errors = [(null_bps - actual) ** 2 for actual in actual_bps]

    model_mae = fmean(model_errors)
    null_mae = fmean(null_errors)
    return {
        "schema_version": 1,
        "evaluated_count": len(evaluated),
        "model_mae_bps": model_mae,
        "null_mae_bps": null_mae,
        "mae_delta_bps": model_mae - null_mae,
        "model_wins_mae": model_mae < null_mae,
        "model_rmse_bps": sqrt(fmean(model_sq_errors)),
        "null_rmse_bps": sqrt(fmean(null_sq_errors)),
        "mean_predicted_magnitude_bps": fmean(predicted_bps),
        "mean_actual_magnitude_bps": fmean(actual_bps),
        "null_magnitude_bps": null_bps,
    }
