"""First unified tabular baseline for the trading-focused OurAI project.

The model is deliberately simple, reproducible, and dependency-light:

* one shared regularized logistic model estimates P(UP) vs P(DOWN);
* one shared ridge model estimates future return magnitude;
* ticker and session are context features inside the same model rather than separate
  production engines;
* numeric missingness is explicit and fitted only from the training split.

This is a scientific floor, not a claim of edge. More expressive boosted-tree and
temporal models should only replace it after leakage-safe held-out comparisons.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from math import isfinite
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np

from .dataset import TradingSample
from .evaluation import ForecastPrediction


@dataclass(frozen=True)
class BaselineConfig:
    horizon_minutes: int = 15
    probability_l2: float = 1e-3
    probability_learning_rate: float = 0.05
    probability_epochs: int = 120
    probability_batch_size: int = 4096
    regression_l2: float = 1.0
    minimum_training_samples: int = 40
    seed: int = 42

    def __post_init__(self) -> None:
        if int(self.horizon_minutes) <= 0:
            raise ValueError("horizon_minutes must be positive")
        if float(self.probability_l2) < 0 or float(self.regression_l2) < 0:
            raise ValueError("regularization must be non-negative")
        if float(self.probability_learning_rate) <= 0:
            raise ValueError("probability_learning_rate must be positive")
        if int(self.probability_epochs) <= 0:
            raise ValueError("probability_epochs must be positive")
        if int(self.probability_batch_size) <= 0:
            raise ValueError("probability_batch_size must be positive")
        if int(self.minimum_training_samples) < 4:
            raise ValueError("minimum_training_samples must be at least 4")


@dataclass(frozen=True)
class BaselineFeatureSchema:
    numeric_features: tuple[str, ...]
    symbols: tuple[str, ...]
    sessions: tuple[str, ...]

    @property
    def expanded_feature_names(self) -> tuple[str, ...]:
        names = ["intercept"]
        names.extend(f"num__{name}" for name in self.numeric_features)
        names.extend(f"missing__{name}" for name in self.numeric_features)
        names.extend(f"symbol__{symbol}" for symbol in self.symbols)
        names.extend(f"session__{session}" for session in self.sessions)
        return tuple(names)


@dataclass(frozen=True)
class BaselineFitSummary:
    training_sample_count: int
    directional_sample_count: int
    neutral_sample_count: int
    feature_count: int
    symbols: tuple[str, ...]
    sessions: tuple[str, ...]
    horizon_minutes: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "training_sample_count": self.training_sample_count,
            "directional_sample_count": self.directional_sample_count,
            "neutral_sample_count": self.neutral_sample_count,
            "feature_count": self.feature_count,
            "symbols": list(self.symbols),
            "sessions": list(self.sessions),
            "horizon_minutes": self.horizon_minutes,
        }


@dataclass(frozen=True)
class UnifiedLinearBaseline:
    config: BaselineConfig
    schema: BaselineFeatureSchema
    numeric_means: tuple[float, ...]
    numeric_scales: tuple[float, ...]
    probability_weights: tuple[float, ...]
    return_weights_bps: tuple[float, ...]

    def __post_init__(self) -> None:
        width = len(self.schema.expanded_feature_names)
        if len(self.numeric_means) != len(self.schema.numeric_features):
            raise ValueError("numeric_means does not match feature schema")
        if len(self.numeric_scales) != len(self.schema.numeric_features):
            raise ValueError("numeric_scales does not match feature schema")
        if len(self.probability_weights) != width:
            raise ValueError("probability_weights does not match expanded feature schema")
        if len(self.return_weights_bps) != width:
            raise ValueError("return_weights_bps does not match expanded feature schema")
        if any((not isfinite(value) or value <= 0) for value in self.numeric_scales):
            raise ValueError("numeric_scales must be positive and finite")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "model_family": "unified_linear_baseline",
            "config": asdict(self.config),
            "feature_schema": {
                "numeric_features": list(self.schema.numeric_features),
                "symbols": list(self.schema.symbols),
                "sessions": list(self.schema.sessions),
                "expanded_feature_names": list(self.schema.expanded_feature_names),
            },
            "numeric_means": list(self.numeric_means),
            "numeric_scales": list(self.numeric_scales),
            "probability_weights": list(self.probability_weights),
            "return_weights_bps": list(self.return_weights_bps),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "UnifiedLinearBaseline":
        if payload.get("model_family") != "unified_linear_baseline":
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
            probability_weights=tuple(float(value) for value in payload["probability_weights"]),
            return_weights_bps=tuple(float(value) for value in payload["return_weights_bps"]),
        )

    def predict_sample(
        self,
        sample: TradingSample | Mapping[str, Any],
        *,
        include_actual: bool = True,
    ) -> ForecastPrediction:
        row = _normalize_sample(sample)
        matrix = _transform_rows(
            [row],
            schema=self.schema,
            numeric_means=np.asarray(self.numeric_means, dtype=np.float64),
            numeric_scales=np.asarray(self.numeric_scales, dtype=np.float64),
        )
        probability_weight = np.asarray(self.probability_weights, dtype=np.float64)
        return_weight = np.asarray(self.return_weights_bps, dtype=np.float64)

        logit = float(matrix[0] @ probability_weight)
        p_up = float(_sigmoid(np.asarray([logit], dtype=np.float64))[0])
        expected_return = float(matrix[0] @ return_weight) / 10_000.0

        target = _target_for(row, self.config.horizon_minutes)
        actual_return = None
        if include_actual and target is not None:
            actual_return = float(target["future_return"])

        spread = _optional_finite_float(row["features"].get("spread_bps"))
        return ForecastPrediction(
            symbol=row["symbol"],
            timestamp=row["timestamp"],
            horizon_minutes=self.config.horizon_minutes,
            p_up=p_up,
            expected_return=expected_return,
            actual_return=actual_return,
            spread_bps=spread,
        )


def _optional_finite_float(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if isfinite(numeric) else None


def _normalize_sample(sample: TradingSample | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(sample, TradingSample):
        return {
            "symbol": sample.symbol,
            "timestamp": sample.timestamp,
            "session": sample.session,
            "features": dict(sample.features),
            "targets": {key: dict(value) for key, value in sample.targets.items()},
        }
    if not isinstance(sample, Mapping):
        raise TypeError("sample must be TradingSample or mapping")

    timestamp = sample.get("timestamp")
    if isinstance(timestamp, str):
        timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    if not isinstance(timestamp, datetime) or timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ValueError("sample timestamp must be timezone-aware")

    symbol = str(sample.get("symbol") or "").strip().upper()
    session = str(sample.get("session") or "").strip().lower()
    features = sample.get("features")
    targets = sample.get("targets", {})
    if not symbol:
        raise ValueError("sample symbol is required")
    if not session:
        raise ValueError("sample session is required")
    if not isinstance(features, Mapping):
        raise ValueError("sample features must be a mapping")
    if not isinstance(targets, Mapping):
        raise ValueError("sample targets must be a mapping")

    return {
        "symbol": symbol,
        "timestamp": timestamp,
        "session": session,
        "features": dict(features),
        "targets": {str(key): dict(value) for key, value in targets.items()},
    }


def _target_for(row: Mapping[str, Any], horizon_minutes: int) -> Optional[Mapping[str, Any]]:
    target = row["targets"].get(f"{int(horizon_minutes)}m")
    return target if isinstance(target, Mapping) else None


def _discover_schema(rows: Sequence[Mapping[str, Any]]) -> BaselineFeatureSchema:
    candidate_names = sorted(
        {
            str(name)
            for row in rows
            for name in row["features"].keys()
        }
    )
    numeric_features: list[str] = []
    for name in candidate_names:
        observed = [
            row["features"].get(name)
            for row in rows
            if row["features"].get(name) is not None
        ]
        if not observed:
            continue
        if all(_optional_finite_float(value) is not None for value in observed):
            numeric_features.append(name)

    if not numeric_features:
        raise ValueError("no numeric features are available for the baseline")

    return BaselineFeatureSchema(
        numeric_features=tuple(numeric_features),
        symbols=tuple(sorted({row["symbol"] for row in rows})),
        sessions=tuple(sorted({row["session"] for row in rows})),
    )


def _fit_numeric_scaler(
    rows: Sequence[Mapping[str, Any]],
    schema: BaselineFeatureSchema,
) -> tuple[np.ndarray, np.ndarray]:
    raw = np.full((len(rows), len(schema.numeric_features)), np.nan, dtype=np.float64)
    for row_index, row in enumerate(rows):
        features = row["features"]
        for column_index, name in enumerate(schema.numeric_features):
            value = _optional_finite_float(features.get(name))
            if value is not None:
                raw[row_index, column_index] = value

    means = np.zeros(raw.shape[1], dtype=np.float64)
    scales = np.ones(raw.shape[1], dtype=np.float64)
    for column in range(raw.shape[1]):
        values = raw[:, column]
        present = values[np.isfinite(values)]
        if present.size == 0:
            continue
        means[column] = float(np.mean(present))
        standard_deviation = float(np.std(present))
        if isfinite(standard_deviation) and standard_deviation > 1e-12:
            scales[column] = standard_deviation
    return means, scales


def _transform_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    schema: BaselineFeatureSchema,
    numeric_means: np.ndarray,
    numeric_scales: np.ndarray,
) -> np.ndarray:
    numeric_count = len(schema.numeric_features)
    width = len(schema.expanded_feature_names)
    matrix = np.zeros((len(rows), width), dtype=np.float32)
    matrix[:, 0] = 1.0

    symbol_offsets = {
        symbol: 1 + numeric_count * 2 + index
        for index, symbol in enumerate(schema.symbols)
    }
    session_base = 1 + numeric_count * 2 + len(schema.symbols)
    session_offsets = {
        session: session_base + index
        for index, session in enumerate(schema.sessions)
    }

    for row_index, row in enumerate(rows):
        features = row["features"]
        for column_index, name in enumerate(schema.numeric_features):
            value = _optional_finite_float(features.get(name))
            value_offset = 1 + column_index
            missing_offset = 1 + numeric_count + column_index
            if value is None:
                matrix[row_index, value_offset] = 0.0
                matrix[row_index, missing_offset] = 1.0
            else:
                matrix[row_index, value_offset] = (
                    (value - numeric_means[column_index]) / numeric_scales[column_index]
                )

        symbol_offset = symbol_offsets.get(row["symbol"])
        if symbol_offset is not None:
            matrix[row_index, symbol_offset] = 1.0
        session_offset = session_offsets.get(row["session"])
        if session_offset is not None:
            matrix[row_index, session_offset] = 1.0
    return matrix


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _fit_logistic(
    matrix: np.ndarray,
    labels: np.ndarray,
    *,
    config: BaselineConfig,
) -> np.ndarray:
    indices = np.flatnonzero(labels >= 0)
    if indices.size < config.minimum_training_samples:
        raise ValueError(
            "not enough directional samples for probability baseline: "
            f"{indices.size} < {config.minimum_training_samples}"
        )
    directional_labels = labels[indices]
    if len(np.unique(directional_labels)) < 2:
        raise ValueError("probability baseline requires both UP and DOWN training examples")

    weights = np.zeros(matrix.shape[1], dtype=np.float64)
    rng = np.random.default_rng(config.seed)
    batch_size = min(config.probability_batch_size, indices.size)

    for _ in range(config.probability_epochs):
        shuffled = np.array(indices, copy=True)
        rng.shuffle(shuffled)
        for start in range(0, shuffled.size, batch_size):
            batch_indices = shuffled[start : start + batch_size]
            batch = matrix[batch_indices].astype(np.float64, copy=False)
            outcomes = labels[batch_indices].astype(np.float64, copy=False)
            probabilities = _sigmoid(batch @ weights)
            gradient = (batch.T @ (probabilities - outcomes)) / batch.shape[0]
            penalty = config.probability_l2 * weights
            penalty[0] = 0.0
            weights -= config.probability_learning_rate * (gradient + penalty)
    return weights


def _fit_ridge(
    matrix: np.ndarray,
    returns: np.ndarray,
    *,
    l2: float,
    chunk_size: int = 50_000,
) -> np.ndarray:
    width = matrix.shape[1]
    xtx = np.zeros((width, width), dtype=np.float64)
    xty = np.zeros(width, dtype=np.float64)
    targets_bps = returns.astype(np.float64, copy=False) * 10_000.0

    for start in range(0, matrix.shape[0], chunk_size):
        stop = min(matrix.shape[0], start + chunk_size)
        batch = matrix[start:stop].astype(np.float64, copy=False)
        target = targets_bps[start:stop]
        xtx += batch.T @ batch
        xty += batch.T @ target

    penalty = np.eye(width, dtype=np.float64) * float(l2)
    penalty[0, 0] = 0.0
    try:
        return np.linalg.solve(xtx + penalty, xty)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(xtx + penalty) @ xty


def fit_unified_linear_baseline(
    samples: Iterable[TradingSample | Mapping[str, Any]],
    *,
    config: BaselineConfig | None = None,
) -> tuple[UnifiedLinearBaseline, BaselineFitSummary]:
    """Fit the first shared OurAI direction + return baseline on training samples only."""

    config = config or BaselineConfig()
    normalized = [_normalize_sample(sample) for sample in samples]
    usable: list[dict[str, Any]] = []
    directions: list[int] = []
    returns: list[float] = []

    for row in normalized:
        target = _target_for(row, config.horizon_minutes)
        if target is None:
            continue
        future_return = _optional_finite_float(target.get("future_return"))
        direction = str(target.get("direction") or "").strip().upper()
        if future_return is None or direction not in {"UP", "DOWN", "NEUTRAL"}:
            continue
        usable.append(row)
        returns.append(future_return)
        directions.append(1 if direction == "UP" else 0 if direction == "DOWN" else -1)

    if len(usable) < config.minimum_training_samples:
        raise ValueError(
            "not enough resolved training samples for baseline: "
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
    direction_array = np.asarray(directions, dtype=np.int8)
    return_array = np.asarray(returns, dtype=np.float64)

    probability_weights = _fit_logistic(matrix, direction_array, config=config)
    return_weights = _fit_ridge(matrix, return_array, l2=config.regression_l2)

    model = UnifiedLinearBaseline(
        config=config,
        schema=schema,
        numeric_means=tuple(float(value) for value in means),
        numeric_scales=tuple(float(value) for value in scales),
        probability_weights=tuple(float(value) for value in probability_weights),
        return_weights_bps=tuple(float(value) for value in return_weights),
    )
    directional_count = int(np.sum(direction_array >= 0))
    summary = BaselineFitSummary(
        training_sample_count=len(usable),
        directional_sample_count=directional_count,
        neutral_sample_count=len(usable) - directional_count,
        feature_count=len(schema.expanded_feature_names),
        symbols=schema.symbols,
        sessions=schema.sessions,
        horizon_minutes=config.horizon_minutes,
    )
    return model, summary


def forecast_samples(
    model: UnifiedLinearBaseline,
    samples: Iterable[TradingSample | Mapping[str, Any]],
    *,
    include_actual: bool = True,
) -> list[ForecastPrediction]:
    forecasts: list[ForecastPrediction] = []
    for sample in samples:
        row = _normalize_sample(sample)
        if _target_for(row, model.config.horizon_minutes) is None and include_actual:
            continue
        forecasts.append(model.predict_sample(row, include_actual=include_actual))
    return forecasts
