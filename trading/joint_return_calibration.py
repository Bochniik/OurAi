"""Train-OOF empirical distribution joining direction and magnitude predictions.

EXPERIMENTAL / NOT V3: fit only on *past, out-of-fold TRAIN predictions*.
Returns a coherent signed-return distribution rather than multiplying P(UP) by
an unsigned magnitude (which silently assumes equal conditional magnitudes).
Non-overlapping UTC anchors avoid counting adjacent 15m outcomes as independent.
No validation/test or forward-journal data may be passed to fit.
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from datetime import datetime, timezone
from math import isfinite
from typing import Iterable, Mapping

import numpy as np


# Fixed research bins, not chosen after inspecting holdout scores.
PROBABILITY_EDGES = (0.0, 0.35, 0.50, 0.65, 1.0)
MAGNITUDE_BPS_EDGES = (0.0, 10.0, 25.0, 50.0, float("inf"))


@dataclass(frozen=True)
class JointCalibrationConfig:
    horizon_minutes: int = 15
    minimum_cell_support: int = 30
    neutral_band_bps: float = 3.0

    def __post_init__(self) -> None:
        if isinstance(self.horizon_minutes, bool) or self.horizon_minutes <= 0:
            raise ValueError("horizon_minutes must be positive")
        if isinstance(self.minimum_cell_support, bool) or self.minimum_cell_support < 2:
            raise ValueError("minimum_cell_support must be >= 2")
        if not isfinite(self.neutral_band_bps) or self.neutral_band_bps < 0:
            raise ValueError("neutral_band_bps must be finite and non-negative")


@dataclass(frozen=True)
class OOFReturnObservation:
    symbol: str
    timestamp: datetime
    horizon_minutes: int
    raw_p_up: float
    predicted_abs_return: float
    realized_return: float
    is_train_oof: bool = False

    def __post_init__(self) -> None:
        _check_forecast(self.raw_p_up, self.predicted_abs_return)
        if not isfinite(self.realized_return):
            raise ValueError("realized_return must be finite")
        if not self.symbol.strip():
            raise ValueError("symbol is required")
        if self.timestamp.tzinfo is None or self.timestamp.utcoffset() is None:
            raise ValueError("timestamp must be timezone-aware")
        if not self.is_train_oof:
            raise ValueError("calibration requires explicit train out-of-fold predictions")


@dataclass(frozen=True)
class JointReturnDistribution:
    horizon_minutes: int
    raw_p_up: float
    p_up: float
    p_down: float
    p_neutral: float
    expected_signed_return: float
    expected_abs_return: float
    q10_return: float
    q50_return: float
    q90_return: float
    supporting_observations: int
    method: str = "train_oof_empirical_bucket_v1"

    def price_estimates(self, reference_price: float) -> dict[str, float]:
        """Project signed return quantiles into prices, without promising coverage."""
        price = float(reference_price)
        if not isfinite(price) or price <= 0:
            raise ValueError("reference_price must be positive and finite")
        return {
            "expected_price": price * (1.0 + self.expected_signed_return),
            "q10_price": price * (1.0 + self.q10_return),
            "q50_price": price * (1.0 + self.q50_return),
            "q90_price": price * (1.0 + self.q90_return),
        }


@dataclass(frozen=True)
class JointReturnCalibrator:
    config: JointCalibrationConfig
    # Each summary comes from TRAIN OOF outcomes only; no raw labels retained.
    cells: Mapping[tuple[int, int], tuple[float, ...]]
    included_anchor_count: int
    skipped_nonanchor_count: int

    def predict(
        self, *, raw_p_up: float, predicted_abs_return: float, horizon_minutes: int
    ) -> JointReturnDistribution | None:
        _check_forecast(raw_p_up, predicted_abs_return)
        if horizon_minutes != self.config.horizon_minutes:
            raise ValueError("forecast/calibrator horizon mismatch")
        key = _bucket(raw_p_up, predicted_abs_return)
        values = self.cells.get(key)
        if values is None:
            return None  # Insufficient training evidence: abstain rather than guess.
        p_up, p_down, p_neutral, signed, absolute, q10, q50, q90, count = values
        return JointReturnDistribution(
            horizon_minutes=horizon_minutes,
            raw_p_up=float(raw_p_up),
            p_up=p_up,
            p_down=p_down,
            p_neutral=p_neutral,
            expected_signed_return=signed,
            expected_abs_return=absolute,
            q10_return=q10,
            q50_return=q50,
            q90_return=q90,
            supporting_observations=int(count),
        )


def _check_forecast(p_up: float, magnitude: float) -> None:
    if not isfinite(p_up) or not 0.0 <= p_up <= 1.0:
        raise ValueError("raw_p_up must be finite and in [0, 1]")
    if not isfinite(magnitude) or magnitude < 0:
        raise ValueError("predicted_abs_return must be finite and non-negative")


def _bucket(p_up: float, magnitude: float) -> tuple[int, int]:
    return (
        min(bisect_right(PROBABILITY_EDGES, p_up) - 1, len(PROBABILITY_EDGES) - 2),
        min(bisect_right(MAGNITUDE_BPS_EDGES, magnitude * 10_000.0) - 1,
            len(MAGNITUDE_BPS_EDGES) - 2),
    )


def _is_utc_anchor(timestamp: datetime, horizon_minutes: int) -> bool:
    utc = timestamp.astimezone(timezone.utc)
    # Use wall-clock-independent epoch intervals, not a realized-label gate.
    return int(utc.timestamp()) % (horizon_minutes * 60) == 0 and utc.microsecond == 0


def fit_joint_return_calibrator(
    observations: Iterable[OOFReturnObservation],
    *, config: JointCalibrationConfig | None = None,
) -> JointReturnCalibrator:
    """Build empirical joint distributions from explicitly marked TRAIN OOF rows.

    Callers are responsible for generating genuinely out-of-fold predictions with
    an embargo and ensuring no validation/test/forward outcomes enter this fit.
    Time anchors are chosen independently of all returns and model forecasts.
    """

    config = config or JointCalibrationConfig()
    buckets: dict[tuple[int, int], list[float]] = {}
    seen: set[tuple[str, datetime]] = set()
    included = skipped = 0
    for row in observations:
        if not isinstance(row, OOFReturnObservation):
            raise TypeError("observations must be OOFReturnObservation records")
        if row.horizon_minutes != config.horizon_minutes:
            raise ValueError("mixed calibration horizons are not allowed")
        key = (row.symbol.upper(), row.timestamp.astimezone(timezone.utc))
        if key in seen:
            raise ValueError("duplicate symbol/timestamp in calibration data")
        seen.add(key)
        if not _is_utc_anchor(row.timestamp, config.horizon_minutes):
            skipped += 1
            continue
        included += 1
        buckets.setdefault(_bucket(row.raw_p_up, row.predicted_abs_return), []).append(
            float(row.realized_return)
        )

    band = config.neutral_band_bps / 10_000.0
    cells: dict[tuple[int, int], tuple[float, ...]] = {}
    for key, returns in buckets.items():
        if len(returns) < config.minimum_cell_support:
            continue
        values = np.asarray(returns, dtype=np.float64)
        n = len(values)
        q10, q50, q90 = np.quantile(values, (0.10, 0.50, 0.90))
        cells[key] = (
            float(np.count_nonzero(values > band) / n),
            float(np.count_nonzero(values < -band) / n),
            float(np.count_nonzero(np.abs(values) <= band) / n),
            float(np.mean(values)),
            float(np.mean(np.abs(values))),
            float(q10), float(q50), float(q90), float(n),
        )

    if not cells:
        raise ValueError("no calibration cell meets minimum TRAIN OOF support")
    return JointReturnCalibrator(
        config=config,
        cells=cells,
        included_anchor_count=included,
        skipped_nonanchor_count=skipped,
    )


def evaluate_joint_return_distributions(
    paired_outcomes: Iterable[tuple[JointReturnDistribution, float]],
    *, neutral_band_bps: float = 3.0,
) -> dict[str, float | int]:
    """Score a separate, already resolved evaluation set (never used in fit)."""

    if not isfinite(neutral_band_bps) or neutral_band_bps < 0:
        raise ValueError("neutral_band_bps must be finite and non-negative")
    rows = list(paired_outcomes)
    if not rows:
        raise ValueError("at least one resolved distribution/outcome pair required")
    signed_errors, magnitude_errors, scores, covered = [], [], [], []
    for pred, actual in rows:
        if not isfinite(actual):
            raise ValueError("realized return must be finite")
        signed_errors.append(abs(pred.expected_signed_return - actual) * 10_000)
        magnitude_errors.append(abs(pred.expected_abs_return - abs(actual)) * 10_000)
        band = neutral_band_bps / 10_000
        target = (float(actual > band), float(actual < -band), float(abs(actual) <= band))
        scores.append(sum((prob - y) ** 2 for prob, y in zip(
            (pred.p_up, pred.p_down, pred.p_neutral), target
        )))
        covered.append(float(pred.q10_return <= actual <= pred.q90_return))
    return {
        "evaluated_count": len(rows),
        "signed_return_mae_bps": float(np.mean(signed_errors)),
        "absolute_magnitude_mae_bps": float(np.mean(magnitude_errors)),
        "three_class_brier": float(np.mean(scores)),
        "q10_q90_empirical_coverage": float(np.mean(covered)),
        "nominal_interval_coverage": 0.80,
    }
