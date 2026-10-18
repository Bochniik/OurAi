"""Purged, chronological TRAIN-only OOF inputs for joint-return calibration.

RESEARCH ONLY. Never reads files, journal, or any V3 artifacts. The caller must
load TRAIN samples only; this function rejects any other declared source split.
All model fitting uses historical folds. Held-out fold predictions receive
feature-only objects (with targets removed) before resolved outcomes are read.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from math import isfinite
from typing import Any, Sequence

from .baseline import BaselineConfig, fit_unified_linear_baseline
from .dataset import TradingSample
from .joint_return_calibration import OOFReturnObservation, _is_utc_anchor
from .magnitude import fit_unified_linear_magnitude_baseline
from .validation import purged_walk_forward_folds, sample_label_end, walk_forward_manifest


@dataclass(frozen=True)
class JointOOFConfig:
    horizon_minutes: int = 15
    min_train_market_dates: int = 55
    validation_market_dates: int = 15
    step_market_dates: int = 15
    embargo_minutes: float = 60.0

    def __post_init__(self) -> None:
        if type(self.horizon_minutes) is not int or self.horizon_minutes <= 0:
            raise ValueError("horizon_minutes must be a positive integer")
        for name in ("min_train_market_dates", "validation_market_dates",
                     "step_market_dates"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.step_market_dates < self.validation_market_dates:
            raise ValueError("step_market_dates must be >= validation_market_dates (no repeated OOF rows)")
        if not isfinite(self.embargo_minutes) or self.embargo_minutes < self.horizon_minutes:
            raise ValueError("embargo_minutes must be finite and >= horizon_minutes")


@dataclass(frozen=True)
class JointOOFResult:
    observations: tuple[OOFReturnObservation, ...]
    manifest: dict[str, Any]


def _resolved_horizon_target(
    row: TradingSample, horizon_minutes: int,
) -> float | None:
    target = row.targets.get(f"{horizon_minutes}m")
    if target is None:
        return None
    if not isinstance(target, dict) and not hasattr(target, "get"):
        raise ValueError("15m target must be a mapping")
    end = target.get("future_timestamp")
    if not isinstance(end, (str, datetime)):
        raise ValueError("resolved target must have an explicit future_timestamp")
    try:
        end_time = datetime.fromisoformat(end.replace("Z", "+00:00")) if isinstance(end, str) else end
    except ValueError as exc:
        raise ValueError("invalid resolved future_timestamp") from exc
    if end_time.tzinfo is None or end_time.utcoffset() is None:
        raise ValueError("future_timestamp must be timezone-aware")
    if end_time.astimezone(timezone.utc) != (
        row.timestamp.astimezone(timezone.utc) + timedelta(minutes=horizon_minutes)
    ):
        raise ValueError("resolved target must end at exact forecast horizon")
    realized = target.get("future_return")
    if isinstance(realized, bool):
        raise ValueError("future_return must be finite")
    try:
        number = float(realized)
    except (ValueError, TypeError) as exc:
        raise ValueError("future_return must be finite") from exc
    if not isfinite(number):
        raise ValueError("future_return must be finite")
    if str(target.get("direction") or "").upper() not in {"UP", "DOWN", "NEUTRAL"}:
        raise ValueError("resolved target must have a valid direction")
    return number


def _feature_only(row: TradingSample) -> dict[str, Any]:
    """A model cannot accidentally peek into the held-out outcome."""
    return {
        "symbol": row.symbol,
        "timestamp": row.timestamp,
        "session": row.session,
        "features": dict(row.features),
        "targets": {},
    }


def generate_joint_train_oof(
    train_samples: Sequence[TradingSample],
    *,
    source_split: str,
    config: JointOOFConfig | None = None,
    direction_config: BaselineConfig | None = None,
    magnitude_config: BaselineConfig | None = None,
) -> JointOOFResult:
    """Build original TRAIN-period out-of-fold records for calibration, not validation.

    The two frozen *model families* are the existing tabular logistic UP/DOWN
    baseline and ridge absolute-return magnitude baseline. They are refitted
    independently within each fold. No held-out target is passed to prediction.

    source_split='train' is an explicit contract, not proof of provenance:
    upstream callers must audit the actual dataset path/manifest.
    """

    if source_split != "train":
        raise ValueError("only the train split may generate calibration OOF data")
    config = config or JointOOFConfig()
    direction_config = direction_config or BaselineConfig(horizon_minutes=config.horizon_minutes)
    magnitude_config = magnitude_config or BaselineConfig(horizon_minutes=config.horizon_minutes)
    if (direction_config.horizon_minutes != config.horizon_minutes or
            magnitude_config.horizon_minutes != config.horizon_minutes):
        raise ValueError("direction, magnitude and OOF horizons must match")
    if not train_samples:
        raise ValueError("empty train sample input")
    seen: set[tuple[str, datetime]] = set()
    for row in train_samples:
        if not isinstance(row, TradingSample):
            raise TypeError("train_samples must contain TradingSample instances")
        if row.timestamp.tzinfo is None or row.timestamp.utcoffset() is None:
            raise ValueError("sample timestamp must be timezone-aware")
        unique = (row.symbol.upper(), row.timestamp.astimezone(timezone.utc))
        if unique in seen:
            raise ValueError("duplicate symbol/timestamp in train sample input")
        seen.add(unique)
        _resolved_horizon_target(row, config.horizon_minutes)  # require label maturity metadata

    folds = purged_walk_forward_folds(
        train_samples,
        min_train_market_dates=config.min_train_market_dates,
        validation_market_dates=config.validation_market_dates,
        step_market_dates=config.step_market_dates,
        embargo_minutes=config.embargo_minutes,
    )
    observations: list[OOFReturnObservation] = []
    fold_summaries: list[dict[str, Any]] = []
    used: set[tuple[str, datetime]] = set()
    for fold in folds:
        if max(sample_label_end(row) for row in fold.train) >= fold.validation_start:
            raise AssertionError("training label reaches held-out fold")
        chosen: list[tuple[TradingSample, float]] = []
        for row in fold.validation:
            if not _is_utc_anchor(row.timestamp, config.horizon_minutes):
                continue
            actual = _resolved_horizon_target(row, config.horizon_minutes)
            if actual is not None:
                chosen.append((row, actual))
        if not chosen:
            raise ValueError(f"fold {fold.index} has no resolved non-overlap anchors")

        direction_model, direction_fit = fit_unified_linear_baseline(
            fold.train, config=direction_config
        )
        magnitude_model, magnitude_fit = fit_unified_linear_magnitude_baseline(
            fold.train, config=magnitude_config
        )

        for row, actual in chosen:
            key = (row.symbol.upper(), row.timestamp.astimezone(timezone.utc))
            if key in used:
                raise ValueError("duplicate OOF timestamp across folds")
            used.add(key)
            scrubbed = _feature_only(row)
            up = direction_model.predict_sample(scrubbed, include_actual=False)
            magnitude = magnitude_model.predict_sample(scrubbed, include_actual=False)
            observations.append(OOFReturnObservation(
                symbol=row.symbol,
                timestamp=row.timestamp,
                horizon_minutes=config.horizon_minutes,
                raw_p_up=up.p_up,
                predicted_abs_return=magnitude.expected_magnitude,
                realized_return=actual,
                is_train_oof=True,
            ))
        fold_summaries.append({
            "index": fold.index,
            "train_samples": len(fold.train),
            "validation_samples": len(fold.validation),
            "selected_nonoverlap_oof_rows": len(chosen),
            "purged_overlap_count": fold.purged_overlap_count,
            "embargoed_count": fold.embargoed_count,
            "direction_fit": direction_fit.to_dict(),
            "magnitude_fit": magnitude_fit.to_dict(),
        })

    observations.sort(key=lambda row: (row.timestamp, row.symbol))
    if not observations:
        raise ValueError("no OOF observations generated")
    return JointOOFResult(
        observations=tuple(observations),
        manifest={
            "schema_version": 1,
            "source_split": "train",
            "forward_or_test_accessed": False,
            "horizon_minutes": config.horizon_minutes,
            "oof_observation_count": len(observations),
            "direction_model_family": "unified_linear_baseline",
            "magnitude_model_family": "unified_linear_magnitude_baseline",
            "direction_config": asdict(direction_config),
            "magnitude_config": asdict(magnitude_config),
            "oof_config": asdict(config),
            "anchor_rule": "UTC epoch divisible by horizon; no realized-outcome selection",
            "walk_forward": walk_forward_manifest(folds),
            "fold_summaries": fold_summaries,
        },
    )
