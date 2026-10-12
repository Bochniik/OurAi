"""Point-in-time joins for non-price trading context.

This module attaches numeric context from sources such as news, liquidity, market
regime, sector regime, and market-integrity analytics to ``TradingSample`` rows.
The key invariant is availability time: an observation can only affect a sample if
it was actually available at or before that sample timestamp.

Event time and availability time are deliberately separate. A news event may have
happened at 10:00 but only become available to the system at 10:03; a 10:01 sample
must not receive it. Conversely, a scheduled event such as earnings may have an
``event_time`` in the future while its schedule was already known earlier. Causality
therefore depends on ``available_at``, not on the ordering of those two timestamps.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from math import isfinite
import re
from typing import Any, Iterable, Mapping, Optional, Sequence

from .dataset import TradingSample


_VALID_SCOPE = {"global", "sector", "symbol"}
_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_]*$")


def _aware_utc(value: datetime, name: str) -> datetime:
    if not isinstance(value, datetime):
        raise TypeError(f"{name} must be a datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")
    return value.astimezone(timezone.utc)


def _clean_name(value: str, name: str) -> str:
    cleaned = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    if not cleaned or not _NAME_RE.fullmatch(cleaned):
        raise ValueError(f"{name} must contain only lowercase letters, digits and underscores")
    return cleaned


@dataclass(frozen=True)
class ContextObservation:
    """One versioned piece of context known to the system at ``available_at``.

    ``event_time`` describes when the underlying event/state occurred or is
    scheduled to occur. It may be before or after ``available_at``. The only
    timestamp that decides whether a sample may use the observation is
    ``available_at``.

    Scope semantics:
    - ``global``: applies to every symbol (for example SPY/market regime)
    - ``sector``: applies to symbols mapped to the named sector
    - ``symbol``: applies only to the named ticker
    """

    source: str
    available_at: datetime
    features: Mapping[str, Optional[float]]
    event_time: Optional[datetime] = None
    scope: str = "global"
    entity: Optional[str] = None
    valid_until: Optional[datetime] = None
    observation_id: Optional[str] = None

    def __post_init__(self) -> None:
        source = _clean_name(self.source, "source")
        object.__setattr__(self, "source", source)

        scope = str(self.scope or "").strip().lower()
        if scope not in _VALID_SCOPE:
            raise ValueError(f"scope must be one of {sorted(_VALID_SCOPE)}")
        object.__setattr__(self, "scope", scope)

        available_at = _aware_utc(self.available_at, "available_at")
        object.__setattr__(self, "available_at", available_at)

        event_time = self.event_time
        if event_time is not None:
            event_time = _aware_utc(event_time, "event_time")
            object.__setattr__(self, "event_time", event_time)

        valid_until = self.valid_until
        if valid_until is not None:
            valid_until = _aware_utc(valid_until, "valid_until")
            if valid_until < available_at:
                raise ValueError("valid_until cannot be earlier than available_at")
            object.__setattr__(self, "valid_until", valid_until)

        entity = str(self.entity or "").strip().upper() or None
        if scope == "global":
            if entity is not None:
                raise ValueError("global context must not define entity")
        elif entity is None:
            raise ValueError(f"{scope} context requires entity")
        object.__setattr__(self, "entity", entity)

        if not self.features:
            raise ValueError("features must not be empty")
        normalized: dict[str, Optional[float]] = {}
        for key, value in self.features.items():
            feature = _clean_name(str(key), "feature name")
            if feature in normalized:
                raise ValueError(f"duplicate normalized context feature: {feature}")
            if value is None:
                normalized[feature] = None
                continue
            numeric = float(value)
            if not isfinite(numeric):
                raise ValueError(f"context feature {feature} must be finite")
            normalized[feature] = numeric
        object.__setattr__(self, "features", normalized)

        if self.observation_id is not None:
            object.__setattr__(self, "observation_id", str(self.observation_id))


@dataclass(frozen=True)
class ContextSourcePolicy:
    """Optional freshness policy for one context source."""

    max_age_minutes: Optional[float] = None

    def __post_init__(self) -> None:
        if self.max_age_minutes is not None and self.max_age_minutes < 0:
            raise ValueError("max_age_minutes must be non-negative")

    @property
    def max_age(self) -> Optional[timedelta]:
        if self.max_age_minutes is None:
            return None
        return timedelta(minutes=float(self.max_age_minutes))


@dataclass(frozen=True)
class ContextJoinResult:
    samples: tuple[TradingSample, ...]
    audit: tuple[Mapping[str, Any], ...] = field(default_factory=tuple)
    feature_names: tuple[str, ...] = field(default_factory=tuple)


def _scope_matches(
    observation: ContextObservation,
    sample: TradingSample,
    symbol_sectors: Mapping[str, str],
) -> bool:
    if observation.scope == "global":
        return True
    if observation.scope == "symbol":
        return observation.entity == sample.symbol.upper()
    sector = symbol_sectors.get(sample.symbol.upper())
    return sector is not None and observation.entity == sector.upper()


def _eligible(
    observation: ContextObservation,
    sample: TradingSample,
    policy: ContextSourcePolicy,
    symbol_sectors: Mapping[str, str],
) -> bool:
    if not _scope_matches(observation, sample, symbol_sectors):
        return False
    if observation.available_at > sample.timestamp:
        return False
    if observation.valid_until is not None and sample.timestamp > observation.valid_until:
        return False
    if policy.max_age is not None and sample.timestamp - observation.available_at > policy.max_age:
        return False
    return True


def _latest_by_scope(
    observations: Sequence[ContextObservation],
    sample: TradingSample,
    policy: ContextSourcePolicy,
    symbol_sectors: Mapping[str, str],
) -> list[ContextObservation]:
    chosen: dict[str, ContextObservation] = {}
    for observation in observations:
        if not _eligible(observation, sample, policy, symbol_sectors):
            continue
        current = chosen.get(observation.scope)
        candidate_key = (
            observation.available_at,
            observation.event_time or observation.available_at,
            observation.observation_id or "",
        )
        current_key = (
            current.available_at,
            current.event_time or current.available_at,
            current.observation_id or "",
        ) if current is not None else None
        if current_key is None or candidate_key > current_key:
            chosen[observation.scope] = observation
    return [chosen[scope] for scope in ("global", "sector", "symbol") if scope in chosen]


def join_point_in_time_context(
    samples: Sequence[TradingSample],
    observations: Iterable[ContextObservation],
    *,
    symbol_sectors: Optional[Mapping[str, str]] = None,
    policies: Optional[Mapping[str, ContextSourcePolicy]] = None,
    expected_sources: Sequence[str] = (),
) -> ContextJoinResult:
    """Attach the latest *available* context to each trading sample.

    Within one source, matching global context is applied first, then sector context,
    then symbol context. More specific scopes may override a feature with the same
    name. This allows a broad market default plus a symbol-specific refinement while
    preserving a single stable feature namespace.

    The returned feature schema is stable across rows: every feature name observed
    for a source is present on every output row, using ``None`` when no eligible
    observation existed. Each source also gets ``available`` and ``age_minutes``
    features, plus a row-level ``context_missing_fraction``.
    """

    sectors = {
        str(symbol).strip().upper(): str(sector).strip().upper()
        for symbol, sector in (symbol_sectors or {}).items()
        if str(symbol).strip() and str(sector).strip()
    }
    normalized_policies = {
        _clean_name(source, "policy source"): policy
        for source, policy in (policies or {}).items()
    }

    grouped: dict[str, list[ContextObservation]] = {}
    source_features: dict[str, set[str]] = {}
    for observation in observations:
        if not isinstance(observation, ContextObservation):
            raise TypeError("observations must contain ContextObservation instances")
        grouped.setdefault(observation.source, []).append(observation)
        source_features.setdefault(observation.source, set()).update(observation.features)

    source_names = set(grouped)
    source_names.update(_clean_name(source, "expected source") for source in expected_sources)
    ordered_sources = sorted(source_names)

    context_feature_names: list[str] = []
    for source in ordered_sources:
        for feature in sorted(source_features.get(source, set())):
            context_feature_names.append(f"ctx__{source}__{feature}")
        context_feature_names.extend(
            [f"ctx__{source}__available", f"ctx__{source}__age_minutes"]
        )
    if ordered_sources:
        context_feature_names.append("context_missing_fraction")

    outputs: list[TradingSample] = []
    audits: list[Mapping[str, Any]] = []

    for sample in samples:
        if sample.timestamp.tzinfo is None or sample.timestamp.utcoffset() is None:
            raise ValueError("TradingSample timestamp must be timezone-aware")

        features: dict[str, Optional[float]] = dict(sample.features)
        matched_sources = 0
        source_audit: dict[str, Any] = {}

        for source in ordered_sources:
            policy = normalized_policies.get(source, ContextSourcePolicy())
            selected = _latest_by_scope(
                grouped.get(source, ()), sample, policy, sectors
            )

            merged: dict[str, Optional[float]] = {}
            for observation in selected:
                merged.update(observation.features)

            for feature in sorted(source_features.get(source, set())):
                features[f"ctx__{source}__{feature}"] = merged.get(feature)

            if selected:
                matched_sources += 1
                latest_available = max(item.available_at for item in selected)
                age_minutes = (sample.timestamp - latest_available).total_seconds() / 60.0
                features[f"ctx__{source}__available"] = 1.0
                features[f"ctx__{source}__age_minutes"] = max(0.0, age_minutes)
            else:
                features[f"ctx__{source}__available"] = 0.0
                features[f"ctx__{source}__age_minutes"] = None

            source_audit[source] = [
                {
                    "scope": item.scope,
                    "entity": item.entity,
                    "event_time": item.event_time.isoformat() if item.event_time else None,
                    "available_at": item.available_at.isoformat(),
                    "valid_until": item.valid_until.isoformat() if item.valid_until else None,
                    "observation_id": item.observation_id,
                }
                for item in selected
            ]

        if ordered_sources:
            features["context_missing_fraction"] = 1.0 - matched_sources / len(ordered_sources)

        outputs.append(
            TradingSample(
                symbol=sample.symbol,
                timestamp=sample.timestamp,
                session_date=sample.session_date,
                session=sample.session,
                features=features,
                targets=sample.targets,
            )
        )
        audits.append(
            {
                "symbol": sample.symbol,
                "timestamp": sample.timestamp.isoformat(),
                "sources": source_audit,
            }
        )

    return ContextJoinResult(
        samples=tuple(outputs),
        audit=tuple(audits),
        feature_names=tuple(context_feature_names),
    )
