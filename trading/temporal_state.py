"""Causal temporal-state features for short-horizon direction research.

Every feature in this module is derived from the current sample and earlier samples
of the same symbol, market date, and session. Targets and future samples are never
read. Exact-minute lag lookups are used so missing bars are not silently treated as
contiguous history.
"""

from __future__ import annotations

from collections import defaultdict
from math import isfinite, sqrt
from statistics import fmean
from typing import Iterable

from .dataset import TradingSample


def _finite_feature(sample: TradingSample, name: str) -> float | None:
    value = sample.features.get(name)
    if value is None or isinstance(value, bool):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if isfinite(numeric) else None


def _safe_return(current: float | None, previous: float | None) -> float | None:
    if current is None or previous is None or previous <= 0:
        return None
    value = current / previous - 1.0
    return value if isfinite(value) else None


def _std(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    mean = fmean(values)
    variance = fmean((value - mean) ** 2 for value in values)
    return sqrt(variance)


def _linear_slope(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    n = len(values)
    x_mean = (n - 1) / 2.0
    y_mean = fmean(values)
    denominator = sum((index - x_mean) ** 2 for index in range(n))
    if denominator <= 0:
        return None
    numerator = sum(
        (index - x_mean) * (value - y_mean)
        for index, value in enumerate(values)
    )
    return numerator / denominator


def enrich_temporal_state(samples: Iterable[TradingSample]) -> list[TradingSample]:
    """Append deterministic same-symbol temporal features without future leakage."""

    rows = list(samples)
    groups: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    for index, sample in enumerate(rows):
        groups[(sample.symbol, sample.session_date, sample.session)].append(index)

    enriched: list[TradingSample | None] = [None] * len(rows)
    for indices in groups.values():
        ordered = sorted(indices, key=lambda index: rows[index].timestamp)
        by_timestamp = {rows[index].timestamp: index for index in ordered}
        session_open_price = _finite_feature(rows[ordered[0]], "price") if ordered else None

        for index in ordered:
            sample = rows[index]
            current_price = _finite_feature(sample, "price")
            additions: dict[str, float | None] = {}

            for minutes in (2, 3, 10, 20):
                lag_index = by_timestamp.get(
                    sample.timestamp.__class__.fromtimestamp(
                        sample.timestamp.timestamp() - minutes * 60,
                        tz=sample.timestamp.tzinfo,
                    )
                )
                lag_price = (
                    _finite_feature(rows[lag_index], "price")
                    if lag_index is not None
                    else None
                )
                additions[f"temporal_return_{minutes}m"] = _safe_return(
                    current_price, lag_price
                )

            additions["temporal_return_since_session_open"] = _safe_return(
                current_price, session_open_price
            )

            for lag in (1, 2, 3, 5, 10):
                lag_index = by_timestamp.get(
                    sample.timestamp.__class__.fromtimestamp(
                        sample.timestamp.timestamp() - lag * 60,
                        tz=sample.timestamp.tzinfo,
                    )
                )
                additions[f"temporal_return_1m_lag_{lag}"] = (
                    _finite_feature(rows[lag_index], "return_1m")
                    if lag_index is not None
                    else None
                )

            for window in (5, 15):
                recent_returns: list[float] = []
                for lag in range(window - 1, -1, -1):
                    lag_index = by_timestamp.get(
                        sample.timestamp.__class__.fromtimestamp(
                            sample.timestamp.timestamp() - lag * 60,
                            tz=sample.timestamp.tzinfo,
                        )
                    )
                    if lag_index is None:
                        continue
                    value = _finite_feature(rows[lag_index], "return_1m")
                    if value is not None:
                        recent_returns.append(value)
                additions[f"temporal_return_1m_mean_{window}"] = (
                    fmean(recent_returns) if recent_returns else None
                )
                additions[f"temporal_return_1m_std_{window}"] = _std(recent_returns)
                additions[f"temporal_return_1m_slope_{window}"] = _linear_slope(
                    recent_returns
                )
                additions[f"temporal_positive_return_fraction_{window}"] = (
                    fmean(1.0 if value > 0 else 0.0 for value in recent_returns)
                    if recent_returns
                    else None
                )

            return_5m = _finite_feature(sample, "return_5m")
            return_15m = _finite_feature(sample, "return_15m")
            return_30m = _finite_feature(sample, "return_30m")
            additions["temporal_momentum_acceleration_5_vs_15"] = (
                return_5m / 5.0 - return_15m / 15.0
                if return_5m is not None and return_15m is not None
                else None
            )
            additions["temporal_momentum_acceleration_15_vs_30"] = (
                return_15m / 15.0 - return_30m / 30.0
                if return_15m is not None and return_30m is not None
                else None
            )

            lag5_index = by_timestamp.get(
                sample.timestamp.__class__.fromtimestamp(
                    sample.timestamp.timestamp() - 5 * 60,
                    tz=sample.timestamp.tzinfo,
                )
            )
            if lag5_index is not None:
                lag5 = rows[lag5_index]
                for name in (
                    "distance_from_vwap",
                    "range_position_20",
                    "relative_volume_20",
                    "realized_volatility_20",
                ):
                    current = _finite_feature(sample, name)
                    previous = _finite_feature(lag5, name)
                    additions[f"temporal_{name}_delta_5m"] = (
                        current - previous
                        if current is not None and previous is not None
                        else None
                    )
            else:
                for name in (
                    "distance_from_vwap",
                    "range_position_20",
                    "relative_volume_20",
                    "realized_volatility_20",
                ):
                    additions[f"temporal_{name}_delta_5m"] = None

            temporal_values = list(additions.values())
            additions["temporal_state_missing_fraction"] = (
                sum(value is None for value in temporal_values)
                / max(1, len(temporal_values))
            )

            features = dict(sample.features)
            features.update(additions)
            enriched[index] = TradingSample(
                symbol=sample.symbol,
                timestamp=sample.timestamp,
                session_date=sample.session_date,
                session=sample.session,
                features=features,
                targets={key: dict(value) for key, value in sample.targets.items()},
            )

    if any(sample is None for sample in enriched):
        raise AssertionError("temporal-state enrichment left an unprocessed sample")
    return [sample for sample in enriched if sample is not None]
