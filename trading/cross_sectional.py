"""Causal cross-sectional context features for multi-symbol trading research.

Every derived value uses only features from samples carrying the exact same
availability timestamp. The current symbol is excluded from peer aggregates so the
context means market/universe state rather than a disguised copy of the stock's own
feature. No targets or future observations are used.
"""

from __future__ import annotations

from collections import defaultdict
from math import isfinite
from statistics import fmean
from typing import Iterable

from .dataset import TradingSample


RETURN_FEATURES = ("return_1m", "return_5m", "return_15m", "return_30m")
LEVEL_FEATURES = (
    "relative_volume_20",
    "realized_volatility_20",
    "distance_from_vwap",
    "range_position_20",
)


def _finite_feature(sample: TradingSample, name: str) -> float | None:
    value = sample.features.get(name)
    if value is None or isinstance(value, bool):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if isfinite(numeric) else None


def enrich_cross_sectional_context(
    samples: Iterable[TradingSample],
) -> list[TradingSample]:
    """Append peer-only same-timestamp market context to trading samples.

    The function is deterministic and preserves input order. For each sample it adds
    peer means and peer breadth for short-horizon returns, stock-minus-peer excess
    returns, peer means for volume/volatility/VWAP/range context, and a context
    missingness fraction. A peer is another symbol at the exact same availability
    timestamp.
    """

    rows = list(samples)
    by_timestamp: dict[object, list[int]] = defaultdict(list)
    for index, sample in enumerate(rows):
        by_timestamp[sample.timestamp].append(index)

    enriched: list[TradingSample | None] = [None] * len(rows)
    for indices in by_timestamp.values():
        for index in indices:
            sample = rows[index]
            peer_indices = [
                peer_index
                for peer_index in indices
                if peer_index != index and rows[peer_index].symbol != sample.symbol
            ]
            additions: dict[str, float | None] = {}

            for name in RETURN_FEATURES:
                peer_values = [
                    value
                    for peer_index in peer_indices
                    if (value := _finite_feature(rows[peer_index], name)) is not None
                ]
                peer_mean = fmean(peer_values) if peer_values else None
                additions[f"peer_{name}_mean"] = peer_mean
                additions[f"peer_{name}_breadth_up"] = (
                    fmean(1.0 if value > 0 else 0.0 for value in peer_values)
                    if peer_values
                    else None
                )
                own_value = _finite_feature(sample, name)
                additions[f"{name}_excess_vs_peers"] = (
                    own_value - peer_mean
                    if own_value is not None and peer_mean is not None
                    else None
                )

            for name in LEVEL_FEATURES:
                peer_values = [
                    value
                    for peer_index in peer_indices
                    if (value := _finite_feature(rows[peer_index], name)) is not None
                ]
                additions[f"peer_{name}_mean"] = (
                    fmean(peer_values) if peer_values else None
                )

            context_values = list(additions.values())
            additions["cross_sectional_context_missing_fraction"] = (
                sum(value is None for value in context_values)
                / max(1, len(context_values))
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
        raise AssertionError("cross-sectional enrichment left an unprocessed sample")
    return [sample for sample in enriched if sample is not None]
