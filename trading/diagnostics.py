"""Deterministic, research-only slices of resolved validation forecasts.

Every slice is evaluated independently using the same cost and neutral-outcome
policy as the headline report. These diagnostics are descriptive; choosing
thresholds from them requires a *new* future validation window.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable

from .dataset import NEW_YORK
from .evaluation import DecisionPolicyConfig, ForecastPrediction, evaluate_forecasts


def _tenth_bin(value: float) -> str:
    index = min(9, max(0, int(value * 10)))
    return f"{index / 10:.1f}-{(index + 1) / 10:.1f}"


def _confidence_bin(value: float) -> str:
    index = min(4, max(0, int((value - 0.5) * 10)))
    return f"{0.5 + index / 10:.1f}-{0.6 + index / 10:.1f}"


def _slice_summary(
    rows: list[ForecastPrediction],
    *,
    policy: DecisionPolicyConfig,
    neutral_threshold_bps: float,
) -> dict[str, Any]:
    report = evaluate_forecasts(
        rows,
        policy=policy,
        neutral_threshold_bps=neutral_threshold_bps,
    )
    probability = report["probability"]
    decisions = report["decision_policy"]
    return {
        "sample_count": report["evaluated_count"],
        "directional_count": report["directional_count"],
        "neutral_count": report["neutral_count"],
        "brier_score": probability["brier_score"],
        "log_loss": probability["log_loss"],
        "direction_accuracy": probability["direction_accuracy"],
        "balanced_accuracy": probability["balanced_accuracy"],
        "return_mae_bps": report["return"]["mae_bps"],
        "trade_count": decisions["trade_count"],
        "coverage": decisions["coverage"],
        "trade_direction_accuracy": decisions["trade_direction_accuracy"],
        "average_realized_net_return_bps": decisions[
            "average_realized_net_return_bps"
        ],
    }


def validation_slice_diagnostics(
    forecasts: Iterable[ForecastPrediction],
    *,
    policy: DecisionPolicyConfig | None = None,
    neutral_threshold_bps: float = 3.0,
) -> dict[str, Any]:
    """Summarize one horizon of validation predictions by fixed, interpretable bins.

    Uses only predictions and outcomes supplied by the caller. It never opens
    the test split and never estimates a new trading threshold. The time-of-day
    buckets are in America/New_York (DST-aware), not UTC. Small slices must not
    be interpreted as evidence of an edge.
    """

    policy = policy or DecisionPolicyConfig()
    rows = [row for row in forecasts if row.actual_return is not None]
    if not rows:
        raise ValueError("validation diagnostics require resolved forecasts")

    # Also validates mixed horizons and negative neutral thresholds.
    overall = _slice_summary(
        rows, policy=policy, neutral_threshold_bps=neutral_threshold_bps
    )
    groups: dict[str, dict[str, list[ForecastPrediction]]] = {
        name: defaultdict(list)
        for name in (
            "by_symbol",
            "by_market_date",
            "by_ny_half_hour",
            "by_p_up_bin",
            "by_confidence_bin",
        )
    }

    for row in rows:
        local = row.timestamp.astimezone(NEW_YORK)
        half_hour = (local.hour * 60 + local.minute) // 30 * 30
        confidence = max(row.p_up, row.p_down)
        keys = {
            "by_symbol": row.symbol,
            "by_market_date": local.date().isoformat(),
            "by_ny_half_hour": f"{half_hour // 60:02d}:{half_hour % 60:02d}",
            "by_p_up_bin": _tenth_bin(row.p_up),
            "by_confidence_bin": _confidence_bin(confidence),
        }
        for name, key in keys.items():
            groups[name][key].append(row)

    result: dict[str, Any] = {
        "schema_version": 1,
        "purpose": "validation_diagnostics_only",
        "horizon_minutes": rows[0].horizon_minutes,
        "timezone": "America/New_York",
        "neutral_threshold_bps": neutral_threshold_bps,
        "notes": [
            "Slices are descriptive and not independent trials.",
            "Overlapping intraday forecast horizons induce correlated outcomes.",
            "Small slices and selected high-confidence bins are not proof of edge.",
            "No test observations are loaded by this diagnostic function.",
        ],
        "overall": overall,
    }
    for name, buckets in groups.items():
        result[name] = {
            key: _slice_summary(
                buckets[key],
                policy=policy,
                neutral_threshold_bps=neutral_threshold_bps,
            )
            for key in sorted(buckets)
        }
    return result
