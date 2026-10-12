"""Forecast evaluation, calibration diagnostics, and abstaining trade decisions.

This module deliberately separates a model's forecast from the policy that decides
whether a forecast is tradable after estimated costs. It provides dependency-light
metrics that can be reused by simple baselines and later LightGBM/neural models.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from math import isfinite, log
from statistics import fmean
from typing import Any, Iterable, Optional


LONG = "LONG"
SHORT = "SHORT"
NO_TRADE = "NO_TRADE"


@dataclass(frozen=True)
class ForecastPrediction:
    symbol: str
    timestamp: datetime
    horizon_minutes: int
    p_up: float
    expected_return: float
    actual_return: Optional[float] = None
    spread_bps: Optional[float] = None

    def __post_init__(self) -> None:
        symbol = str(self.symbol or "").strip().upper()
        if not symbol:
            raise ValueError("symbol is required")
        object.__setattr__(self, "symbol", symbol)

        if self.timestamp.tzinfo is None or self.timestamp.utcoffset() is None:
            raise ValueError("timestamp must be timezone-aware")
        object.__setattr__(self, "timestamp", self.timestamp.astimezone(timezone.utc))

        horizon = int(self.horizon_minutes)
        if horizon <= 0:
            raise ValueError("horizon_minutes must be positive")
        object.__setattr__(self, "horizon_minutes", horizon)

        p_up = float(self.p_up)
        if not isfinite(p_up) or not 0.0 <= p_up <= 1.0:
            raise ValueError("p_up must be a finite probability in [0, 1]")
        object.__setattr__(self, "p_up", p_up)

        expected_return = float(self.expected_return)
        if not isfinite(expected_return):
            raise ValueError("expected_return must be finite")
        object.__setattr__(self, "expected_return", expected_return)

        if self.actual_return is not None:
            actual_return = float(self.actual_return)
            if not isfinite(actual_return):
                raise ValueError("actual_return must be finite when supplied")
            object.__setattr__(self, "actual_return", actual_return)

        if self.spread_bps is not None:
            spread_bps = float(self.spread_bps)
            if not isfinite(spread_bps) or spread_bps < 0:
                raise ValueError("spread_bps must be finite and non-negative when supplied")
            object.__setattr__(self, "spread_bps", spread_bps)

    @property
    def p_down(self) -> float:
        return 1.0 - self.p_up

    @property
    def expected_return_pct(self) -> float:
        return self.expected_return * 100.0


@dataclass(frozen=True)
class DecisionPolicyConfig:
    min_direction_probability: float = 0.60
    minimum_edge_bps: float = 2.0
    slippage_bps: float = 1.5
    fees_bps: float = 0.5
    fallback_spread_bps: float = 5.0

    def __post_init__(self) -> None:
        probability = float(self.min_direction_probability)
        if not 0.5 < probability < 1.0:
            raise ValueError("min_direction_probability must be between 0.5 and 1")
        object.__setattr__(self, "min_direction_probability", probability)

        for name in (
            "minimum_edge_bps",
            "slippage_bps",
            "fees_bps",
            "fallback_spread_bps",
        ):
            value = float(getattr(self, name))
            if not isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
            object.__setattr__(self, name, value)


@dataclass(frozen=True)
class TradeDecision:
    action: str
    p_up: float
    p_down: float
    confidence: float
    expected_return: float
    expected_return_pct: float
    estimated_cost_bps: float
    expected_net_edge_bps: float
    reason: str


def _estimated_cost_bps(
    forecast: ForecastPrediction,
    config: DecisionPolicyConfig,
) -> float:
    spread = (
        forecast.spread_bps
        if forecast.spread_bps is not None
        else config.fallback_spread_bps
    )
    return spread + config.slippage_bps + config.fees_bps


def decide_forecast(
    forecast: ForecastPrediction,
    *,
    config: DecisionPolicyConfig | None = None,
) -> TradeDecision:
    """Convert one probabilistic forecast into LONG/SHORT/NO_TRADE.

    The probability direction and expected-return sign must agree. The expected move
    must also clear estimated spread/slippage/fees plus a configurable safety edge.
    """

    config = config or DecisionPolicyConfig()
    cost_bps = _estimated_cost_bps(forecast, config)
    expected_bps = forecast.expected_return * 10_000.0
    required_bps = cost_bps + config.minimum_edge_bps
    confidence = max(forecast.p_up, forecast.p_down)

    if (
        forecast.p_up >= config.min_direction_probability
        and expected_bps >= required_bps
    ):
        action = LONG
        net_edge_bps = expected_bps - cost_bps
        reason = "up_probability_and_expected_edge_clear_thresholds"
    elif (
        forecast.p_down >= config.min_direction_probability
        and expected_bps <= -required_bps
    ):
        action = SHORT
        net_edge_bps = -expected_bps - cost_bps
        reason = "down_probability_and_expected_edge_clear_thresholds"
    else:
        action = NO_TRADE
        dominant_expected_bps = expected_bps if forecast.p_up >= forecast.p_down else -expected_bps
        net_edge_bps = dominant_expected_bps - cost_bps
        if confidence < config.min_direction_probability:
            reason = "direction_probability_below_threshold"
        elif forecast.p_up >= forecast.p_down and expected_bps <= 0:
            reason = "probability_return_sign_disagreement"
        elif forecast.p_down > forecast.p_up and expected_bps >= 0:
            reason = "probability_return_sign_disagreement"
        else:
            reason = "expected_edge_does_not_clear_costs"

    return TradeDecision(
        action=action,
        p_up=forecast.p_up,
        p_down=forecast.p_down,
        confidence=confidence,
        expected_return=forecast.expected_return,
        expected_return_pct=forecast.expected_return_pct,
        estimated_cost_bps=cost_bps,
        expected_net_edge_bps=net_edge_bps,
        reason=reason,
    )


def _actual_direction(actual_return: float, neutral_threshold_bps: float) -> Optional[int]:
    actual_bps = actual_return * 10_000.0
    if actual_bps > neutral_threshold_bps:
        return 1
    if actual_bps < -neutral_threshold_bps:
        return 0
    return None


def calibration_table(
    forecasts: Iterable[ForecastPrediction],
    *,
    bins: int = 10,
    neutral_threshold_bps: float = 0.0,
) -> list[dict[str, Any]]:
    """Build an equal-width reliability table for P(UP).

    Neutral outcomes are excluded because the direction head is interpreted as a
    conditional UP-vs-DOWN probability. Expected return and the decision policy are
    responsible for abstaining on weak/neutral opportunities.
    """

    if bins <= 0:
        raise ValueError("bins must be positive")
    if neutral_threshold_bps < 0:
        raise ValueError("neutral_threshold_bps must be non-negative")

    buckets: list[list[tuple[float, int]]] = [[] for _ in range(bins)]
    for forecast in forecasts:
        if forecast.actual_return is None:
            continue
        direction = _actual_direction(forecast.actual_return, neutral_threshold_bps)
        if direction is None:
            continue
        index = min(bins - 1, int(forecast.p_up * bins))
        buckets[index].append((forecast.p_up, direction))

    table: list[dict[str, Any]] = []
    for index, rows in enumerate(buckets):
        lower = index / bins
        upper = (index + 1) / bins
        if rows:
            mean_probability = fmean(probability for probability, _ in rows)
            actual_up_rate = fmean(outcome for _, outcome in rows)
        else:
            mean_probability = None
            actual_up_rate = None
        table.append(
            {
                "lower": lower,
                "upper": upper,
                "count": len(rows),
                "mean_p_up": mean_probability,
                "actual_up_rate": actual_up_rate,
            }
        )
    return table


def evaluate_forecasts(
    forecasts: Iterable[ForecastPrediction],
    *,
    policy: DecisionPolicyConfig | None = None,
    neutral_threshold_bps: float = 0.0,
    calibration_bins: int = 10,
) -> dict[str, Any]:
    """Evaluate one horizon of held-out forecasts.

    This function is intended for validation/test/forward data only. It reports
    probability quality, return error, abstention coverage, selective direction
    accuracy, and simple cost-adjusted realized returns for the decision policy.
    """

    policy = policy or DecisionPolicyConfig()
    rows = list(forecasts)
    if not rows:
        raise ValueError("at least one forecast is required")
    if neutral_threshold_bps < 0:
        raise ValueError("neutral_threshold_bps must be non-negative")

    horizons = {row.horizon_minutes for row in rows}
    if len(horizons) != 1:
        raise ValueError("evaluate_forecasts requires exactly one prediction horizon")

    evaluated = [row for row in rows if row.actual_return is not None]
    if not evaluated:
        raise ValueError("at least one forecast with actual_return is required")

    directional: list[tuple[ForecastPrediction, int]] = []
    for row in evaluated:
        direction = _actual_direction(row.actual_return or 0.0, neutral_threshold_bps)
        if direction is not None:
            directional.append((row, direction))

    epsilon = 1e-12
    if directional:
        brier = fmean((row.p_up - outcome) ** 2 for row, outcome in directional)
        log_loss = fmean(
            -(
                outcome * log(min(1.0 - epsilon, max(epsilon, row.p_up)))
                + (1 - outcome)
                * log(min(1.0 - epsilon, max(epsilon, row.p_down)))
            )
            for row, outcome in directional
        )
        accuracy = fmean(
            int((row.p_up >= 0.5) == bool(outcome))
            for row, outcome in directional
        )

        positives = [(row, outcome) for row, outcome in directional if outcome == 1]
        negatives = [(row, outcome) for row, outcome in directional if outcome == 0]
        if positives and negatives:
            true_positive_rate = fmean(int(row.p_up >= 0.5) for row, _ in positives)
            true_negative_rate = fmean(int(row.p_up < 0.5) for row, _ in negatives)
            balanced_accuracy = (true_positive_rate + true_negative_rate) / 2.0
        else:
            balanced_accuracy = None
    else:
        brier = None
        log_loss = None
        accuracy = None
        balanced_accuracy = None

    return_mae_bps = fmean(
        abs(row.expected_return - (row.actual_return or 0.0)) * 10_000.0
        for row in evaluated
    )

    decisions = [(row, decide_forecast(row, config=policy)) for row in evaluated]
    traded = [(row, decision) for row, decision in decisions if decision.action != NO_TRADE]
    realized_net_bps: list[float] = []
    correct_trade_direction: list[int] = []
    long_count = 0
    short_count = 0

    for row, decision in traded:
        actual_bps = (row.actual_return or 0.0) * 10_000.0
        if decision.action == LONG:
            gross_bps = actual_bps
            long_count += 1
        else:
            gross_bps = -actual_bps
            short_count += 1
        realized_net_bps.append(gross_bps - decision.estimated_cost_bps)
        correct_trade_direction.append(int(gross_bps > neutral_threshold_bps))

    trade_count = len(traded)
    coverage = trade_count / len(evaluated)

    return {
        "schema_version": 1,
        "horizon_minutes": next(iter(horizons)),
        "forecast_count": len(rows),
        "evaluated_count": len(evaluated),
        "directional_count": len(directional),
        "neutral_count": len(evaluated) - len(directional),
        "symbols": sorted({row.symbol for row in evaluated}),
        "probability": {
            "brier_score": brier,
            "log_loss": log_loss,
            "direction_accuracy": accuracy,
            "balanced_accuracy": balanced_accuracy,
            "calibration": calibration_table(
                evaluated,
                bins=calibration_bins,
                neutral_threshold_bps=neutral_threshold_bps,
            ),
        },
        "return": {
            "mae_bps": return_mae_bps,
        },
        "decision_policy": {
            "coverage": coverage,
            "trade_count": trade_count,
            "no_trade_count": len(evaluated) - trade_count,
            "long_count": long_count,
            "short_count": short_count,
            "trade_direction_accuracy": (
                fmean(correct_trade_direction) if correct_trade_direction else None
            ),
            "average_realized_net_return_bps": (
                fmean(realized_net_bps) if realized_net_bps else None
            ),
            "positive_net_trade_rate": (
                fmean(int(value > 0) for value in realized_net_bps)
                if realized_net_bps
                else None
            ),
            "config": {
                "min_direction_probability": policy.min_direction_probability,
                "minimum_edge_bps": policy.minimum_edge_bps,
                "slippage_bps": policy.slippage_bps,
                "fees_bps": policy.fees_bps,
                "fallback_spread_bps": policy.fallback_spread_bps,
            },
        },
    }
