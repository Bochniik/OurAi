"""Leakage-safe tabular dataset construction for OurAI trading models.

The first trading data layer is deliberately provider-agnostic and dependency-light.
It accepts timestamped OHLCV bars, creates features using only information available
at each feature timestamp, attaches exact future-return targets, and splits by whole
market dates so one trading day cannot leak across train/validation/test.

Point-in-time joins for news, liquidity, market regime and market-integrity context
are intentionally a later layer. Those sources need their own event/as-of timestamp
rules instead of being silently forward-filled here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, time, timedelta, timezone
from math import isfinite, sqrt
from statistics import fmean
from typing import Any, Iterable, Mapping, Optional, Sequence
from zoneinfo import ZoneInfo


NEW_YORK = ZoneInfo("America/New_York")
DEFAULT_HORIZONS_MINUTES = (5, 15, 30, 60)
RETURN_LOOKBACKS_MINUTES = (1, 5, 15, 30)


@dataclass(frozen=True)
class MarketBar:
    """One timestamped market bar.

    Timestamps must be timezone-aware. Values are intentionally raw market fields;
    adjusted/normalized provider policy belongs to the ingestion layer.
    """

    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    bid: Optional[float] = None
    ask: Optional[float] = None
    session: Optional[str] = None

    def __post_init__(self) -> None:
        symbol = str(self.symbol or "").strip().upper()
        if not symbol:
            raise ValueError("symbol is required")
        object.__setattr__(self, "symbol", symbol)

        if self.timestamp.tzinfo is None or self.timestamp.utcoffset() is None:
            raise ValueError("timestamp must be timezone-aware")
        object.__setattr__(self, "timestamp", self.timestamp.astimezone(timezone.utc))

        for name in ("open", "high", "low", "close"):
            value = float(getattr(self, name))
            if not isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be a positive finite number")
            object.__setattr__(self, name, value)

        volume = float(self.volume)
        if not isfinite(volume) or volume < 0:
            raise ValueError("volume must be a non-negative finite number")
        object.__setattr__(self, "volume", volume)

        if self.high < max(self.open, self.close, self.low):
            raise ValueError("high is inconsistent with OHLC values")
        if self.low > min(self.open, self.close, self.high):
            raise ValueError("low is inconsistent with OHLC values")

        for name in ("bid", "ask"):
            value = getattr(self, name)
            if value is None:
                continue
            numeric = float(value)
            if not isfinite(numeric) or numeric <= 0:
                raise ValueError(f"{name} must be positive and finite when supplied")
            object.__setattr__(self, name, numeric)

        if self.bid is not None and self.ask is not None and self.ask < self.bid:
            raise ValueError("ask cannot be below bid")

        if self.session:
            object.__setattr__(self, "session", str(self.session).strip().lower())

    @classmethod
    def from_mapping(cls, row: Mapping[str, Any]) -> "MarketBar":
        timestamp = row.get("timestamp")
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        if not isinstance(timestamp, datetime):
            raise ValueError("timestamp must be a datetime or ISO-8601 string")
        return cls(
            symbol=row.get("symbol", ""),
            timestamp=timestamp,
            open=row.get("open"),
            high=row.get("high"),
            low=row.get("low"),
            close=row.get("close"),
            volume=row.get("volume"),
            bid=row.get("bid"),
            ask=row.get("ask"),
            session=row.get("session"),
        )


@dataclass(frozen=True)
class TradingSample:
    symbol: str
    timestamp: datetime
    session_date: str
    session: str
    features: Mapping[str, Optional[float]]
    targets: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp.isoformat(),
            "session_date": self.session_date,
            "session": self.session,
            "features": dict(self.features),
            "targets": {key: dict(value) for key, value in self.targets.items()},
        }


def infer_session(timestamp: datetime) -> str:
    local = timestamp.astimezone(NEW_YORK)
    current = local.time().replace(tzinfo=None)
    if time(4, 0) <= current < time(9, 30):
        return "premarket"
    if time(9, 30) <= current < time(16, 0):
        return "regular"
    if time(16, 0) <= current < time(20, 0):
        return "afterhours"
    return "offhours"


def _session_progress(timestamp: datetime, session: str) -> Optional[float]:
    local = timestamp.astimezone(NEW_YORK)
    minute = local.hour * 60 + local.minute + local.second / 60.0
    windows = {
        "premarket": (4 * 60, 9 * 60 + 30),
        "regular": (9 * 60 + 30, 16 * 60),
        "afterhours": (16 * 60, 20 * 60),
    }
    bounds = windows.get(session)
    if bounds is None:
        return None
    start, end = bounds
    return max(0.0, min(1.0, (minute - start) / (end - start)))


def _spread_bps(bar: MarketBar) -> Optional[float]:
    if bar.bid is None or bar.ask is None:
        return None
    midpoint = (bar.bid + bar.ask) / 2.0
    if midpoint <= 0:
        return None
    return (bar.ask - bar.bid) / midpoint * 10_000.0


def _return(current: float, previous: float) -> Optional[float]:
    if previous <= 0:
        return None
    value = current / previous - 1.0
    return value if isfinite(value) else None


def _same_session(a: MarketBar, b: MarketBar) -> bool:
    a_session = a.session or infer_session(a.timestamp)
    b_session = b.session or infer_session(b.timestamp)
    a_date = a.timestamp.astimezone(NEW_YORK).date()
    b_date = b.timestamp.astimezone(NEW_YORK).date()
    return a_date == b_date and a_session == b_session


def _build_features(
    bar: MarketBar,
    *,
    by_timestamp: Mapping[datetime, MarketBar],
    session_bars_to_now: Sequence[MarketBar],
    availability_timestamp: datetime,
) -> dict[str, Optional[float]]:
    features: dict[str, Optional[float]] = {
        "price": bar.close,
        "volume": bar.volume,
    }

    for minutes in RETURN_LOOKBACKS_MINUTES:
        previous = by_timestamp.get(bar.timestamp - timedelta(minutes=minutes))
        value = None
        if previous is not None and _same_session(previous, bar):
            value = _return(bar.close, previous.close)
        features[f"return_{minutes}m"] = value

    # Session VWAP uses only bars whose values are available by this sample.
    total_volume = sum(item.volume for item in session_bars_to_now)
    if total_volume > 0:
        total_value = sum(
            ((item.high + item.low + item.close) / 3.0) * item.volume
            for item in session_bars_to_now
        )
        vwap = total_value / total_volume
        features["distance_from_vwap"] = _return(bar.close, vwap)
    else:
        features["distance_from_vwap"] = None

    prior = list(session_bars_to_now[:-1])[-20:]
    if prior:
        average_volume = fmean(item.volume for item in prior)
        features["relative_volume_20"] = (
            bar.volume / average_volume if average_volume > 0 else None
        )
    else:
        features["relative_volume_20"] = None

    range_window = list(session_bars_to_now)[-20:]
    if range_window:
        high = max(item.high for item in range_window)
        low = min(item.low for item in range_window)
        width = high - low
        features["range_position_20"] = (
            (bar.close - low) / width if width > 0 else 0.5
        )
    else:
        features["range_position_20"] = None

    # Realized volatility is calculated from already-observed contiguous 1m closes.
    returns: list[float] = []
    recent = list(session_bars_to_now)[-21:]
    for previous, current in zip(recent, recent[1:]):
        if current.timestamp - previous.timestamp != timedelta(minutes=1):
            continue
        value = _return(current.close, previous.close)
        if value is not None:
            returns.append(value)
    if len(returns) >= 4:
        mean = fmean(returns)
        variance = fmean((value - mean) ** 2 for value in returns)
        features["realized_volatility_20"] = sqrt(variance)
    else:
        features["realized_volatility_20"] = None

    session = bar.session or infer_session(bar.timestamp)
    features["spread_bps"] = _spread_bps(bar)
    features["session_progress"] = _session_progress(availability_timestamp, session)
    local = availability_timestamp.astimezone(NEW_YORK)
    features["minute_of_day_scaled"] = (local.hour * 60 + local.minute) / 1440.0

    missing = sum(value is None for value in features.values())
    features["feature_missing_fraction"] = missing / max(1, len(features))
    return features


def build_causal_samples(
    bars: Iterable[MarketBar | Mapping[str, Any]],
    *,
    horizons_minutes: Sequence[int] = DEFAULT_HORIZONS_MINUTES,
    minimum_absolute_return_bps: float = 3.0,
    spread_multiplier: float = 1.5,
    allow_session_transition: bool = False,
    bar_availability_delay_minutes: float = 0.0,
) -> list[TradingSample]:
    """Create feature/target rows without using future data in features.

    ``MarketBar.timestamp`` remains the provider/bar timestamp used to locate exact
    lookbacks and future bars. ``bar_availability_delay_minutes`` moves the emitted
    sample timestamp to the time when the current bar is actually observable. The
    same delay is applied to target ``future_timestamp`` values, preserving the
    requested close-to-close horizon while making execution timing explicit.

    Future targets require an exact timestamp match. Missing bars therefore do not
    turn a 7-minute gap into a nominal 5-minute target. By default targets also stay
    inside one named market session; callers may explicitly allow session transitions
    later if a cross-open/cross-close research design requires them.
    """

    horizons = tuple(sorted({int(value) for value in horizons_minutes}))
    if not horizons or any(value <= 0 for value in horizons):
        raise ValueError("horizons_minutes must contain positive integers")
    if minimum_absolute_return_bps < 0 or spread_multiplier < 0:
        raise ValueError("threshold configuration must be non-negative")
    try:
        delay_minutes = float(bar_availability_delay_minutes)
    except (TypeError, ValueError) as exc:
        raise ValueError("bar_availability_delay_minutes must be finite and non-negative") from exc
    if not isfinite(delay_minutes) or delay_minutes < 0:
        raise ValueError("bar_availability_delay_minutes must be finite and non-negative")
    availability_delay = timedelta(minutes=delay_minutes)

    normalized = [
        item if isinstance(item, MarketBar) else MarketBar.from_mapping(item)
        for item in bars
    ]
    normalized.sort(key=lambda item: (item.symbol, item.timestamp))

    symbols: dict[str, list[MarketBar]] = {}
    for bar in normalized:
        symbols.setdefault(bar.symbol, []).append(bar)

    samples: list[TradingSample] = []
    for symbol, symbol_bars in symbols.items():
        by_timestamp = {bar.timestamp: bar for bar in symbol_bars}
        if len(by_timestamp) != len(symbol_bars):
            raise ValueError(f"duplicate timestamp for {symbol}")

        session_history: dict[tuple[str, str], list[MarketBar]] = {}
        for bar in symbol_bars:
            session = bar.session or infer_session(bar.timestamp)
            session_date = bar.timestamp.astimezone(NEW_YORK).date().isoformat()
            availability_timestamp = bar.timestamp + availability_delay
            key = (session_date, session)
            history = session_history.setdefault(key, [])
            history.append(bar)

            features = _build_features(
                bar,
                by_timestamp=by_timestamp,
                session_bars_to_now=history,
                availability_timestamp=availability_timestamp,
            )

            spread = _spread_bps(bar)
            threshold_bps = max(
                float(minimum_absolute_return_bps),
                (spread or 0.0) * float(spread_multiplier),
            )
            threshold_return = threshold_bps / 10_000.0

            targets: dict[str, Mapping[str, Any]] = {}
            for horizon in horizons:
                future = by_timestamp.get(bar.timestamp + timedelta(minutes=horizon))
                if future is None:
                    continue
                if not allow_session_transition and not _same_session(bar, future):
                    continue
                future_return = _return(future.close, bar.close)
                if future_return is None:
                    continue
                direction = (
                    "UP"
                    if future_return > threshold_return
                    else "DOWN"
                    if future_return < -threshold_return
                    else "NEUTRAL"
                )
                targets[f"{horizon}m"] = {
                    "future_timestamp": (future.timestamp + availability_delay).isoformat(),
                    "future_return": future_return,
                    "future_return_pct": future_return * 100.0,
                    "direction": direction,
                    "neutral_threshold_bps": threshold_bps,
                }

            if targets:
                samples.append(
                    TradingSample(
                        symbol=symbol,
                        timestamp=availability_timestamp,
                        session_date=session_date,
                        session=session,
                        features=features,
                        targets=targets,
                    )
                )

    return samples


def chronological_session_split(
    samples: Sequence[TradingSample],
    *,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
) -> dict[str, list[TradingSample]]:
    """Split by complete market dates in chronological order.

    A date is global across all symbols and sessions. This prevents, for example,
    AAPL on one day from entering train while NVDA from the same market day enters
    validation.
    """

    if not (0 < train_fraction < 1):
        raise ValueError("train_fraction must be between 0 and 1")
    if not (0 < validation_fraction < 1):
        raise ValueError("validation_fraction must be between 0 and 1")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("train + validation fractions must leave room for test")

    dates = sorted({sample.session_date for sample in samples})
    if len(dates) < 3:
        raise ValueError("at least three distinct market dates are required")

    train_end = max(1, int(len(dates) * train_fraction))
    validation_end = max(train_end + 1, int(len(dates) * (train_fraction + validation_fraction)))
    validation_end = min(validation_end, len(dates) - 1)

    train_dates = set(dates[:train_end])
    validation_dates = set(dates[train_end:validation_end])
    test_dates = set(dates[validation_end:])

    return {
        "train": [sample for sample in samples if sample.session_date in train_dates],
        "validation": [sample for sample in samples if sample.session_date in validation_dates],
        "test": [sample for sample in samples if sample.session_date in test_dates],
    }


def dataset_manifest(
    samples: Sequence[TradingSample],
    splits: Optional[Mapping[str, Sequence[TradingSample]]] = None,
) -> dict[str, Any]:
    horizons: dict[str, int] = {}
    symbols: dict[str, int] = {}
    sessions: dict[str, int] = {}

    for sample in samples:
        symbols[sample.symbol] = symbols.get(sample.symbol, 0) + 1
        sessions[sample.session] = sessions.get(sample.session, 0) + 1
        for horizon in sample.targets:
            horizons[horizon] = horizons.get(horizon, 0) + 1

    output: dict[str, Any] = {
        "sample_count": len(samples),
        "market_date_count": len({sample.session_date for sample in samples}),
        "symbols": dict(sorted(symbols.items())),
        "sessions": dict(sorted(sessions.items())),
        "resolved_targets": dict(sorted(horizons.items())),
        "feature_policy": "point_in_time_only",
        "target_policy": "exact_timestamp_future_return",
    }
    if splits is not None:
        output["splits"] = {
            name: {
                "sample_count": len(rows),
                "market_dates": sorted({sample.session_date for sample in rows}),
            }
            for name, rows in splits.items()
        }
    return output
