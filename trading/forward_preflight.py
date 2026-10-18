"""Side-effect-free timing and data-quality diagnostics for v3 shadow preflight.

A historical probe verifies IEX *access*, not live readiness. An intraday probe
requires fresh bars for every frozen symbol to claim current coverage. Exchange
holidays are not inferred from weekday/date math.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time, timedelta, timezone
from typing import Iterable

from .dataset import NEW_YORK, MarketBar


@dataclass(frozen=True)
class ProbeWindow:
    start: datetime
    end: datetime
    kind: str  # "intraday" or "historical"
    market_date: str


def _aware_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("now_utc must be timezone-aware")
    return value.astimezone(timezone.utc)


def choose_probe_window(now_utc: datetime) -> ProbeWindow:
    """Choose bounded IEX bar query, including weekends and pre-open checks.

    Intraday means *clock within US regular session*, not confirmation of an
    open exchange. Holiday calendars are intentionally not guessed.
    """
    now = _aware_utc(now_utc)
    local = now.astimezone(NEW_YORK)
    today = local.date()
    today_open = datetime.combine(today, time(9, 30), tzinfo=NEW_YORK)
    today_close = datetime.combine(today, time(16, 0), tzinfo=NEW_YORK)

    if local.weekday() < 5 and today_open + timedelta(minutes=3) <= local < today_close:
        start = max(today_open, local - timedelta(minutes=16))
        end = local - timedelta(minutes=1)
        return ProbeWindow(
            start=start.astimezone(timezone.utc),
            end=end.astimezone(timezone.utc),
            kind="intraday",
            market_date=today.isoformat(),
        )

    if local.weekday() < 5 and local >= today_close:
        date = today
    else:
        date = today - timedelta(days=1)
        while date.weekday() >= 5:
            date -= timedelta(days=1)
    close = datetime.combine(date, time(16, 0), tzinfo=NEW_YORK)
    return ProbeWindow(
        start=(close - timedelta(minutes=20)).astimezone(timezone.utc),
        end=close.astimezone(timezone.utc),
        kind="historical",
        market_date=date.isoformat(),
    )


def assess_bar_coverage(
    bars: Iterable[MarketBar],
    *,
    required_symbols: Iterable[str],
    window: ProbeWindow,
    now_utc: datetime,
    freshness_limit_minutes: float = 8.0,
    availability_delay_minutes: float = 1.0,
) -> dict:
    """Summarize source coverage without treating sparse IEX as SIP complete."""
    if freshness_limit_minutes <= 0 or availability_delay_minutes < 0:
        raise ValueError("freshness parameters are invalid")
    now = _aware_utc(now_utc)
    symbols = tuple(dict.fromkeys(str(s).upper() for s in required_symbols))
    if not symbols:
        raise ValueError("required_symbols cannot be empty")
    latest: dict[str, datetime] = {}
    count = 0
    outside_window_count = 0
    required = set(symbols)
    for bar in bars:
        if bar.symbol not in required:
            continue
        if not window.start <= bar.timestamp < window.end:
            outside_window_count += 1
            continue
        count += 1
        if bar.symbol not in latest or latest[bar.symbol] < bar.timestamp:
            latest[bar.symbol] = bar.timestamp

    missing = sorted(required.difference(latest))
    age_minutes: dict[str, float] = {}
    if window.kind == "intraday":
        for symbol, timestamp in latest.items():
            age_minutes[symbol] = round(
                (now - timestamp).total_seconds() / 60 - availability_delay_minutes, 2
            )
    stale = sorted(
        symbol
        for symbol, minutes in age_minutes.items()
        if minutes > freshness_limit_minutes or minutes < -0.1
    )
    complete = not missing and not stale and count > 0
    live_ready = window.kind == "intraday" and complete
    return {
        "window_type": window.kind,
        "session_date": window.market_date,
        "requested_symbols": list(symbols),
        "observed_symbols": sorted(latest),
        "missing_symbols": missing,
        "stale_symbols": stale,
        "latest_bar_left_edge_utc": {
            symbol: timestamp.astimezone(timezone.utc).isoformat()
            for symbol, timestamp in sorted(latest.items())
        },
        "age_since_bar_availability_minutes": age_minutes,
        "bar_count": count,
        "outside_probe_window_count": outside_window_count,
        "historical_access_observed": count > 0,
        "all_symbols_observed": not missing,
        "live_freshness_confirmed": live_ready,
        "assessment": (
            "LIVE_COVERAGE_OK" if live_ready else
            "LIVE_COVERAGE_INCOMPLETE_OR_STALE" if window.kind == "intraday" else
            "HISTORICAL_ACCESS_OK_LIVE_UNTESTED" if count else
            "NO_HISTORICAL_BARS"
        ),
        "limitations": (
            "IEX reflects one exchange, not consolidated US volume. "
            "A historical probe cannot certify upcoming live market access. "
            "Exchange holidays are not verified by this check."
        ),
    }
