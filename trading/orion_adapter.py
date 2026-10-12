"""Provider-neutral bridge from OrionTrader market rows to OurAI market bars.

The adapter intentionally depends only on the canonical ``MarketBar`` contract. It
accepts common Orion/Yahoo/MT5 row shapes without importing OrionTrader itself, so
OurAI remains reproducible and can be trained from exported rows or DataFrame records.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from math import isfinite
from typing import Any, Iterable, Mapping, Optional, Sequence
from zoneinfo import ZoneInfo

from .dataset import MarketBar


DEFAULT_TIMESTAMP_KEYS = ("timestamp", "time", "datetime")
DEFAULT_SYMBOL_KEYS = ("symbol", "ticker")
DEFAULT_VOLUME_KEYS = ("volume", "tick_volume", "real_volume")
DEFAULT_BID_KEYS = ("bid", "best_bid")
DEFAULT_ASK_KEYS = ("ask", "best_ask")
DEFAULT_SESSION_KEYS = ("session", "market_session")


@dataclass(frozen=True)
class OrionAdapterConfig:
    """Configuration for normalizing exported OrionTrader market rows.

    ``source_timezone`` is required only when the source emits naive datetimes.
    Numeric timestamps are interpreted as Unix seconds (or milliseconds when large
    enough) and are therefore UTC by definition.
    """

    source_name: str = "oriontrader"
    default_symbol: Optional[str] = None
    source_timezone: Optional[str] = None
    strict_duplicates: bool = True
    timestamp_keys: Sequence[str] = DEFAULT_TIMESTAMP_KEYS
    symbol_keys: Sequence[str] = DEFAULT_SYMBOL_KEYS
    volume_keys: Sequence[str] = DEFAULT_VOLUME_KEYS
    bid_keys: Sequence[str] = DEFAULT_BID_KEYS
    ask_keys: Sequence[str] = DEFAULT_ASK_KEYS
    session_keys: Sequence[str] = DEFAULT_SESSION_KEYS

    def timezone(self) -> Optional[ZoneInfo]:
        return ZoneInfo(self.source_timezone) if self.source_timezone else None


@dataclass(frozen=True)
class OrionAdapterResult:
    bars: tuple[MarketBar, ...]
    manifest: Mapping[str, Any]


def _mapping_view(row: Any) -> Optional[Mapping[str, Any]]:
    if isinstance(row, Mapping):
        return row
    candidate = getattr(row, "_mapping", None)
    if isinstance(candidate, Mapping):
        return candidate
    return None


def _lookup(row: Any, keys: Sequence[str]) -> tuple[Optional[str], Any]:
    mapping = _mapping_view(row)
    if mapping is not None:
        lower_to_actual = {str(key).lower(): key for key in mapping.keys()}
        for key in keys:
            actual = lower_to_actual.get(str(key).lower())
            if actual is not None:
                value = mapping[actual]
                if value is not None and str(value).strip() != "":
                    return str(actual), value
        return None, None

    for key in keys:
        if hasattr(row, key):
            value = getattr(row, key)
            if value is not None and str(value).strip() != "":
                return key, value
    return None, None


def _required(row: Any, keys: Sequence[str], field_name: str) -> tuple[str, Any]:
    key, value = _lookup(row, keys)
    if key is None:
        raise ValueError(
            f"missing {field_name}; expected one of: {', '.join(str(item) for item in keys)}"
        )
    return key, value


def _optional_float(row: Any, keys: Sequence[str]) -> tuple[Optional[str], Optional[float]]:
    key, value = _lookup(row, keys)
    if key is None:
        return None, None
    numeric = float(value)
    if not isfinite(numeric):
        raise ValueError(f"{key} must be finite when supplied")
    return key, numeric


def _parse_timestamp(value: Any, *, source_timezone: Optional[ZoneInfo]) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        numeric = float(value)
        if not isfinite(numeric):
            raise ValueError("numeric timestamp must be finite")
        # MT5 commonly exports Unix seconds; some tabular exports use milliseconds.
        if abs(numeric) >= 100_000_000_000:
            numeric /= 1000.0
        parsed = datetime.fromtimestamp(numeric, tz=timezone.utc)
    else:
        text = str(value or "").strip()
        if not text:
            raise ValueError("timestamp is required")
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))

    if parsed.tzinfo is None or parsed.utcoffset() is None:
        if source_timezone is None:
            raise ValueError(
                "naive Orion timestamp is forbidden unless source_timezone is explicitly configured"
            )
        parsed = parsed.replace(tzinfo=source_timezone)
    return parsed


def orion_row_to_market_bar(
    row: Any,
    *,
    config: OrionAdapterConfig | None = None,
) -> tuple[MarketBar, dict[str, Optional[str]]]:
    """Normalize one Orion/Yahoo/MT5-style row into a canonical ``MarketBar``.

    The second return value records which source aliases were consumed. This is used
    by the adapter manifest so experiments can distinguish real volume from MT5 tick
    volume and can reproduce the exact source contract.
    """

    config = config or OrionAdapterConfig()
    timezone_hint = config.timezone()

    timestamp_key, timestamp_value = _required(row, config.timestamp_keys, "timestamp")
    open_key, open_value = _required(row, ("open",), "open")
    high_key, high_value = _required(row, ("high",), "high")
    low_key, low_value = _required(row, ("low",), "low")
    close_key, close_value = _required(row, ("close",), "close")
    volume_key, volume_value = _required(row, config.volume_keys, "volume")

    symbol_key, symbol_value = _lookup(row, config.symbol_keys)
    symbol = str(symbol_value or config.default_symbol or "").strip()
    if not symbol:
        raise ValueError(
            "missing symbol; provide a symbol/ticker field or OrionAdapterConfig.default_symbol"
        )

    bid_key, bid = _optional_float(row, config.bid_keys)
    ask_key, ask = _optional_float(row, config.ask_keys)
    session_key, session_value = _lookup(row, config.session_keys)
    session = str(session_value).strip() if session_value is not None else None

    bar = MarketBar(
        symbol=symbol,
        timestamp=_parse_timestamp(timestamp_value, source_timezone=timezone_hint),
        open=float(open_value),
        high=float(high_value),
        low=float(low_value),
        close=float(close_value),
        volume=float(volume_value),
        bid=bid,
        ask=ask,
        session=session,
    )
    aliases = {
        "timestamp": timestamp_key,
        "symbol": symbol_key or "<default_symbol>",
        "open": open_key,
        "high": high_key,
        "low": low_key,
        "close": close_key,
        "volume": volume_key,
        "bid": bid_key,
        "ask": ask_key,
        "session": session_key,
    }
    return bar, aliases


def normalize_orion_rows(
    rows: Iterable[Any],
    *,
    config: OrionAdapterConfig | None = None,
) -> OrionAdapterResult:
    """Convert Orion-exported rows into sorted, duplicate-checked market bars."""

    config = config or OrionAdapterConfig()
    bars: list[MarketBar] = []
    seen: set[tuple[str, datetime]] = set()
    aliases_used: dict[str, set[str]] = {}
    input_row_count = 0
    duplicates_skipped = 0

    for row_number, row in enumerate(rows, start=1):
        input_row_count += 1
        try:
            bar, aliases = orion_row_to_market_bar(row, config=config)
        except Exception as exc:
            raise ValueError(f"invalid Orion market row {row_number}: {exc}") from exc

        key = (bar.symbol, bar.timestamp)
        if key in seen:
            if config.strict_duplicates:
                raise ValueError(
                    f"duplicate symbol/timestamp at Orion row {row_number}: "
                    f"{bar.symbol} {bar.timestamp.isoformat()}"
                )
            duplicates_skipped += 1
            continue

        seen.add(key)
        bars.append(bar)
        for canonical, alias in aliases.items():
            if alias:
                aliases_used.setdefault(canonical, set()).add(alias)

    if not bars:
        raise ValueError("Orion row collection contains no valid market bars")

    bars.sort(key=lambda item: (item.symbol, item.timestamp))
    manifest = {
        "schema_version": 1,
        "adapter": "orion_market_rows",
        "source_name": str(config.source_name or "oriontrader"),
        "source_timezone": config.source_timezone,
        "strict_duplicates": config.strict_duplicates,
        "input_row_count": input_row_count,
        "row_count": len(bars),
        "duplicates_skipped": duplicates_skipped,
        "symbols": sorted({bar.symbol for bar in bars}),
        "first_timestamp": min(bar.timestamp for bar in bars).isoformat(),
        "last_timestamp": max(bar.timestamp for bar in bars).isoformat(),
        "aliases_used": {
            key: sorted(values)
            for key, values in sorted(aliases_used.items())
        },
        "volume_semantics": (
            "tick_volume"
            if aliases_used.get("volume") == {"tick_volume"}
            else "source_volume"
        ),
    }
    return OrionAdapterResult(bars=tuple(bars), manifest=manifest)
