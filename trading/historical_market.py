"""Historical intraday market-data acquisition and dataset building for OurAI.

This module deliberately keeps provider credentials out of repository state. Alpaca
credentials are read from explicit arguments or local environment variables, while
all fetched bars are normalized into the provider-neutral ``MarketBar`` contract
before feature/target construction.

The first production contract supports 1-minute Alpaca stock bars only. OurAI's
current causal feature builder uses exact 1-minute lookbacks, so accepting coarser
bars here would silently change feature semantics.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from .dataset import (
    DEFAULT_HORIZONS_MINUTES,
    MarketBar,
    TradingSample,
    build_causal_samples,
    chronological_session_split,
    dataset_manifest,
    infer_session,
)
from .ingestion import sha256_file, write_samples_jsonl


ALPACA_DATA_URL = "https://data.alpaca.markets"
DEFAULT_SYMBOLS = ("NVDA", "AAPL", "TSLA", "JPM", "AMD", "META", "MSFT")
SUPPORTED_FEEDS = {"iex", "sip", "boats", "otc"}
SUPPORTED_ADJUSTMENTS = {"raw", "split", "dividend", "all"}
SUPPORTED_SESSIONS = {"premarket", "regular", "afterhours"}
BUILDER_ID = "ourai-historical-market-v1"

RequestJSON = Callable[[str, Mapping[str, str]], Mapping[str, Any]]


@dataclass(frozen=True)
class AlpacaCredentials:
    key_id: str
    secret_key: str

    def __post_init__(self) -> None:
        if not str(self.key_id or "").strip() or not str(self.secret_key or "").strip():
            raise ValueError("Alpaca key_id and secret_key are required")

    @classmethod
    def from_environment(cls) -> "AlpacaCredentials":
        key_id = os.getenv("ALPACA_API_KEY_ID") or os.getenv("APCA_API_KEY_ID")
        secret = os.getenv("ALPACA_API_SECRET_KEY") or os.getenv("APCA_API_SECRET_KEY")
        if not key_id or not secret:
            raise ValueError(
                "Alpaca credentials are missing. Set ALPACA_API_KEY_ID and "
                "ALPACA_API_SECRET_KEY in the local environment."
            )
        return cls(key_id=key_id, secret_key=secret)

    def headers(self) -> dict[str, str]:
        return {
            "APCA-API-KEY-ID": self.key_id,
            "APCA-API-SECRET-KEY": self.secret_key,
            "Accept": "application/json",
        }


@dataclass(frozen=True)
class HistoricalMarketConfig:
    symbols: tuple[str, ...] = DEFAULT_SYMBOLS
    start: datetime | None = None
    end: datetime | None = None
    timeframe: str = "1Min"
    feed: str = "iex"
    adjustment: str = "raw"
    sessions: tuple[str, ...] = ("regular",)
    batch_size: int = 20
    page_limit: int = 10_000
    max_pages_per_batch: int = 100
    strict_duplicates: bool = True
    allow_partial: bool = False

    def __post_init__(self) -> None:
        symbols = tuple(
            dict.fromkeys(str(symbol).strip().upper() for symbol in self.symbols if str(symbol).strip())
        )
        if not symbols:
            raise ValueError("at least one symbol is required")
        if len(symbols) > 200:
            raise ValueError("at most 200 symbols are allowed per dataset build")
        object.__setattr__(self, "symbols", symbols)

        if self.start is None or self.end is None:
            raise ValueError("timezone-aware start and end are required")
        for name, value in (("start", self.start), ("end", self.end)):
            if value.tzinfo is None or value.utcoffset() is None:
                raise ValueError(f"{name} must be timezone-aware")
            object.__setattr__(self, name, value.astimezone(timezone.utc))
        if self.end <= self.start:
            raise ValueError("end must be after start")

        if self.timeframe != "1Min":
            raise ValueError("historical builder v1 supports timeframe=1Min only")
        if self.feed not in SUPPORTED_FEEDS:
            raise ValueError(f"unsupported Alpaca feed: {self.feed}")
        if self.adjustment not in SUPPORTED_ADJUSTMENTS:
            raise ValueError(f"unsupported Alpaca adjustment: {self.adjustment}")

        sessions = tuple(dict.fromkeys(str(value).strip().lower() for value in self.sessions))
        if not sessions or any(value not in SUPPORTED_SESSIONS for value in sessions):
            raise ValueError(
                "sessions must contain one or more of: premarket, regular, afterhours"
            )
        object.__setattr__(self, "sessions", sessions)

        if not 1 <= int(self.batch_size) <= 50:
            raise ValueError("batch_size must be in [1, 50]")
        if not 1 <= int(self.page_limit) <= 10_000:
            raise ValueError("page_limit must be in [1, 10000]")
        if not 1 <= int(self.max_pages_per_batch) <= 500:
            raise ValueError("max_pages_per_batch must be in [1, 500]")


@dataclass(frozen=True)
class HistoricalFetchResult:
    bars: tuple[MarketBar, ...]
    manifest: Mapping[str, Any]


def _iso_z(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _default_request_json(url: str, headers: Mapping[str, str]) -> Mapping[str, Any]:
    request = Request(url, headers=dict(headers), method="GET")
    with urlopen(request, timeout=30) as response:  # nosec B310 - fixed HTTPS provider URL
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("Alpaca response must be a JSON object")
    return payload


def _alpaca_row_to_bar(symbol: str, row: Mapping[str, Any]) -> MarketBar:
    timestamp = row.get("t") or row.get("timestamp")
    if not timestamp:
        raise ValueError("Alpaca bar is missing timestamp")
    parsed = datetime.fromisoformat(str(timestamp).replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("Alpaca bar timestamp must be timezone-aware")

    def required(short: str, long: str) -> Any:
        value = row.get(short)
        if value is None:
            value = row.get(long)
        if value is None:
            raise ValueError(f"Alpaca bar is missing {long}")
        return value

    return MarketBar(
        symbol=symbol,
        timestamp=parsed,
        open=float(required("o", "open")),
        high=float(required("h", "high")),
        low=float(required("l", "low")),
        close=float(required("c", "close")),
        volume=float(required("v", "volume")),
        session=infer_session(parsed),
    )


def _normalize_response_bars(
    payload: Mapping[str, Any],
    *,
    allowed_sessions: set[str],
) -> list[MarketBar]:
    grouped = payload.get("bars") or {}
    if not isinstance(grouped, Mapping):
        raise ValueError("Alpaca response field 'bars' must be grouped by symbol")

    output: list[MarketBar] = []
    for raw_symbol, rows in grouped.items():
        symbol = str(raw_symbol or "").strip().upper()
        if not symbol or not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
            continue
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            bar = _alpaca_row_to_bar(symbol, row)
            if bar.session in allowed_sessions:
                output.append(bar)
    return output


def fetch_alpaca_historical_bars(
    config: HistoricalMarketConfig,
    *,
    credentials: AlpacaCredentials | None = None,
    request_json: RequestJSON | None = None,
) -> HistoricalFetchResult:
    """Fetch bounded, paginated Alpaca 1-minute bars and normalize them.

    Credentials are never written to the returned manifest. The manifest records the
    exact provider query contract and whether the configured pagination ceiling was
    reached so a partial history cannot be mistaken for a complete training source.
    """

    credentials = credentials or AlpacaCredentials.from_environment()
    request_json = request_json or _default_request_json
    headers = credentials.headers()
    allowed_sessions = set(config.sessions)

    bars: list[MarketBar] = []
    seen: set[tuple[str, datetime]] = set()
    duplicates_skipped = 0
    batch_records: list[dict[str, Any]] = []
    partial = False

    for offset in range(0, len(config.symbols), config.batch_size):
        symbols = config.symbols[offset : offset + config.batch_size]
        page_token: Optional[str] = None
        pages = 0
        fetched_rows = 0

        while True:
            params = {
                "symbols": ",".join(symbols),
                "timeframe": config.timeframe,
                "start": _iso_z(config.start),
                "end": _iso_z(config.end),
                "limit": str(config.page_limit),
                "adjustment": config.adjustment,
                "feed": config.feed,
                "sort": "asc",
            }
            if page_token:
                params["page_token"] = page_token
            url = f"{ALPACA_DATA_URL}/v2/stocks/bars?{urlencode(params)}"
            payload = request_json(url, headers)
            pages += 1

            normalized = _normalize_response_bars(payload, allowed_sessions=allowed_sessions)
            fetched_rows += len(normalized)
            for bar in normalized:
                key = (bar.symbol, bar.timestamp)
                if key in seen:
                    if config.strict_duplicates:
                        raise ValueError(
                            f"duplicate Alpaca symbol/timestamp: {bar.symbol} "
                            f"{bar.timestamp.isoformat()}"
                        )
                    duplicates_skipped += 1
                    continue
                seen.add(key)
                bars.append(bar)

            raw_token = payload.get("next_page_token")
            page_token = str(raw_token) if raw_token else None
            if not page_token:
                break
            if pages >= config.max_pages_per_batch:
                partial = True
                break

        batch_records.append(
            {
                "symbols": list(symbols),
                "pages": pages,
                "accepted_rows": fetched_rows,
                "partial": bool(page_token),
            }
        )

    if partial and not config.allow_partial:
        raise ValueError(
            "Alpaca pagination limit was reached; refusing to build from partial history. "
            "Increase max_pages_per_batch or explicitly set allow_partial=True for research."
        )
    if not bars:
        raise ValueError("Alpaca returned no usable bars for the requested contract")

    bars.sort(key=lambda item: (item.symbol, item.timestamp))
    manifest = {
        "schema_version": 1,
        "provider": "alpaca",
        "endpoint": "/v2/stocks/bars",
        "feed": config.feed,
        "timeframe": config.timeframe,
        "adjustment": config.adjustment,
        "start": _iso_z(config.start),
        "end": _iso_z(config.end),
        "requested_symbols": list(config.symbols),
        "sessions": list(config.sessions),
        "row_count": len(bars),
        "symbols_with_rows": sorted({bar.symbol for bar in bars}),
        "first_timestamp": bars[0].timestamp.isoformat(),
        "last_timestamp": max(bar.timestamp for bar in bars).isoformat(),
        "duplicates_skipped": duplicates_skipped,
        "partial": partial,
        "batches": batch_records,
        "credentials_persisted": False,
        "limitations": [
            (
                "IEX is a single-exchange feed and must not be interpreted as consolidated "
                "US market volume or quote coverage."
                if config.feed == "iex"
                else "Feed completeness depends on the Alpaca market-data entitlement."
            )
        ],
    }
    return HistoricalFetchResult(bars=tuple(bars), manifest=manifest)


def write_canonical_bars_csv(
    bars: Iterable[MarketBar],
    path: str | Path,
) -> dict[str, Any]:
    """Persist a deterministic canonical bar cache without provider credentials."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(bars, key=lambda item: (item.symbol, item.timestamp))
    if not ordered:
        raise ValueError("cannot write an empty canonical bar cache")

    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("symbol", "timestamp", "open", "high", "low", "close", "volume", "session"))
        for bar in ordered:
            writer.writerow(
                (
                    bar.symbol,
                    bar.timestamp.isoformat(),
                    format(bar.open, ".12g"),
                    format(bar.high, ".12g"),
                    format(bar.low, ".12g"),
                    format(bar.close, ".12g"),
                    format(bar.volume, ".12g"),
                    bar.session or infer_session(bar.timestamp),
                )
            )

    return {
        "path": destination.name,
        "row_count": len(ordered),
        "sha256": sha256_file(destination),
    }


def _configuration_fingerprint(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_historical_market_dataset(
    fetch_result: HistoricalFetchResult,
    output_dir: str | Path,
    *,
    horizons_minutes: Sequence[int] = DEFAULT_HORIZONS_MINUTES,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
    minimum_absolute_return_bps: float = 3.0,
    spread_multiplier: float = 1.5,
) -> dict[str, Any]:
    """Create a hashed local source cache plus chronological model-ready splits."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    bars = list(fetch_result.bars)
    if not bars:
        raise ValueError("fetch_result contains no bars")

    source_cache = write_canonical_bars_csv(bars, destination / "source_bars.csv")
    samples = build_causal_samples(
        bars,
        horizons_minutes=tuple(int(value) for value in horizons_minutes),
        minimum_absolute_return_bps=minimum_absolute_return_bps,
        spread_multiplier=spread_multiplier,
    )
    if not samples:
        raise ValueError("no resolved samples could be built from historical bars")

    splits = chronological_session_split(
        samples,
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
    )
    outputs = {
        name: write_samples_jsonl(rows, destination / f"{name}.jsonl")
        for name, rows in splits.items()
    }
    configuration = {
        "horizons_minutes": [int(value) for value in horizons_minutes],
        "train_fraction": train_fraction,
        "validation_fraction": validation_fraction,
        "minimum_absolute_return_bps": minimum_absolute_return_bps,
        "spread_multiplier": spread_multiplier,
    }
    lineage_payload = {
        "builder": BUILDER_ID,
        "source_sha256": source_cache["sha256"],
        "source_contract": dict(fetch_result.manifest),
        "configuration": configuration,
    }
    dataset_id = _configuration_fingerprint(lineage_payload)[:20]
    manifest = {
        "schema_version": 1,
        "builder": BUILDER_ID,
        "dataset_id": dataset_id,
        "source": dict(fetch_result.manifest),
        "source_cache": source_cache,
        "dataset": dataset_manifest(samples, splits),
        "configuration": configuration,
        "outputs": outputs,
    }
    manifest_path = destination / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        **manifest,
        "manifest_sha256": sha256_file(manifest_path),
    }


def fetch_and_build_alpaca_dataset(
    output_dir: str | Path,
    *,
    market_config: HistoricalMarketConfig,
    credentials: AlpacaCredentials | None = None,
    request_json: RequestJSON | None = None,
    horizons_minutes: Sequence[int] = DEFAULT_HORIZONS_MINUTES,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
    minimum_absolute_return_bps: float = 3.0,
    spread_multiplier: float = 1.5,
) -> dict[str, Any]:
    fetched = fetch_alpaca_historical_bars(
        market_config,
        credentials=credentials,
        request_json=request_json,
    )
    return build_historical_market_dataset(
        fetched,
        output_dir,
        horizons_minutes=horizons_minutes,
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
        minimum_absolute_return_bps=minimum_absolute_return_bps,
        spread_multiplier=spread_multiplier,
    )
