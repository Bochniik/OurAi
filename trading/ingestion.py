"""Deterministic provider-neutral ingestion for OurAI trading datasets."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Optional
from zoneinfo import ZoneInfo

from .dataset import (
    DEFAULT_HORIZONS_MINUTES,
    MarketBar,
    TradingSample,
    build_causal_samples,
    chronological_session_split,
    dataset_manifest,
)


REQUIRED_COLUMNS = ("symbol", "timestamp", "open", "high", "low", "close", "volume")
OPTIONAL_COLUMNS = ("bid", "ask", "session")


@dataclass(frozen=True)
class CSVIngestionConfig:
    source_timezone: Optional[str] = None
    strict_duplicates: bool = True

    def timezone(self) -> Optional[ZoneInfo]:
        return ZoneInfo(self.source_timezone) if self.source_timezone else None


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip()
    return None if not text else float(text)


def _parse_timestamp(value: str, *, source_timezone: Optional[ZoneInfo]) -> datetime:
    text = str(value or "").strip()
    if not text:
        raise ValueError("timestamp is required")
    parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        if source_timezone is None:
            raise ValueError(
                "naive timestamps are forbidden unless source_timezone is explicitly configured"
            )
        parsed = parsed.replace(tzinfo=source_timezone)
    return parsed


def load_market_bars_csv(
    path: str | Path,
    *,
    config: CSVIngestionConfig | None = None,
) -> tuple[list[MarketBar], dict[str, Any]]:
    """Load and validate a CSV file into canonical MarketBar objects.

    The returned manifest records the raw file SHA-256 and basic temporal/symbol
    coverage so experiments can prove exactly which source file they consumed.
    """

    config = config or CSVIngestionConfig()
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)

    timezone = config.timezone()
    bars: list[MarketBar] = []
    seen: set[tuple[str, datetime]] = set()

    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = tuple(reader.fieldnames or ())
        missing = [name for name in REQUIRED_COLUMNS if name not in fieldnames]
        if missing:
            raise ValueError(f"CSV is missing required columns: {', '.join(missing)}")

        for line_number, row in enumerate(reader, start=2):
            try:
                bar = MarketBar(
                    symbol=row["symbol"],
                    timestamp=_parse_timestamp(row["timestamp"], source_timezone=timezone),
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                    volume=float(row["volume"]),
                    bid=_optional_float(row.get("bid")),
                    ask=_optional_float(row.get("ask")),
                    session=(row.get("session") or None),
                )
            except Exception as exc:
                raise ValueError(f"invalid market row at CSV line {line_number}: {exc}") from exc

            key = (bar.symbol, bar.timestamp)
            if key in seen:
                if config.strict_duplicates:
                    raise ValueError(
                        f"duplicate symbol/timestamp at CSV line {line_number}: "
                        f"{bar.symbol} {bar.timestamp.isoformat()}"
                    )
                continue
            seen.add(key)
            bars.append(bar)

    if not bars:
        raise ValueError("CSV contains no valid market bars")

    bars.sort(key=lambda item: (item.symbol, item.timestamp))
    manifest = {
        "schema_version": 1,
        "source_path": source.name,
        "source_sha256": sha256_file(source),
        "row_count": len(bars),
        "symbols": sorted({bar.symbol for bar in bars}),
        "first_timestamp": min(bar.timestamp for bar in bars).isoformat(),
        "last_timestamp": max(bar.timestamp for bar in bars).isoformat(),
        "source_timezone": config.source_timezone,
        "strict_duplicates": config.strict_duplicates,
        "required_columns": list(REQUIRED_COLUMNS),
        "optional_columns_present": [name for name in OPTIONAL_COLUMNS if name in fieldnames],
    }
    return bars, manifest


def write_samples_jsonl(samples: Iterable[TradingSample], path: str | Path) -> dict[str, Any]:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with destination.open("w", encoding="utf-8", newline="\n") as handle:
        for sample in samples:
            handle.write(json.dumps(sample.to_dict(), sort_keys=True, separators=(",", ":")))
            handle.write("\n")
            count += 1
    return {
        "path": destination.name,
        "sample_count": count,
        "sha256": sha256_file(destination),
    }


def prepare_trading_dataset(
    source_csv: str | Path,
    output_dir: str | Path,
    *,
    config: CSVIngestionConfig | None = None,
    horizons_minutes: tuple[int, ...] = DEFAULT_HORIZONS_MINUTES,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
    minimum_absolute_return_bps: float = 3.0,
    spread_multiplier: float = 1.5,
) -> dict[str, Any]:
    """Create deterministic train/validation/test JSONL files from raw CSV bars."""

    bars, source_manifest = load_market_bars_csv(source_csv, config=config)
    samples = build_causal_samples(
        bars,
        horizons_minutes=horizons_minutes,
        minimum_absolute_return_bps=minimum_absolute_return_bps,
        spread_multiplier=spread_multiplier,
    )
    if not samples:
        raise ValueError("no resolved training samples could be built from the input")

    splits = chronological_session_split(
        samples,
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
    )
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    output_files = {
        name: write_samples_jsonl(rows, destination / f"{name}.jsonl")
        for name, rows in splits.items()
    }
    manifest = {
        "schema_version": 1,
        "source": source_manifest,
        "dataset": dataset_manifest(samples, splits),
        "configuration": {
            "horizons_minutes": list(horizons_minutes),
            "train_fraction": train_fraction,
            "validation_fraction": validation_fraction,
            "minimum_absolute_return_bps": minimum_absolute_return_bps,
            "spread_multiplier": spread_multiplier,
        },
        "outputs": output_files,
    }
    manifest_path = destination / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest["manifest_sha256"] = sha256_file(manifest_path)
    return manifest
