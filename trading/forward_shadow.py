"""Future-forward shadow evaluation primitives for frozen OurAI models.

This module is intentionally small and audit-focused. It provides feature-only causal
sample construction for live prediction plus a hash-chained append-only JSONL journal.
Predictions and later resolutions are separate events so an outcome is never written
into or used to rewrite the original prediction record.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timedelta, timezone
from math import isfinite
from pathlib import Path
from typing import Any, Iterable, Mapping

from .dataset import (
    NEW_YORK,
    MarketBar,
    TradingSample,
    _build_features,
    infer_session,
)


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamp must be timezone-aware")
    return value.astimezone(timezone.utc)


def build_feature_only_samples(
    bars: Iterable[MarketBar | Mapping[str, Any]],
    *,
    bar_availability_delay_minutes: float = 1.0,
) -> list[TradingSample]:
    """Build causal feature rows without requiring or inspecting future targets.

    The logic deliberately mirrors :func:`trading.dataset.build_causal_samples` for
    features and timestamp semantics, but emits every observed bar with an empty
    target mapping. This is the path used for genuine forward predictions.
    """

    delay = float(bar_availability_delay_minutes)
    if not isfinite(delay) or delay < 0:
        raise ValueError("bar_availability_delay_minutes must be finite and non-negative")
    availability_delay = timedelta(minutes=delay)

    normalized = [
        item if isinstance(item, MarketBar) else MarketBar.from_mapping(item)
        for item in bars
    ]
    normalized.sort(key=lambda item: (item.symbol, item.timestamp))

    by_symbol: dict[str, list[MarketBar]] = {}
    for bar in normalized:
        by_symbol.setdefault(bar.symbol, []).append(bar)

    samples: list[TradingSample] = []
    for symbol, symbol_bars in by_symbol.items():
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
            samples.append(
                TradingSample(
                    symbol=symbol,
                    timestamp=availability_timestamp,
                    session_date=session_date,
                    session=session,
                    features=features,
                    targets={},
                )
            )

    return samples


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _event_hash_payload(event: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in event.items() if key != "event_sha256"}


def read_journal(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    if not source.exists():
        return []
    rows: list[dict[str, Any]] = []
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid shadow journal JSON at line {line_number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"shadow journal line {line_number} must be an object")
            rows.append(row)
    return rows


def verify_journal(events: Iterable[Mapping[str, Any]]) -> str | None:
    """Verify sequence numbers and the full hash chain; return the final event hash."""

    previous: str | None = None
    seen_event_ids: set[str] = set()
    for expected_sequence, raw in enumerate(events):
        event = dict(raw)
        if int(event.get("sequence", -1)) != expected_sequence:
            raise ValueError("shadow journal sequence is not contiguous")
        if event.get("prev_event_sha256") != previous:
            raise ValueError("shadow journal previous-hash link is invalid")
        event_id = str(event.get("event_id") or "")
        if not event_id or event_id in seen_event_ids:
            raise ValueError("shadow journal event_id is missing or duplicated")
        seen_event_ids.add(event_id)
        stored_hash = str(event.get("event_sha256") or "")
        calculated = canonical_json_sha256(_event_hash_payload(event))
        if stored_hash != calculated:
            raise ValueError("shadow journal event hash mismatch")
        previous = stored_hash
    return previous


def append_journal_events(
    path: str | Path,
    events: Iterable[Mapping[str, Any]],
    *,
    recorded_at: datetime | None = None,
) -> list[dict[str, Any]]:
    """Append an ordered batch with one full-chain check and one durable flush.

    The on-disk JSONL schema, hash algorithm and event ordering are identical
    to repeated single-event appends. Every event ID (existing or new) is
    validated BEFORE any bytes are written. Existing bytes are never rewritten.

    This requires a *single writer* for the journal. No concurrency/atomic
    multi-event transaction guarantee is implied: an interrupted append may
    leave a truncated JSON line, and subsequent checks must fail closed.
    """
    incoming = [dict(item) for item in events]
    if not incoming:
        return []

    destination = Path(path)
    # Validate timestamp and all submitted events before creating directories.
    timestamp = _as_utc(recorded_at or datetime.now(timezone.utc))
    seen_incoming: set[str] = set()
    for payload in incoming:
        for reserved in ("sequence", "prev_event_sha256", "event_sha256", "recorded_at"):
            if reserved in payload:
                raise ValueError(f"caller may not set reserved journal field {reserved!r}")
        event_id = str(payload.get("event_id") or "").strip()
        if not event_id:
            raise ValueError("event_id is required")
        if event_id in seen_incoming:
            raise ValueError(f"duplicate event_id in batch: {event_id}")
        seen_incoming.add(event_id)

    existing = read_journal(destination)
    previous = verify_journal(existing)
    existing_ids = {str(row["event_id"]) for row in existing}
    duplicated = existing_ids & seen_incoming
    if duplicated:
        raise ValueError(f"duplicate event_id: {sorted(duplicated)[0]}")

    result: list[dict[str, Any]] = []
    for index, payload in enumerate(incoming, start=len(existing)):
        payload["recorded_at"] = timestamp.isoformat().replace("+00:00", "Z")
        payload["sequence"] = index
        payload["prev_event_sha256"] = previous
        payload["event_sha256"] = canonical_json_sha256(payload)
        previous = payload["event_sha256"]
        result.append(payload)

    lines = "".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
        for row in result
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Do not remove or repair a truncated/corrupt journal automatically.
    with destination.open("a", encoding="utf-8", newline="") as handle:
        handle.write(lines)
        handle.flush()
        os.fsync(handle.fileno())
    return result


def append_journal_event(
    path: str | Path,
    event: Mapping[str, Any],
    *,
    recorded_at: datetime | None = None,
) -> dict[str, Any]:
    """Backward-compatible single-event append using the same verified writer."""
    return append_journal_events(path, [event], recorded_at=recorded_at)[0]

def prediction_id(
    *,
    protocol_name: str,
    model_sha256: str,
    symbol: str,
    sample_timestamp: datetime,
) -> str:
    raw = "|".join(
        (
            str(protocol_name),
            str(model_sha256),
            str(symbol).upper(),
            _as_utc(sample_timestamp).isoformat(),
        )
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


def journal_state(events: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """Return prediction/resolution maps after semantic journal validation."""

    rows = [dict(row) for row in events]
    verify_journal(rows)
    predictions: dict[str, dict[str, Any]] = {}
    resolutions: dict[str, dict[str, Any]] = {}
    for row in rows:
        kind = row.get("event_type")
        if kind == "prediction":
            pid = str(row.get("prediction_id") or "")
            if not pid or pid in predictions:
                raise ValueError("prediction_id is missing or duplicated")
            predictions[pid] = row
        elif kind == "resolution":
            pid = str(row.get("prediction_id") or "")
            if pid not in predictions:
                raise ValueError("resolution refers to an unknown prediction")
            if pid in resolutions:
                raise ValueError("prediction has more than one resolution")
            resolutions[pid] = row
    return {"predictions": predictions, "resolutions": resolutions}
