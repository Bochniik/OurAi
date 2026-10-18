"""Read-only, fail-closed analytics for the preregistered v3 shadow journal.

No model loading, Alpaca requests, journal writes, or backfilling. Descriptive
forward metrics are not test-lockbox acceptance, profitability or direction.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from math import isfinite
from pathlib import Path
from typing import Any

from .forward_shadow import journal_state, read_journal


PROTOCOL = "ourai-v3-15m-magnitude-future-forward-v2"
FORWARD_START = datetime(2026, 10, 19, 13, 30, tzinfo=timezone.utc)
FROZEN_SYMBOLS = ("NVDA", "AAPL", "TSLA", "JPM", "AMD", "META", "MSFT")
HORIZON = timedelta(minutes=15)


def _utc(value: object, name: str) -> datetime:
    try:
        stamp = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if stamp.tzinfo is None or stamp.utcoffset() is None:
            raise ValueError(name)
        return stamp.astimezone(timezone.utc)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError(f"invalid forward event timestamp: {name}") from exc


def _nonnegative(value: object, name: str) -> float:
    try:
        if isinstance(value, bool) or value is None:
            raise ValueError(name)
        number = float(value)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError(f"invalid forward event numeric field: {name}") from exc
    if not isfinite(number) or number < 0:
        raise ValueError(f"invalid forward event numeric field: {name}")
    return number


def _average(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 3) if values else None


def read_forward_report(
    journal_path: str | Path,
    *,
    now_utc: datetime | None = None,
) -> dict[str, Any]:
    """Verify the full SHA chain and causal fields before reporting ANY figures.

    All metrics are computed from *resolved* predictions only. Missing journals
    produce an explicit empty report, not synthetic or historical estimates.
    """
    now = _utc(now_utc or datetime.now(timezone.utc), "now_utc")
    path = Path(journal_path)
    present = path.is_file()
    # journal_state() verifies hash continuity, event IDs and resolution links.
    events = read_journal(path)
    state = journal_state(events)
    for event in events:
        if event.get("protocol_name") != PROTOCOL:
            raise ValueError("forward journal contains another protocol")
        if event.get("event_type") not in {"prediction", "resolution"}:
            raise ValueError("unexpected forward journal event type")
    predictions = state["predictions"]
    resolutions = state["resolutions"]

    # Ensure a valid hash chain cannot conceal inconsistent analysis values.
    champion_hashes: set[str] = set()
    for pid, row in predictions.items():
        if row.get("event_id") != f"prediction:{pid}":
            raise ValueError("forward prediction event ID mismatch")
        if str(row.get("symbol")) not in FROZEN_SYMBOLS:
            raise ValueError("forward journal has an unexpected symbol")
        sample = _utc(row.get("sample_timestamp"), "sample_timestamp")
        source = _utc(row.get("source_bar_timestamp"), "source_bar_timestamp")
        due = _utc(row.get("target_due_at"), "target_due_at")
        target = _utc(row.get("target_bar_timestamp"), "target_bar_timestamp")
        predicted = _utc(row.get("predicted_at"), "predicted_at")
        recorded = _utc(row.get("recorded_at"), "recorded_at")
        if (sample < FORWARD_START or source + timedelta(minutes=1) != sample
                or sample + HORIZON != due or source + HORIZON != target
                or recorded < sample or recorded >= due or predicted != recorded):
            raise ValueError("forward prediction violates causal timing")
        if row.get("session_date") != sample.astimezone(
            __import__("zoneinfo").ZoneInfo("America/New_York")
        ).date().isoformat():
            raise ValueError("forward prediction session date mismatch")
        for field in ("reference_close", "expected_magnitude_bps",
                      "linear_expected_magnitude_bps", "null_expected_magnitude_bps"):
            number = _nonnegative(row.get(field), field)
            if field == "reference_close" and number == 0:
                raise ValueError("forward reference_close cannot be zero")
        model_hash = str(row.get("model_sha256") or "")
        if len(model_hash) != 64 or any(c not in "0123456789abcdef" for c in model_hash):
            raise ValueError("forward journal model hash is invalid")
        champion_hashes.add(model_hash)
    if len(champion_hashes) > 1:
        raise ValueError("forward journal mixes different champion models")

    total_errors: dict[str, list[float]] = {
        "champion": [], "linear": [], "null": []
    }
    by_symbol: dict[str, dict[str, Any]] = {
        symbol: {"predictions": 0, "resolved": 0, "pending": 0,
                 "overdue_pending": 0, "errors": {"champion": [], "linear": [], "null": []}}
        for symbol in FROZEN_SYMBOLS
    }
    last_prediction: dict[str, Any] | None = None
    for pid, prediction in predictions.items():
        sym = str(prediction["symbol"])
        target_summary = by_symbol[sym]
        target_summary["predictions"] += 1
        recorded_at = _utc(prediction["recorded_at"], "recorded_at")
        if last_prediction is None or (recorded_at, prediction["sequence"]) > (
            _utc(last_prediction["recorded_at"], "recorded_at"),
            last_prediction["sequence"],
        ):
            last_prediction = prediction
        resolution = resolutions.get(pid)
        if resolution is None:
            target_summary["pending"] += 1
            if _utc(prediction["target_due_at"], "target_due_at") <= now:
                target_summary["overdue_pending"] += 1
            continue
        if resolution.get("event_id") != f"resolution:{pid}":
            raise ValueError("forward resolution event ID mismatch")
        for field in ("symbol", "session_date", "sample_timestamp",
                      "target_due_at", "target_bar_timestamp"):
            if resolution.get(field) != prediction.get(field):
                raise ValueError(f"forward resolution does not match prediction: {field}")
        recorded_resolution = _utc(resolution.get("recorded_at"), "recorded_at")
        resolved = _utc(resolution.get("resolved_at"), "resolved_at")
        if resolved != recorded_resolution or resolved < _utc(
            prediction["target_due_at"], "target_due_at"
        ):
            raise ValueError("forward resolution recorded before its target was available")
        actual = _nonnegative(resolution.get("actual_magnitude_bps"), "actual_magnitude_bps")
        for name, forecast_field, error_field in (
            ("champion", "expected_magnitude_bps", "absolute_error_bps"),
            ("linear", "linear_expected_magnitude_bps", "linear_absolute_error_bps"),
            ("null", "null_expected_magnitude_bps", "null_absolute_error_bps"),
        ):
            calculated = abs(_nonnegative(prediction[forecast_field], forecast_field) - actual)
            recorded_error = _nonnegative(resolution.get(error_field), error_field)
            if not abs(recorded_error - calculated) <= 1e-6:
                raise ValueError(f"forward resolution {error_field} does not match forecast")
            total_errors[name].append(calculated)
            target_summary["errors"][name].append(calculated)
        target_summary["resolved"] += 1

    symbols: dict[str, Any] = {}
    for symbol, row in by_symbol.items():
        symbols[symbol] = {
            "predictions": row["predictions"],
            "resolved": row["resolved"],
            "pending": row["pending"],
            "overdue_pending": row["overdue_pending"],
            "champion_mae_bps": _average(row["errors"]["champion"]),
            "linear_mae_bps": _average(row["errors"]["linear"]),
            "null_mae_bps": _average(row["errors"]["null"]),
        }
    latest = None
    if last_prediction:
        latest = {
            "symbol": last_prediction["symbol"],
            "predicted_at": last_prediction["predicted_at"],
            "sample_timestamp": last_prediction["sample_timestamp"],
            "target_due_at": last_prediction["target_due_at"],
            "expected_magnitude_bps": _nonnegative(
                last_prediction["expected_magnitude_bps"], "expected_magnitude_bps"
            ),
            "resolved": last_prediction["prediction_id"] in resolutions,
        }
    return {
        "protocol": PROTOCOL,
        "journal_exists": present,
        "journal_valid": True,
        "journal_event_count": len(events),
        "prediction_count": len(predictions),
        "resolution_count": len(resolutions),
        "pending_count": len(predictions) - len(resolutions),
        "overdue_pending_count": sum(x["overdue_pending"] for x in symbols.values()),
        "champion_mae_bps": _average(total_errors["champion"]),
        "linear_mae_bps": _average(total_errors["linear"]),
        "null_mae_bps": _average(total_errors["null"]),
        "symbols": symbols,
        "latest_prediction": latest,
        "evaluated_at_utc": now.isoformat().replace("+00:00", "Z"),
        "source": "verified_local_forward_journal",
        "limits": (
            "15-minute absolute magnitude only; no direction probabilities. "
            "MAE uses resolved forward entries, not pending predictions. "
            "Observations are not a guarantee of trading profit or forward acceptance."
        ),
    }
