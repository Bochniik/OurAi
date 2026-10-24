"""Run append-only future-forward shadow predictions for the frozen v3 champion.

The runner has three modes:

* ``cycle``: resolve any due prior predictions, then write new current predictions;
* ``serve``: repeat ``cycle`` locally until interrupted;
* ``status``: verify and summarize the local hash-chained journal.

The model never trades. Predictions are written before their 15-minute target is
available, and resolutions are appended as separate later events.
"""

from __future__ import annotations

import argparse
import json
import pickle
import socket
import time as time_module
from collections import defaultdict
from datetime import datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping
from urllib.error import HTTPError, URLError

from trading.dataset import NEW_YORK, MarketBar
from trading.forward_shadow import (
    append_journal_events,
    build_feature_only_samples,
    canonical_json_sha256,
    journal_state,
    prediction_id,
    read_journal,
    sha256_file,
    verify_journal,
)
from trading.historical_market import (
    HistoricalMarketConfig,
    fetch_alpaca_historical_bars,
)
from trading.magnitude import UnifiedLinearMagnitudeBaseline
from trading.nonlinear import UnifiedHistGradientMagnitude


DEFAULT_BUNDLE = "runs/first-real-15m-v3/v3_15m_forward_bundle.json"
DEFAULT_JOURNAL = "runs/first-real-15m-v3/forward_shadow_v2/events.jsonl"


def shadow_failure_category(exc: BaseException) -> str:
    """Classify transport outages without exposing credentials or blaming the model.

    Categories are diagnostics only: they never change model, journal or retries.
    Follow urllib's .reason as well as exception chains for Windows socket errors.
    """
    seen: set[int] = set()
    pending: list[BaseException] = [exc]
    errors: list[BaseException] = []
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        errors.append(current)
        for related in (current.__cause__, current.__context__, getattr(current, "reason", None)):
            if isinstance(related, BaseException):
                pending.append(related)
    if any(isinstance(error, socket.gaierror) for error in errors):
        return "provider_dns_failure"
    if any(
        isinstance(error, TimeoutError)
        or getattr(error, "winerror", None) == 10060
        or getattr(error, "errno", None) in (10060, 110)
        for error in errors
    ):
        return "provider_timeout"
    if any(isinstance(error, HTTPError) for error in errors):
        return "provider_http_error"
    if any(isinstance(error, (ConnectionError, URLError)) for error in errors):
        return "provider_connection_failure"
    return "cycle_failure_unclassified"



def _parse_aware(value: object, name: str) -> datetime:
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")
    return parsed.astimezone(timezone.utc)


def _iso_z(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _load_runtime(bundle_path: str | Path) -> tuple[dict[str, Any], UnifiedHistGradientMagnitude, UnifiedLinearMagnitudeBaseline]:
    bundle_file = Path(bundle_path)
    bundle = json.loads(bundle_file.read_text(encoding="utf-8"))
    if not isinstance(bundle, dict):
        raise ValueError("forward bundle must contain a JSON object")
    if bundle.get("status") != "forward_shadow_ready":
        raise ValueError("forward bundle is not marked ready")
    if bundle.get("production_ready") is not False:
        raise ValueError("shadow runner refuses a production-ready bundle")
    if bundle.get("model_updates_during_forward_period") is not False:
        raise ValueError("forward bundle must freeze model updates")
    if str(bundle.get("session")) != "regular" or str(bundle.get("bar_timeframe")) != "1Min":
        raise ValueError("forward bundle requires regular 1Min bars")
    if float(bundle.get("bar_availability_delay_minutes", -1)) != 1.0:
        raise ValueError("forward bundle requires one-minute availability delay")
    if int(bundle.get("horizon_minutes", -1)) != 15:
        raise ValueError("this runner is frozen to the 15-minute champion")

    protocol_path = Path(str(bundle.get("protocol_path")))
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if not isinstance(protocol, dict):
        raise ValueError("forward protocol must contain a JSON object")
    if canonical_json_sha256(protocol) != str(bundle.get("protocol_sha256")):
        raise ValueError("forward protocol SHA256 changed after bundle preparation")
    if protocol.get("protocol_name") != bundle.get("protocol_name"):
        raise ValueError("forward protocol identity mismatch")
    if bundle.get("protocol_name") != "ourai-v3-15m-magnitude-future-forward-v2":
        raise ValueError("this shadow runner requires the safe future-only v2 protocol")
    if bundle.get("forward_start_utc") != "2026-10-19T13:30:00Z":
        raise ValueError("forward start does not match preregistered v2 start")

    champion = _mapping(bundle.get("champion"), "bundle.champion")
    champion_path = Path(str(champion.get("model_path")))
    if sha256_file(champion_path) != str(champion.get("model_sha256")):
        raise ValueError("champion model SHA256 changed")
    nonlinear = pickle.loads(champion_path.read_bytes())
    if not isinstance(nonlinear, UnifiedHistGradientMagnitude):
        raise ValueError("champion artifact has unexpected model type")

    incumbent = _mapping(bundle.get("linear_incumbent"), "bundle.linear_incumbent")
    linear_path = Path(str(incumbent.get("model_path")))
    if sha256_file(linear_path) != str(incumbent.get("model_sha256")):
        raise ValueError("linear incumbent SHA256 changed")
    linear = pickle.loads(linear_path.read_bytes())
    if not isinstance(linear, UnifiedLinearMagnitudeBaseline):
        raise ValueError("linear incumbent artifact has unexpected model type")

    return bundle, nonlinear, linear


def _regular_window(now_utc: datetime) -> tuple[datetime, datetime]:
    local = now_utc.astimezone(NEW_YORK)
    session_open = datetime.combine(local.date(), time(9, 30), tzinfo=NEW_YORK)
    session_close = datetime.combine(local.date(), time(16, 0), tzinfo=NEW_YORK)
    return session_open.astimezone(timezone.utc), session_close.astimezone(timezone.utc)


def _fetch_bars(
    *,
    symbols: list[str],
    start: datetime,
    end: datetime,
    feed: str,
) -> list[MarketBar]:
    if end <= start:
        return []
    config = HistoricalMarketConfig(
        symbols=tuple(symbols),
        start=start,
        end=end,
        timeframe="1Min",
        feed=feed,
        adjustment="raw",
        sessions=("regular",),
        batch_size=20,
        page_limit=10_000,
        max_pages_per_batch=20,
        strict_duplicates=True,
        allow_partial=False,
    )
    try:
        result = fetch_alpaca_historical_bars(config)
    except ValueError as exc:
        if "no usable bars" in str(exc):
            return []
        raise
    return list(result.bars)


def _pending_due_predictions(
    state: Mapping[str, Mapping[str, dict[str, Any]]],
    *,
    now_utc: datetime,
) -> list[dict[str, Any]]:
    predictions = state["predictions"]
    resolutions = state["resolutions"]
    due: list[dict[str, Any]] = []
    for pid, row in predictions.items():
        if pid in resolutions:
            continue
        if _parse_aware(row.get("target_due_at"), "target_due_at") <= now_utc:
            due.append(dict(row))
    return due


def _resolve_due(
    *,
    bundle: Mapping[str, Any],
    journal_path: str | Path,
    state: Mapping[str, Mapping[str, dict[str, Any]]],
    now_utc: datetime,
) -> int:
    due = _pending_due_predictions(state, now_utc=now_utc)
    if not due:
        return 0

    by_date: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in due:
        by_date[str(row["session_date"])].append(row)

    symbols = [str(value).upper() for value in bundle["universe"]]
    feed = str(bundle.get("feed") or "iex")
    resolution_events: list[dict[str, Any]] = []
    for rows in by_date.values():
        target_times = [_parse_aware(row["target_bar_timestamp"], "target_bar_timestamp") for row in rows]
        bars = _fetch_bars(
            symbols=symbols,
            start=min(target_times) - timedelta(minutes=1),
            end=max(target_times) + timedelta(minutes=2),
            feed=feed,
        )
        by_key = {(bar.symbol, bar.timestamp): bar for bar in bars}
        for row in rows:
            pid = str(row["prediction_id"])
            target_timestamp = _parse_aware(row["target_bar_timestamp"], "target_bar_timestamp")
            target = by_key.get((str(row["symbol"]).upper(), target_timestamp))
            if target is None:
                continue
            reference_close = float(row["reference_close"])
            actual_bps = abs(target.close / reference_close - 1.0) * 10_000.0
            nonlinear_error = abs(float(row["expected_magnitude_bps"]) - actual_bps)
            linear_error = abs(float(row["linear_expected_magnitude_bps"]) - actual_bps)
            null_error = abs(float(row["null_expected_magnitude_bps"]) - actual_bps)
            resolution_events.append(
                {
                    "event_id": f"resolution:{pid}",
                    "event_type": "resolution",
                    "prediction_id": pid,
                    "protocol_name": bundle["protocol_name"],
                    "symbol": row["symbol"],
                    "session_date": row["session_date"],
                    "sample_timestamp": row["sample_timestamp"],
                    "target_due_at": row["target_due_at"],
                    "target_bar_timestamp": row["target_bar_timestamp"],
                    "target_close": target.close,
                    "actual_magnitude_bps": actual_bps,
                    "absolute_error_bps": nonlinear_error,
                    "linear_absolute_error_bps": linear_error,
                    "null_absolute_error_bps": null_error,
                    "resolved_at": _iso_z(now_utc),
                }
            )
    # Verify the existing hash chain only once, then fsync the batch once.
    return len(append_journal_events(journal_path, resolution_events, recorded_at=now_utc))


def _write_current_predictions(
    *,
    bundle: Mapping[str, Any],
    nonlinear: UnifiedHistGradientMagnitude,
    linear: UnifiedLinearMagnitudeBaseline,
    journal_path: str | Path,
    state: Mapping[str, Mapping[str, dict[str, Any]]],
    now_utc: datetime,
    max_prediction_lag_seconds: float,
) -> int:
    forward_start = _parse_aware(bundle["forward_start_utc"], "forward_start_utc")
    if now_utc < forward_start:
        return 0

    session_open, session_close = _regular_window(now_utc)
    if now_utc < session_open + timedelta(minutes=1) or now_utc > session_close:
        return 0

    symbols = [str(value).upper() for value in bundle["universe"]]
    feed = str(bundle.get("feed") or "iex")
    # Fetch only information available by now. A provider bar timestamp is its left edge;
    # build_feature_only_samples applies the frozen +1 minute availability timestamp.
    bars = _fetch_bars(
        symbols=symbols,
        start=session_open,
        end=now_utc,
        feed=feed,
    )
    if not bars:
        return 0

    by_bar_key = {(bar.symbol, bar.timestamp): bar for bar in bars}
    samples = build_feature_only_samples(
        bars,
        bar_availability_delay_minutes=float(bundle["bar_availability_delay_minutes"]),
    )
    existing = state["predictions"]
    horizon = timedelta(minutes=int(bundle["horizon_minutes"]))
    delay = timedelta(minutes=float(bundle["bar_availability_delay_minutes"]))
    null_bps = float(_mapping(bundle["null"], "bundle.null")["magnitude_bps"])
    champion_sha = str(_mapping(bundle["champion"], "bundle.champion")["model_sha256"])
    linear_sha = str(_mapping(bundle["linear_incumbent"], "bundle.linear_incumbent")["model_sha256"])

    prediction_events: list[dict[str, Any]] = []
    for sample in samples:
        if sample.session != "regular" or sample.timestamp < forward_start or sample.timestamp > now_utc:
            continue
        lag_seconds = (now_utc - sample.timestamp).total_seconds()
        if lag_seconds < 0 or lag_seconds > max_prediction_lag_seconds:
            continue
        target_due = sample.timestamp + horizon
        if target_due > session_close or now_utc >= target_due:
            continue

        pid = prediction_id(
            protocol_name=str(bundle["protocol_name"]),
            model_sha256=champion_sha,
            symbol=sample.symbol,
            sample_timestamp=sample.timestamp,
        )
        if pid in existing:
            continue

        source_bar_timestamp = sample.timestamp - delay
        source_bar = by_bar_key.get((sample.symbol, source_bar_timestamp))
        if source_bar is None:
            continue
        target_bar_timestamp = source_bar_timestamp + horizon

        nonlinear_prediction = nonlinear.predict_sample(sample, include_actual=False)
        linear_prediction = linear.predict_sample(sample, include_actual=False)
        feature_payload = {
            "symbol": sample.symbol,
            "sample_timestamp": _iso_z(sample.timestamp),
            "features": dict(sample.features),
        }
        prediction_events.append(
            {
                "event_id": f"prediction:{pid}",
                "event_type": "prediction",
                "prediction_id": pid,
                "protocol_name": bundle["protocol_name"],
                "protocol_sha256": bundle["protocol_sha256"],
                "symbol": sample.symbol,
                "session_date": sample.session_date,
                "sample_timestamp": _iso_z(sample.timestamp),
                "source_bar_timestamp": _iso_z(source_bar_timestamp),
                "reference_close": source_bar.close,
                "predicted_at": _iso_z(now_utc),
                "prediction_lag_seconds": lag_seconds,
                "target_due_at": _iso_z(target_due),
                "target_bar_timestamp": _iso_z(target_bar_timestamp),
                "expected_magnitude_bps": nonlinear_prediction.expected_magnitude_bps,
                "linear_expected_magnitude_bps": linear_prediction.expected_magnitude_bps,
                "null_expected_magnitude_bps": null_bps,
                "model_sha256": champion_sha,
                "linear_model_sha256": linear_sha,
                "feature_payload_sha256": canonical_json_sha256(feature_payload),
            }
        )
    committed = append_journal_events(journal_path, prediction_events, recorded_at=now_utc)
    for event in committed:
        existing[str(event["prediction_id"])] = {"prediction_id": event["prediction_id"]}
    return len(committed)


def run_cycle(
    *,
    bundle_path: str | Path,
    journal_path: str | Path,
    now_utc: datetime | None = None,
    max_prediction_lag_seconds: float = 90.0,
) -> dict[str, Any]:
    now = (now_utc or datetime.now(timezone.utc)).astimezone(timezone.utc)
    if max_prediction_lag_seconds <= 0:
        raise ValueError("max_prediction_lag_seconds must be positive")
    bundle, nonlinear, linear = _load_runtime(bundle_path)

    events = read_journal(journal_path)
    verify_journal(events)
    if any(event.get("protocol_name") != bundle.get("protocol_name") for event in events):
        raise ValueError("forward journal contains events from another protocol; use a fresh v2 journal")
    state = journal_state(events)
    resolved = _resolve_due(
        bundle=bundle,
        journal_path=journal_path,
        state=state,
        now_utc=now,
    )

    # Reload after appending resolutions so the next phase sees current state.
    events = read_journal(journal_path)
    state = journal_state(events)
    predicted = _write_current_predictions(
        bundle=bundle,
        nonlinear=nonlinear,
        linear=linear,
        journal_path=journal_path,
        state=state,
        now_utc=now,
        max_prediction_lag_seconds=max_prediction_lag_seconds,
    )
    final_events = read_journal(journal_path)
    final_state = journal_state(final_events)
    pending = len(final_state["predictions"]) - len(final_state["resolutions"])
    return {
        "now_utc": _iso_z(now),
        "predictions_appended": predicted,
        "resolutions_appended": resolved,
        "prediction_count": len(final_state["predictions"]),
        "resolution_count": len(final_state["resolutions"]),
        "pending_count": pending,
        "journal_event_count": len(final_events),
        "journal_head_sha256": verify_journal(final_events),
    }


def status(journal_path: str | Path) -> dict[str, Any]:
    events = read_journal(journal_path)
    head = verify_journal(events)
    state = journal_state(events)
    predictions = state["predictions"]
    resolutions = state["resolutions"]
    dates = sorted({str(row.get("session_date")) for row in predictions.values()})
    resolved_dates = sorted(
        {
            str(predictions[pid].get("session_date"))
            for pid in resolutions
            if pid in predictions
        }
    )
    return {
        "journal_valid": True,
        "event_count": len(events),
        "prediction_count": len(predictions),
        "resolution_count": len(resolutions),
        "pending_count": len(predictions) - len(resolutions),
        "market_dates_with_predictions": dates,
        "market_dates_with_resolutions": resolved_dates,
        "journal_head_sha256": head,
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Run the frozen v3 15m magnitude future-forward shadow logger/resolver.")
    result.add_argument("--bundle", default=DEFAULT_BUNDLE)
    result.add_argument("--journal", default=DEFAULT_JOURNAL)
    result.add_argument("--mode", choices=("cycle", "serve", "status"), default="cycle")
    result.add_argument("--poll-seconds", type=float, default=20.0)
    result.add_argument("--max-prediction-lag-seconds", type=float, default=90.0)
    return result


def _print_cycle(payload: Mapping[str, Any]) -> None:
    print(
        f"{payload['now_utc']} predicted={payload['predictions_appended']} "
        f"resolved={payload['resolutions_appended']} pending={payload['pending_count']} "
        f"events={payload['journal_event_count']}"
    )


def main() -> int:
    args = parser().parse_args()
    if args.mode == "status":
        payload = status(args.journal)
        print("=== V3 FORWARD SHADOW STATUS ===")
        print("Journal valid:", payload["journal_valid"])
        print("Predictions:", payload["prediction_count"])
        print("Resolutions:", payload["resolution_count"])
        print("Pending:", payload["pending_count"])
        print("Market dates with predictions:", len(payload["market_dates_with_predictions"]))
        print("Journal head SHA256:", payload["journal_head_sha256"])
        return 0

    if args.mode == "cycle":
        _print_cycle(
            run_cycle(
                bundle_path=args.bundle,
                journal_path=args.journal,
                max_prediction_lag_seconds=args.max_prediction_lag_seconds,
            )
        )
        return 0

    if args.poll_seconds < 5:
        raise ValueError("--poll-seconds must be at least 5")
    print("=== V3 FORWARD SHADOW SERVE ===")
    print("Mode: SHADOW ONLY - NO TRADING")
    print("Press Ctrl+C to stop.")
    try:
        while True:
            try:
                payload = run_cycle(
                    bundle_path=args.bundle,
                    journal_path=args.journal,
                    max_prediction_lag_seconds=args.max_prediction_lag_seconds,
                )
                if payload["predictions_appended"] or payload["resolutions_appended"]:
                    _print_cycle(payload)
            except Exception as exc:
                # Keep serving, but never echo exception messages (URLs may carry secrets).
                # A provider outage is not evidence of a model prediction error.
                print(f"shadow cycle failure: {shadow_failure_category(exc)}")
            time_module.sleep(args.poll_seconds)
    except KeyboardInterrupt:
        print("Forward shadow runner stopped.")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
