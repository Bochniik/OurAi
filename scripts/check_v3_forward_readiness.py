"""Read-only forward-shadow readiness check for the frozen v3 magnitude model.

The only journal writes are optional synthetic events in a temporary directory.
The real append-only v2 journal, frozen champion, and protocol are never changed.
The online probe uses a small historical Alpaca IEX bars query with LOCAL keys.
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Mapping
from urllib.error import HTTPError, URLError

from scripts.run_v3_forward_shadow import DEFAULT_BUNDLE, DEFAULT_JOURNAL, _load_runtime
from trading.forward_preflight import assess_bar_coverage, choose_probe_window
from trading.forward_shadow import (
    append_journal_event, append_journal_events, journal_state, read_journal, verify_journal,
)
from trading.historical_market import (
    AlpacaCredentials, HistoricalMarketConfig, fetch_alpaca_historical_bars,
)


EXPECTED_PROTOCOL = "ourai-v3-15m-magnitude-future-forward-v2"


def journal_benchmark(event_count: int = 100) -> dict[str, Any]:
    """Compare legacy single-event calls to one batch in disposable journals."""
    if not 0 <= event_count <= 1000:
        raise ValueError("benchmark event count must be 0..1000")
    if not event_count:
        return {"event_count": 0, "skipped": True}
    rows = [
        {
            "event_id": f"diagnostic:{index}",
            "event_type": "diagnostic",
            "protocol_name": EXPECTED_PROTOCOL,
        }
        for index in range(event_count)
    ]
    with tempfile.TemporaryDirectory(prefix="ourai-shadow-preflight-") as temp:
        path = Path(temp) / "single_events.jsonl"
        batched_path = Path(temp) / "batch_events.jsonl"
        times: list[float] = []
        before_single = perf_counter()
        for row in rows:
            before = perf_counter()
            append_journal_event(path, row)
            times.append(1000 * (perf_counter() - before))
        single_total_ms = round((perf_counter() - before_single) * 1000, 2)
        before = perf_counter()
        verify_journal(read_journal(path))
        verify_ms = round((perf_counter() - before) * 1000, 2)

        before_batch = perf_counter()
        append_journal_events(batched_path, rows)
        batch_total_ms = round((perf_counter() - before_batch) * 1000, 2)
        verify_journal(read_journal(batched_path))
        if len(read_journal(path)) != len(read_journal(batched_path)):
            raise ValueError("batch and single journaling emitted different event counts")

    ordered = sorted(times)
    p95_index = max(0, int(0.95 * len(ordered) + 0.999999) - 1)
    return {
        "event_count": event_count,
        "average_append_ms": round(sum(times) / len(times), 2),
        "p95_append_ms": round(ordered[p95_index], 2),
        "single_total_ms": single_total_ms,
        "batch_total_ms": batch_total_ms,
        "final_read_verify_ms": verify_ms,
        "benchmark_is_temporary": True,
        "warning": (
            "Each *cycle* still rechecks the complete existing journal. "
            "Batching removes per-event full-history rescans, not O(N) per-cycle "
            "verification. Synthetic local timing is not a long-run guarantee. "
            "Run only one writer against a forward journal."
        ),
    }

def _safe_provider_error(exc: Exception) -> str:
    """Avoid logging environment secrets, HTTP request headers or provider bodies."""
    if isinstance(exc, HTTPError):
        return f"HTTP_{exc.code}"
    if isinstance(exc, URLError):
        return "NETWORK_ERROR"
    if isinstance(exc, TimeoutError):
        return "TIMEOUT"
    if isinstance(exc, ValueError) and "no usable bars" in str(exc):
        return "NO_USABLE_BARS"
    return type(exc).__name__.upper()


def run_preflight(
    *,
    bundle_path: str | Path = DEFAULT_BUNDLE,
    journal_path: str | Path = DEFAULT_JOURNAL,
    now_utc: datetime | None = None,
    offline: bool = False,
    benchmark_events: int = 100,
    load_bundle: Callable = _load_runtime,
    fetch_bars: Callable = fetch_alpaca_historical_bars,
) -> dict[str, Any]:
    """Validate local freeze+journal and optionally check Alpaca IEX bar access.

    Online probe is bounded and read-only. It must not create a prediction or
    backfill a prediction for the frozen forward-start window.
    """
    now = now_utc or datetime.now(timezone.utc)
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("now_utc must be timezone-aware")
    now = now.astimezone(timezone.utc)
    start = perf_counter()

    # _load_runtime verifies frozen champion/linear model SHA256, type and protocol.
    bundle, _, _ = load_bundle(bundle_path)
    if bundle.get("protocol_name") != EXPECTED_PROTOCOL:
        raise ValueError("preflight requires frozen v2 forward protocol")
    forward_start = datetime.fromisoformat(
        str(bundle["forward_start_utc"]).replace("Z", "+00:00")
    ).astimezone(timezone.utc)
    if forward_start != datetime(2026, 10, 19, 13, 30, tzinfo=timezone.utc):
        raise ValueError("unexpected forward start UTC")
    symbols = tuple(str(value).upper() for value in bundle["universe"])
    if not symbols:
        raise ValueError("bundle universe is empty")

    before_journal = perf_counter()
    events = read_journal(journal_path)
    head = verify_journal(events)
    if any(item.get("protocol_name") != EXPECTED_PROTOCOL for item in events):
        raise ValueError("v2 journal contains a different protocol: use a separate journal")
    state = journal_state(events)
    journal_elapsed = round(1000 * (perf_counter() - before_journal), 2)
    report: dict[str, Any] = {
        "schema_version": 1,
        "checked_at_utc": now.isoformat().replace("+00:00", "Z"),
        "mode": "offline" if offline else "online",
        "local_bundle": {
            "valid": True,
            "protocol_name": EXPECTED_PROTOCOL,
            "forward_start_utc": forward_start.isoformat().replace("+00:00", "Z"),
            "phase": "WAITING_FOR_START" if now < forward_start else "FORWARD_WINDOW_STARTED",
            "symbol_count": len(symbols),
            "feed": bundle.get("feed"),
            "champion_sha256": bundle["champion"]["model_sha256"],
        },
        "journal": {
            "valid": True,
            "event_count": len(events),
            "prediction_count": len(state["predictions"]),
            "resolution_count": len(state["resolutions"]),
            "pending_count": len(state["predictions"]) - len(state["resolutions"]),
            "head_sha256": head,
            "read_validate_ms": journal_elapsed,
            "real_journal_modified": False,
        },
        "performance": journal_benchmark(benchmark_events),
        "alpaca_probe": {
            "assessment": "SKIPPED_OFFLINE" if offline else "NOT_CHECKED",
            "live_freshness_confirmed": False,
            "credential_values_logged": False,
        },
    }
    if offline:
        report["overall"] = "LOCAL_OK_MARKET_NOT_CHECKED"
        report["ok"] = True
        report["total_elapsed_seconds"] = round(perf_counter() - start, 2)
        return report

    probe_window = choose_probe_window(now)
    probe_info = {
        "feed": "iex",
        "session_date": probe_window.market_date,
        "window_type": probe_window.kind,
        "window_start_utc": probe_window.start.isoformat(),
        "window_end_utc": probe_window.end.isoformat(),
        "credential_values_logged": False,
        "live_freshness_confirmed": False,
    }
    report["alpaca_probe"] = probe_info
    try:
        # Validate presence without ever copying, printing or persisting keys.
        AlpacaCredentials.from_environment()
        config = HistoricalMarketConfig(
            symbols=symbols, start=probe_window.start, end=probe_window.end,
            timeframe="1Min", feed="iex", adjustment="raw",
            sessions=("regular",), batch_size=20, page_limit=1000,
            max_pages_per_batch=3, strict_duplicates=True, allow_partial=False,
        )
        before_fetch = perf_counter()
        result = fetch_bars(config)
        probe_info["provider_fetch_ms"] = round((perf_counter() - before_fetch) * 1000, 2)
        probe_info.update(assess_bar_coverage(
            result.bars,
            required_symbols=symbols,
            window=probe_window,
            now_utc=now,
            availability_delay_minutes=float(bundle["bar_availability_delay_minutes"]),
        ))
    except Exception as exc:
        probe_info["assessment"] = (
            "MISSING_ALPACA_CREDENTIALS"
            if isinstance(exc, ValueError) and "credentials are missing" in str(exc)
            else "PROBE_ERROR"
        )
        probe_info["error_code"] = _safe_provider_error(exc)

    # Historical access is not a claim that live data can be fetched tomorrow.
    if probe_info["assessment"] == "LIVE_COVERAGE_OK":
        report["overall"] = "LIVE_PROBE_OK"
        report["ok"] = True
    elif probe_info["assessment"] == "HISTORICAL_ACCESS_OK_LIVE_UNTESTED":
        report["overall"] = "HISTORICAL_DATA_OK_LIVE_UNTESTED"
        report["ok"] = bool(probe_info.get("all_symbols_observed"))
    else:
        report["overall"] = "DATA_PROBE_NEEDS_ATTENTION"
        report["ok"] = False

    report["total_elapsed_seconds"] = round(perf_counter() - start, 2)
    return report


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Read-only diagnostics for v3 frozen forward shadow data access and journal."
    )
    result.add_argument("--bundle", default=DEFAULT_BUNDLE)
    result.add_argument("--journal", default=DEFAULT_JOURNAL)
    result.add_argument(
        "--offline", action="store_true",
        help="Check frozen models/journal without needing Alpaca credentials or network.",
    )
    result.add_argument(
        "--benchmark-events", type=int, default=100,
        help="Number of synthetic events written to a TEMPORARY journal (0 to disable).",
    )
    result.add_argument("--output", help="Optional diagnostics JSON path (no credentials stored).")
    return result


def main() -> int:
    args = parser().parse_args()
    try:
        report = run_preflight(
            bundle_path=args.bundle, journal_path=args.journal,
            offline=args.offline, benchmark_events=args.benchmark_events,
        )
    except (ValueError, OSError, KeyError, TypeError) as exc:
        # Do not echo exception payloads from untrusted local/provider files.
        print(f"PRE-FLIGHT LOCAL CHECK FAILED ({type(exc).__name__}).")
        print("Check the v2 bundle, frozen artifact paths and journal integrity.")
        return 2
    print("=== OURAI V3 FORWARD PRE-FLIGHT ===")
    print("Frozen bundle / protocol:", report["local_bundle"]["protocol_name"], "OK")
    print("Forward phase:", report["local_bundle"]["phase"])
    print("Journal:", report["journal"]["event_count"], "events;",
          report["journal"]["read_validate_ms"], "ms verification")
    print("Temporary journal benchmark:", report["performance"])
    print("Alpaca IEX:", report["alpaca_probe"]["assessment"])
    if report["alpaca_probe"].get("missing_symbols"):
        print("Symbols missing bars:", ", ".join(report["alpaca_probe"]["missing_symbols"]))
    if report["alpaca_probe"].get("stale_symbols"):
        print("Symbols with stale bars:", ", ".join(report["alpaca_probe"]["stale_symbols"]))
    if report["alpaca_probe"].get("error_code"):
        print("Alpaca error category:", report["alpaca_probe"]["error_code"])
    print("Overall:", report["overall"])
    print("No trading, model updates, retroactive predictions or real-journal writes.")
    if args.output:
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print("Report:", destination)
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
