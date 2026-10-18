from __future__ import annotations

import json
import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from scripts.check_v3_forward_readiness import journal_benchmark, run_preflight
from trading.dataset import MarketBar
from trading.forward_preflight import (
    ProbeWindow, assess_bar_coverage, choose_probe_window,
)
from trading.forward_shadow import append_journal_event
from trading.historical_market import HistoricalFetchResult


SYMBOLS = ("NVDA", "AAPL", "TSLA", "JPM", "AMD", "META", "MSFT")
START = datetime(2026, 10, 19, 13, 30, tzinfo=timezone.utc)


def make_bar(symbol: str, moment: datetime) -> MarketBar:
    return MarketBar(
        symbol=symbol, timestamp=moment, open=100, high=101,
        low=99, close=100, volume=1000, session="regular",
    )


def fake_bundle(_path):
    return ({
        "protocol_name": "ourai-v3-15m-magnitude-future-forward-v2",
        "forward_start_utc": "2026-10-19T13:30:00Z",
        "universe": list(SYMBOLS),
        "feed": "iex",
        "bar_availability_delay_minutes": 1,
        "champion": {"model_sha256": "frozen-model-checksum"},
    }, object(), object())


class ForwardPreflightTests(unittest.TestCase):
    def test_sunday_probes_previous_friday_not_empty_weekend(self):
        sunday = datetime(2026, 10, 18, 10, 11, tzinfo=timezone.utc)
        window = choose_probe_window(sunday)
        self.assertEqual(window.kind, "historical")
        self.assertEqual(window.market_date, "2026-10-16")
        self.assertEqual(window.start, datetime(2026, 10, 16, 19, 40, tzinfo=timezone.utc))
        self.assertEqual(window.end, datetime(2026, 10, 16, 20, 0, tzinfo=timezone.utc))

    def test_monday_preopen_is_only_historical_access(self):
        window = choose_probe_window(datetime(2026, 10, 19, 13, 25, tzinfo=timezone.utc))
        self.assertEqual(window.market_date, "2026-10-16")
        self.assertEqual(window.kind, "historical")

    def test_open_market_with_fresh_bars_can_certify_data_coverage(self):
        now = datetime(2026, 10, 19, 13, 50, tzinfo=timezone.utc)
        window = choose_probe_window(now)
        bars = [make_bar(symbol, now - timedelta(minutes=3)) for symbol in SYMBOLS]
        result = assess_bar_coverage(bars, required_symbols=SYMBOLS, window=window, now_utc=now)
        self.assertEqual(window.kind, "intraday")
        self.assertTrue(result["live_freshness_confirmed"])
        self.assertEqual(result["assessment"], "LIVE_COVERAGE_OK")
        self.assertEqual(result["bar_count"], 7)
        self.assertEqual(result["missing_symbols"], [])

    def test_missing_or_stale_symbol_blocks_live_ready(self):
        now = datetime(2026, 10, 19, 13, 50, tzinfo=timezone.utc)
        window = choose_probe_window(now)
        bars = [make_bar(symbol, now - timedelta(minutes=3)) for symbol in SYMBOLS[:-1]]
        bars.append(make_bar(SYMBOLS[-2], now - timedelta(minutes=12)))
        result = assess_bar_coverage(bars, required_symbols=SYMBOLS, window=window, now_utc=now)
        self.assertFalse(result["live_freshness_confirmed"])
        self.assertEqual(result["missing_symbols"], ["MSFT"])
        # AMD bar is stale, but its other recent bar remains the latest.
        self.assertEqual(result["stale_symbols"], [])
        stale_bars = [make_bar(symbol, now - timedelta(minutes=12)) for symbol in SYMBOLS]
        result = assess_bar_coverage(stale_bars, required_symbols=SYMBOLS, window=window, now_utc=now)
        self.assertIn("NVDA", result["stale_symbols"])
        self.assertFalse(result["live_freshness_confirmed"])

    def test_historical_bars_do_not_claim_live_freshness(self):
        sunday = datetime(2026, 10, 18, 10, 11, tzinfo=timezone.utc)
        window = choose_probe_window(sunday)
        bars = [make_bar(symbol, window.end - timedelta(minutes=1)) for symbol in SYMBOLS]
        result = assess_bar_coverage(bars, required_symbols=SYMBOLS, window=window, now_utc=sunday)
        self.assertTrue(result["all_symbols_observed"])
        self.assertFalse(result["live_freshness_confirmed"])
        self.assertEqual(result["assessment"], "HISTORICAL_ACCESS_OK_LIVE_UNTESTED")

    def test_offline_checks_local_without_credentials_or_mutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            journal = Path(tmp) / "events.jsonl"
            def no_network(config):
                raise AssertionError("offline mode attempted a provider request")
            report = run_preflight(
                journal_path=journal, offline=True, benchmark_events=8,
                load_bundle=fake_bundle, fetch_bars=no_network, now_utc=START - timedelta(days=1),
            )
            self.assertTrue(report["ok"])
            self.assertEqual(report["overall"], "LOCAL_OK_MARKET_NOT_CHECKED")
            self.assertEqual(report["journal"]["event_count"], 0)
            self.assertFalse(journal.exists())
            self.assertTrue(report["performance"]["benchmark_is_temporary"])
            self.assertEqual(report["performance"]["event_count"], 8)

    def test_online_market_probe_checks_each_frozen_symbol_without_logging_keys(self):
        now = datetime(2026, 10, 19, 13, 50, tzinfo=timezone.utc)
        calls = []
        def fake_fetch(config):
            calls.append(config)
            bars = tuple(make_bar(symbol, now - timedelta(minutes=3)) for symbol in SYMBOLS)
            return HistoricalFetchResult(bars=bars, manifest={})
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {
                "ALPACA_API_KEY_ID": "NEVER_LOG_KEY",
                "ALPACA_API_SECRET_KEY": "NEVER_LOG_SECRET",
            }):
                report = run_preflight(
                    journal_path=Path(tmp) / "events.jsonl", now_utc=now,
                    benchmark_events=0, load_bundle=fake_bundle, fetch_bars=fake_fetch,
                )
        self.assertTrue(report["ok"])
        self.assertEqual(report["overall"], "LIVE_PROBE_OK")
        self.assertEqual(calls[0].symbols, SYMBOLS)
        self.assertEqual(calls[0].feed, "iex")
        self.assertEqual(calls[0].sessions, ("regular",))
        self.assertNotIn("NEVER_LOG_KEY", json.dumps(report))
        self.assertNotIn("NEVER_LOG_SECRET", json.dumps(report))
        self.assertTrue(report["alpaca_probe"]["live_freshness_confirmed"])

    def test_missing_credentials_is_actionable_and_does_not_fetch(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {}, clear=True):
                report = run_preflight(
                    journal_path=Path(tmp) / "events.jsonl",
                    now_utc=START - timedelta(days=1),
                    offline=False, benchmark_events=0, load_bundle=fake_bundle,
                    fetch_bars=lambda config: self.fail("fetch without credentials"),
                )
        self.assertFalse(report["ok"])
        self.assertEqual(report["alpaca_probe"]["assessment"], "MISSING_ALPACA_CREDENTIALS")

    def test_mixed_protocol_journal_is_rejected_without_overwriting(self):
        with tempfile.TemporaryDirectory() as tmp:
            journal = Path(tmp) / "events.jsonl"
            append_journal_event(journal, {
                "event_id": "old:0", "event_type": "diagnostic",
                "protocol_name": "old-v1",
            })
            content = journal.read_bytes()
            with self.assertRaisesRegex(ValueError, "different protocol"):
                run_preflight(
                    journal_path=journal, offline=True, benchmark_events=0,
                    load_bundle=fake_bundle,
                )
            self.assertEqual(content, journal.read_bytes())

    def test_rejects_unaware_clock(self):
        with self.assertRaisesRegex(ValueError, "timezone-aware"):
            choose_probe_window(datetime(2026, 10, 19, 13, 30))

    def test_benchmark_zero_is_explicitly_skipped(self):
        self.assertTrue(journal_benchmark(0)["skipped"])


if __name__ == "__main__":
    unittest.main()
