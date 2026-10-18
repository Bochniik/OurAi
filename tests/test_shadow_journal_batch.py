"""Regression coverage for verified batch appends and shadow-runner integration."""
from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from scripts.run_v3_forward_shadow import _resolve_due, _write_current_predictions
from trading.dataset import MarketBar
from trading.forward_shadow import (
    append_journal_event, append_journal_events, journal_state, read_journal,
    verify_journal,
)

TIME = datetime(2026, 10, 19, 13, 35, tzinfo=timezone.utc)
PROTOCOL = "ourai-v3-15m-magnitude-future-forward-v2"


def _event(event_id: str, kind: str = "diagnostic") -> dict:
    return {"event_id": event_id, "event_type": kind, "protocol_name": PROTOCOL}


def _bundle() -> dict:
    return {
        "protocol_name": PROTOCOL,
        "protocol_sha256": "protocol-hash",
        "forward_start_utc": "2026-10-19T13:30:00Z",
        "universe": ["AAPL", "AMD"],
        "feed": "iex",
        "bar_availability_delay_minutes": 1,
        "horizon_minutes": 15,
        "null": {"magnitude_bps": 12.5},
        "champion": {"model_sha256": "champion-sha"},
        "linear_incumbent": {"model_sha256": "linear-sha"},
    }


def _bars(until_index: int = 5) -> list[MarketBar]:
    start = datetime(2026, 10, 19, 13, 30, tzinfo=timezone.utc)
    rows: list[MarketBar] = []
    for symbol in ("AAPL", "AMD"):
        for index in range(until_index):
            t = start + timedelta(minutes=index)
            rows.append(
                MarketBar(
                    symbol=symbol, timestamp=t, session="regular",
                    open=100.0, high=101.0, low=99.0,
                    close=100.0 + index * 0.1, volume=1500,
                )
            )
    return rows


class ModelStub:
    def predict_sample(self, sample, *, include_actual=False):
        if include_actual:
            raise AssertionError("live predictions must not read targets")
        return SimpleNamespace(expected_magnitude_bps=18.0)


class BatchJournalTests(unittest.TestCase):
    def test_batch_matches_repeated_single_event_hash_chain_byte_for_byte(self):
        when = datetime(2026, 10, 19, 13, 32, tzinfo=timezone.utc)
        rows = [_event(f"diagnostic:{i}") for i in range(7)]
        with tempfile.TemporaryDirectory() as temp:
            single = Path(temp) / "single.jsonl"
            batch = Path(temp) / "batch.jsonl"
            for event in rows:
                append_journal_event(single, event, recorded_at=when)
            appended = append_journal_events(batch, rows, recorded_at=when)
            self.assertEqual(len(appended), 7)
            self.assertEqual(single.read_bytes(), batch.read_bytes())
            self.assertEqual(verify_journal(read_journal(batch)), appended[-1]["event_sha256"])

            after = append_journal_event(batch, _event("last"), recorded_at=when)
            self.assertEqual(after["sequence"], 7)
            self.assertEqual(after["prev_event_sha256"], appended[-1]["event_sha256"])

    def test_reject_duplicates_before_appending_any_bytes(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "events.jsonl"
            append_journal_event(path, _event("already"))
            original = path.read_bytes()
            for input_rows in (
                [_event("new"), _event("new")],
                [_event("new"), _event("already")],
                [_event("new"), {"event_type": "diagnostic"}],
                [_event("new"), {"event_id": "reserved", "sequence": 123}],
            ):
                with self.subTest(rows=input_rows):
                    with self.assertRaises(ValueError):
                        append_journal_events(path, input_rows)
                    self.assertEqual(path.read_bytes(), original)

    def test_empty_batch_does_not_create_a_file(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "missing" / "events.jsonl"
            self.assertEqual(append_journal_events(path, []), [])
            self.assertFalse(path.exists())

    def test_tampered_or_truncated_journal_blocks_next_batch_without_repair(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "events.jsonl"
            append_journal_event(path, _event("before"))
            with path.open("ab") as handle:
                handle.write(b'{"event_id":"truncated"')
            corrupt_bytes = path.read_bytes()
            with self.assertRaises(ValueError):
                append_journal_events(path, [_event("later")])
            self.assertEqual(path.read_bytes(), corrupt_bytes)

    def test_one_full_existing_journal_read_and_one_fsync_per_batch(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "events.jsonl"
            append_journal_event(path, _event("prior"))
            with patch("trading.forward_shadow.read_journal", wraps=read_journal) as read_mock, \
                 patch("trading.forward_shadow.os.fsync") as fsync_mock:
                append_journal_events(path, [_event(f"new:{i}") for i in range(10)])
                self.assertEqual(read_mock.call_count, 1)
                self.assertEqual(fsync_mock.call_count, 1)
            self.assertEqual(len(read_journal(path)), 11)

    def test_restart_after_batch_keeps_pending_and_resolved_state(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "events.jsonl"
            predictions = [
                {**_event(f"prediction:{i}", kind="prediction"), "prediction_id": str(i)}
                for i in range(3)
            ]
            append_journal_events(path, predictions)
            append_journal_events(path, [
                {**_event("resolution:0", kind="resolution"), "prediction_id": "0"}
            ])
            state = journal_state(read_journal(path))
            self.assertEqual(len(state["predictions"]), 3)
            self.assertEqual(len(state["resolutions"]), 1)
            self.assertEqual(len(state["predictions"]) - len(state["resolutions"]), 2)

    def test_runner_batches_predictions_and_later_resolutions(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "events.jsonl"
            state = {"predictions": {}, "resolutions": {}}
            with patch("scripts.run_v3_forward_shadow._fetch_bars", return_value=_bars()), \
                 patch("trading.forward_shadow.read_journal", wraps=read_journal) as read_mock:
                written = _write_current_predictions(
                    bundle=_bundle(), nonlinear=ModelStub(), linear=ModelStub(),
                    journal_path=path, state=state, now_utc=TIME,
                    max_prediction_lag_seconds=90,
                )
                self.assertGreater(written, 1)
                self.assertEqual(read_mock.call_count, 1)
            self.assertEqual(len(state["predictions"]), written)
            pre = read_journal(path)
            self.assertEqual(len(pre), written)
            self.assertTrue(all(row["event_type"] == "prediction" for row in pre))
            self.assertTrue(all(row["predicted_at"].endswith("Z") for row in pre))

            # Resolve all due 15m later, after the original causal predictions
            # have been committed and without rewriting a prediction event.
            target_bars = []
            for row in pre:
                source = datetime.fromisoformat(row["target_bar_timestamp"].replace("Z", "+00:00"))
                target_bars.append(MarketBar(
                    symbol=row["symbol"], timestamp=source, session="regular",
                    open=101, high=102, low=100, close=101, volume=1000,
                ))
            with patch("scripts.run_v3_forward_shadow._fetch_bars", return_value=target_bars), \
                 patch("trading.forward_shadow.read_journal", wraps=read_journal) as read_mock:
                resolved = _resolve_due(
                    bundle=_bundle(), journal_path=path,
                    state=journal_state(pre), now_utc=TIME + timedelta(minutes=18),
                )
                self.assertEqual(read_mock.call_count, 1)
            self.assertEqual(resolved, written)
            events = read_journal(path)
            self.assertEqual(events[:written], pre)
            self.assertEqual(len(events), 2 * written)
            self.assertEqual(len(journal_state(events)["resolutions"]), written)
            verify_journal(events)


if __name__ == "__main__":
    unittest.main()
