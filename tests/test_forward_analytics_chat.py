"""Read-only forward journal analytics: real schema, security and chat routing."""
from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from ourai_language.forward_answers import forward_question_kind, maybe_forward_answer
from ourai_language.ollama_runtime import LocalOllamaChatRuntime
from ourai_language.runtime import OurAIChatRuntime
from trading.forward_analytics import read_forward_report
from trading.forward_shadow import append_journal_events, read_journal, verify_journal

PROTOCOL = "ourai-v3-15m-magnitude-future-forward-v2"
SAMPLE = datetime(2026, 10, 19, 13, 31, tzinfo=timezone.utc)
PREDICTED = SAMPLE + timedelta(seconds=30)
DUE = SAMPLE + timedelta(minutes=15)
RESOLVED = DUE + timedelta(minutes=2)
NOW = DUE + timedelta(minutes=10)


def iso(value):
    return value.isoformat().replace("+00:00", "Z")


def prediction(symbol="NVDA", pid="one", **overrides):
    row = {
        "event_id": f"prediction:{pid}",
        "event_type": "prediction",
        "prediction_id": pid,
        "protocol_name": PROTOCOL,
        "protocol_sha256": "b" * 64,
        "symbol": symbol,
        "session_date": "2026-10-19",
        "sample_timestamp": iso(SAMPLE),
        "source_bar_timestamp": iso(SAMPLE - timedelta(minutes=1)),
        "reference_close": 100.0,
        "predicted_at": iso(PREDICTED),
        "prediction_lag_seconds": 30.0,
        "target_due_at": iso(DUE),
        "target_bar_timestamp": iso(DUE - timedelta(minutes=1)),
        "expected_magnitude_bps": 20.0,
        "linear_expected_magnitude_bps": 12.0,
        "null_expected_magnitude_bps": 10.0,
        "model_sha256": "a" * 64,
        "linear_model_sha256": "c" * 64,
        "feature_payload_sha256": "d" * 64,
    }
    return {**row, **overrides}


def resolution(pid="one", symbol="NVDA", **overrides):
    row = {
        "event_id": f"resolution:{pid}",
        "event_type": "resolution",
        "prediction_id": pid,
        "protocol_name": PROTOCOL,
        "symbol": symbol,
        "session_date": "2026-10-19",
        "sample_timestamp": iso(SAMPLE),
        "target_due_at": iso(DUE),
        "target_bar_timestamp": iso(DUE - timedelta(minutes=1)),
        "target_close": 100.15,
        "actual_magnitude_bps": 15.0,
        "absolute_error_bps": 5.0,
        "linear_absolute_error_bps": 3.0,
        "null_absolute_error_bps": 5.0,
        "resolved_at": iso(RESOLVED),
    }
    return {**row, **overrides}


class ForwardAnalyticsTests(unittest.TestCase):
    def test_missing_and_empty_are_not_misrepresented_as_predictions(self):
        with tempfile.TemporaryDirectory() as root:
            file = Path(root) / "missing.jsonl"
            report = read_forward_report(file, now_utc=NOW)
            self.assertFalse(report["journal_exists"])
            self.assertEqual(report["prediction_count"], 0)
            self.assertIsNone(report["champion_mae_bps"])
            self.assertIn("neexistuje", maybe_forward_answer("Aké sú tvoje posledné predikcie?", file))
            file.write_text("", encoding="utf-8")
            self.assertIn("0 predikcií", maybe_forward_answer("Aké sú tvoje posledné predikcie?", file))

    def test_pending_does_not_claim_resolved_accuracy_or_direction(self):
        with tempfile.TemporaryDirectory() as root:
            file = Path(root) / "events.jsonl"
            append_journal_events(file, [prediction()], recorded_at=PREDICTED)
            raw = file.read_bytes()
            report = read_forward_report(file, now_utc=NOW)
            self.assertEqual(report["prediction_count"], 1)
            self.assertEqual(report["pending_count"], 1)
            self.assertEqual(report["overdue_pending_count"], 1)
            self.assertIsNone(report["champion_mae_bps"])
            text = maybe_forward_answer("Aký pohyb očakávaš pri NVDA?", file)
            self.assertIn("20,00 bps", text)
            self.assertIn("0,20 %", text)
            self.assertIn("čaká", text)
            self.assertIn("nie", text.lower())
            self.assertNotIn("P(UP)", text)
            self.assertEqual(file.read_bytes(), raw)

    def test_resolved_errors_are_recomputed_and_ticker_filter_applies(self):
        with tempfile.TemporaryDirectory() as root:
            file = Path(root) / "events.jsonl"
            append_journal_events(file, [prediction(), prediction("AMD", "two")], recorded_at=PREDICTED)
            append_journal_events(file, [resolution()], recorded_at=RESOLVED)
            original = file.read_bytes()
            report = read_forward_report(file, now_utc=NOW)
            self.assertEqual(report["resolution_count"], 1)
            self.assertEqual(report["pending_count"], 1)
            self.assertEqual(report["champion_mae_bps"], 5.0)
            self.assertEqual(report["linear_mae_bps"], 3.0)
            self.assertEqual(report["null_mae_bps"], 5.0)
            self.assertEqual(report["symbols"]["NVDA"]["resolved"], 1)
            self.assertEqual(report["symbols"]["AMD"]["resolved"], 0)
            self.assertIn("lineárny baseline 3,00 bps", maybe_forward_answer("Ako presné boli predikcie NVDA?", file))
            self.assertIn("1 vyhodnotených", maybe_forward_answer("Ako presné boli naše posledné predikcie?", file))
            self.assertIn("Zatiaľ nemáme žiadne vyhodnotené", maybe_forward_answer("Aké chyby má AMD?", file))
            self.assertEqual(file.read_bytes(), original)
            self.assertIsNotNone(verify_journal(read_journal(file)))

    def test_fail_closed_on_tampering_and_semantic_mismatch(self):
        with tempfile.TemporaryDirectory() as root:
            file = Path(root) / "events.jsonl"
            append_journal_events(file, [prediction()], recorded_at=PREDICTED)
            data = file.read_bytes().replace(b'"symbol":"NVDA"', b'"symbol":"TSLA"')
            file.write_bytes(data)
            with self.assertRaisesRegex(ValueError, "hash"):
                read_forward_report(file, now_utc=NOW)
        with tempfile.TemporaryDirectory() as root:
            file = Path(root) / "events.jsonl"
            append_journal_events(file, [prediction()], recorded_at=PREDICTED)
            append_journal_events(
                file, [resolution(absolute_error_bps=999.0)], recorded_at=RESOLVED
            )
            with self.assertRaisesRegex(ValueError, "does not match forecast"):
                read_forward_report(file, now_utc=NOW)
        with tempfile.TemporaryDirectory() as root:
            file = Path(root) / "events.jsonl"
            append_journal_events(
                file, [prediction(sample_timestamp="2026-10-18T13:31:00Z")],
                recorded_at=PREDICTED,
            )
            with self.assertRaisesRegex(ValueError, "causal timing"):
                read_forward_report(file, now_utc=NOW)

    def test_rejects_other_protocol_and_resolution_symbol_mismatch(self):
        with tempfile.TemporaryDirectory() as root:
            file = Path(root) / "events.jsonl"
            append_journal_events(
                file, [prediction(protocol_name="v1")], recorded_at=PREDICTED
            )
            with self.assertRaisesRegex(ValueError, "another protocol"):
                read_forward_report(file, now_utc=NOW)
        with tempfile.TemporaryDirectory() as root:
            file = Path(root) / "events.jsonl"
            append_journal_events(file, [prediction()], recorded_at=PREDICTED)
            append_journal_events(
                file, [resolution(symbol="TSLA")], recorded_at=RESOLVED
            )
            with self.assertRaisesRegex(ValueError, "does not match prediction"):
                read_forward_report(file, now_utc=NOW)

    def test_resolutions_cannot_predate_the_due_time(self):
        with tempfile.TemporaryDirectory() as root:
            file = Path(root) / "events.jsonl"
            append_journal_events(file, [prediction()], recorded_at=PREDICTED)
            early = DUE - timedelta(seconds=1)
            append_journal_events(
                file, [resolution(resolved_at=iso(early))], recorded_at=early
            )
            with self.assertRaisesRegex(ValueError, "before its target"):
                read_forward_report(file, now_utc=NOW)

    def test_conservative_question_routing(self):
        for query in (
            "Čo znamená volatilita?",
            "Ahoj",
            "Ako sa máš?",
            "Čo je preučenie modelu pri akciách?",
            "Vysvetli fundamentálnu analýzu",
        ):
            self.assertIsNone(forward_question_kind(query))
        self.assertEqual(forward_question_kind("Ako presné boli tvoje predikcie?"), "performance")
        self.assertEqual(forward_question_kind("Aký 15-minútový pohyb očakávaš pre TSLA?"), "ticker")
        self.assertEqual(forward_question_kind("Aká je posledná predikcia?"), "latest")

    def test_ollama_and_hf_use_same_readonly_path_without_model_or_network(self):
        with tempfile.TemporaryDirectory() as root:
            file = Path(root) / "events.jsonl"
            append_journal_events(file, [prediction()], recorded_at=PREDICTED)
            with patch("ourai_language.ollama_runtime.urlopen") as mocked:
                ollama = LocalOllamaChatRuntime(forward_journal=file)
                actual = ollama.respond("Aký pohyb očakávaš pri NVDA?")
                mocked.assert_not_called()
            hf = object.__new__(OurAIChatRuntime)
            hf.knowledge_enabled = True
            hf.forward_journal = file
            alternate = hf.respond("Aký pohyb očakávaš pri NVDA?")
            self.assertEqual(actual.text, alternate.text)
            self.assertEqual(actual.answer_source, "verified_forward_journal")
            self.assertEqual(alternate.answer_source, "verified_forward_journal")

    def test_corrupted_journal_blocks_chat_instead_of_guessing(self):
        with tempfile.TemporaryDirectory() as root:
            file = Path(root) / "events.jsonl"
            append_journal_events(file, [prediction()], recorded_at=PREDICTED)
            with file.open("ab") as handle:
                handle.write(b'{"incomplete"')
            with patch("ourai_language.ollama_runtime.urlopen") as mocked:
                runtime = LocalOllamaChatRuntime(forward_journal=file)
                with self.assertRaisesRegex(RuntimeError, "neplatný"):
                    runtime.respond("Aké sú posledné predikcie NVDA?")
                mocked.assert_not_called()


if __name__ == "__main__":
    unittest.main()
