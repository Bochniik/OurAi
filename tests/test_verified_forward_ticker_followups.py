"""Source-bound forward ticker follow-ups; synthetic readers, no local Qwen."""
from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from ourai_language.forward_followups import forward_ticker_followup_question
from ourai_language.ollama_runtime import LocalOllamaChatRuntime


def _history(
    prior: str = "Aké máš výsledky shadow pre NVDA?",
    *, source: str = "verified_forward_journal",
    evidence: str | None = None,
):
    return [
        {"role": "user", "content": prior},
        {"role": "assistant", "content": "Syntetická overená odpoveď.",
         "source": source,
         "evidence_question": prior if evidence is None else evidence},
    ]


class VerifiedForwardTickerFollowupTests(unittest.TestCase):
    def test_preserves_verified_performance_and_latest_scope(self):
        for followup, ticker in (
            ("A čo TSLA?", "TSLA"),
            ("A čo pre Teslu?", "TSLA"),
            ("A pre Microsoft?", "MSFT"),
            ("Čo pre Nvidiu?", "NVDA"),
            ("A co pre AAPL?", "AAPL"),
        ):
            with self.subTest(followup=followup):
                self.assertEqual(
                    forward_ticker_followup_question(followup, _history()),
                    f"Aká je presnosť predikcií {ticker} vo forward denníku?",
                )
        last = _history("Aká je posledná predikcia NVDA vo forward denníku?")
        self.assertEqual(
            forward_ticker_followup_question("A čo TSLA?", last),
            "Aká je posledná predikcia TSLA vo forward denníku?",
        )

    def test_rejects_unverified_or_ambiguous_followups(self):
        history = _history()
        for question in (
            "A čo IBM?", "A čo pre Alphabet?", "A čo TSLA a NVDA?",
            "A čo TSLA alebo NVDA?", "A čo $IBM?",
            "A čo TSLA a obchoduj?", "Čo znamená TSLA?",
            "A čo výsledky lockboxu pre TSLA?",
            "A čo ak TSLA zajtra porastie?", "A čo?",
        ):
            with self.subTest(question=question):
                self.assertIsNone(
                    forward_ticker_followup_question(question, history),
                )

    def test_needs_immediately_previous_verified_forward_source_and_provenance(self):
        history = _history()
        self.assertIsNone(forward_ticker_followup_question("A čo TSLA?", []))
        self.assertIsNone(forward_ticker_followup_question(
            "A čo TSLA?", history[:1],
        ))
        self.assertIsNone(forward_ticker_followup_question(
            "A čo TSLA?", _history(source="historical_v3_reports"),
        ))
        self.assertIsNone(forward_ticker_followup_question(
            "A čo TSLA?", _history(source="model"),
        ))
        self.assertIsNone(forward_ticker_followup_question(
            "A čo TSLA?", [{**history[0]}, {**history[1], "evidence_question": ""}],
        ))
        self.assertIsNone(forward_ticker_followup_question(
            "A čo TSLA?", history + [{"role": "user", "content": "Iná téma."}],
        ))
        self.assertIsNone(forward_ticker_followup_question(
            "A čo TSLA?", _history(
                evidence="Porovnaj historickú validáciu v3 s lineárnym modelom.",
            ),
        ))
        self.assertIsNone(forward_ticker_followup_question(
            "A čo TSLA?", _history(
                prior="Čo je forward?", evidence="Čo je forward?",
            ),
        ))

    def test_runtime_rereads_journal_and_records_forward_provenance(self):
        runtime = LocalOllamaChatRuntime(
            forward_journal="synthetic-journal", knowledge_enabled=False,
            semantic_tool_routing=True,
        )
        def fake_reader(question, journal):
            self.assertEqual(journal, "synthetic-journal")
            if "NVDA" in question:
                return "Syntetická overená odpoveď NVDA."
            if "TSLA" in question:
                return "Syntetická overená odpoveď TSLA."
            return None
        with patch("ourai_language.ollama_runtime.maybe_forward_answer",
                   side_effect=fake_reader), patch.object(
                       runtime, "_generate",
                       side_effect=AssertionError("model must not run"),
                   ):
            original = "Aké máš výsledky shadow pre NVDA?"
            first = runtime.respond(original)
            self.assertEqual(first.answer_source, "verified_forward_journal")
            self.assertEqual(first.evidence_question, original)
            history = [
                {"role": "user", "content": original},
                {"role": "assistant", "content": first.text,
                 "source": first.answer_source,
                 "evidence_question": first.evidence_question},
            ]
            second = runtime.respond("A čo TSLA?", history=history)
        self.assertEqual(second.answer_source, "verified_forward_journal")
        self.assertEqual(second.text, "Syntetická overená odpoveď TSLA.")
        self.assertEqual(
            second.evidence_question,
            "Aká je presnosť predikcií TSLA vo forward denníku?",
        )

    def test_opt_in_selection_records_canonical_forward_provenance(self):
        runtime = LocalOllamaChatRuntime(
            forward_journal="synthetic-journal", knowledge_enabled=False,
            semantic_tool_routing=True,
        )
        proposed = json.dumps({
            "tool": "forward_v2", "action": "performance",
            "split": "both", "baseline": "both",
        })
        with patch("ourai_language.ollama_runtime.maybe_forward_answer",
                   side_effect=lambda q, _: "Overené syntetické dáta." if
                   "vo forward denníku" in q else None), patch.object(
                       runtime, "_generate", return_value=proposed,
                   ) as generate:
            initial = "Ako sa ti darí v shadow režime?"
            first = runtime.respond(initial)
            self.assertEqual(first.answer_source, "verified_forward_journal")
            self.assertIn("vo forward denníku", first.evidence_question)
            history = [
                {"role": "user", "content": initial},
                {"role": "assistant", "content": first.text,
                 "source": first.answer_source,
                 "evidence_question": first.evidence_question},
            ]
            second = runtime.respond("A čo TSLA?", history=history)
        self.assertEqual(generate.call_count, 1)
        self.assertEqual(second.answer_source, "verified_forward_journal")
        self.assertIn("TSLA", second.evidence_question)


if __name__ == "__main__":
    unittest.main()
