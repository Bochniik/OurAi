"""Bounded latest-record follow-ups; synthetic journal reader, no local Qwen."""
from __future__ import annotations

import unittest
from unittest.mock import patch

from ourai_language.forward_followups import forward_latest_followup_question
from ourai_language.ollama_runtime import LocalOllamaChatRuntime


def _history(
    prior: str = "Aké máš výsledky shadow pre NVDA?",
    *, source: str = "verified_forward_journal",
    evidence: str | None = None,
):
    return [
        {"role": "user", "content": prior},
        {"role": "assistant", "content": "Syntetická odpoveď.",
         "source": source,
         "evidence_question": prior if evidence is None else evidence},
    ]


class VerifiedForwardLatestFollowupTests(unittest.TestCase):
    def test_realistic_slovak_latest_followups_preserve_only_verified_ticker(self):
        for question in (
            "A aká bola posledná predikcia?",
            "A aká je posledná predikcia?",
            "A čo posledná predikcia?",
            "A posledná predikcia?",
            "A aka bola posledna predikcia?",
        ):
            with self.subTest(question=question):
                self.assertEqual(
                    forward_latest_followup_question(question, _history()),
                    "Aká je posledná predikcia NVDA vo forward denníku?",
                )
        self.assertEqual(
            forward_latest_followup_question(
                "A aká bola posledná predikcia?",
                _history("A čo pre Teslu?", evidence=(
                    "Aká je presnosť predikcií TSLA vo forward denníku?"
                )),
            ),
            "Aká je posledná predikcia TSLA vo forward denníku?",
        )

    def test_requires_immediate_verified_forward_provenance_and_one_ticker(self):
        followup = "A aká bola posledná predikcia?"
        for history in (
            [],
            _history()[:1],
            _history(source="model"),
            _history(source="historical_v3_reports"),
            _history(evidence=""),
            _history(evidence="Ako sa ti darí v shadow režime?"),
            _history(evidence="Porovnaj historickú MAE s lineárnym modelom."),
            _history(prior="Aké výsledky shadow pre NVDA a TSLA?"),
            _history(prior="Aké výsledky shadow pre NVDA a ibm?"),
            _history(prior="Aké výsledky shadow pre NVDA?",
                     evidence="Aká je presnosť predikcií TSLA vo forward denníku?"),
            _history() + [{"role": "user", "content": "Iná téma."}],
        ):
            with self.subTest(history=history):
                self.assertIsNone(forward_latest_followup_question(followup, history))

    def test_rejects_ambiguous_source_or_stock_switches(self):
        for question in (
            "A aká bola posledná predikcia TSLA?",
            "A aká bola posledná predikcia v lockboxe?",
            "A aká bola posledná historická predikcia?",
            "A aká bola posledná predikcia pre NVDA a TSLA?",
            "A aká bola posledná predikcia a mám kúpiť?",
            "A čo bola posledná predikcia na zajtra?",
            "Čo znamená posledná predikcia?",
        ):
            with self.subTest(question=question):
                self.assertIsNone(
                    forward_latest_followup_question(question, _history()),
                )

    def test_runtime_rereads_same_ticker_instead_of_overall_latest(self):
        runtime = LocalOllamaChatRuntime(
            forward_journal="synthetic-journal", knowledge_enabled=False,
            semantic_tool_routing=True,
        )
        calls = []
        initial = "Aké máš výsledky shadow pre NVDA?"
        canonical = "Aká je posledná predikcia NVDA vo forward denníku?"

        def fake_reader(question, journal):
            self.assertEqual(journal, "synthetic-journal")
            calls.append(question)
            if question == initial:
                return "Syntetická presnosť NVDA."
            if question == canonical:
                return "Syntetický posledný záznam NVDA."
            raise AssertionError(f"Unexpected journal question: {question}")

        with patch("ourai_language.ollama_runtime.maybe_forward_answer",
                   side_effect=fake_reader), patch.object(
                       runtime, "_generate",
                       side_effect=AssertionError("local model must not run"),
                   ):
            first = runtime.respond(initial)
            history = [
                {"role": "user", "content": initial},
                {"role": "assistant", "content": first.text,
                 "source": first.answer_source,
                 "evidence_question": first.evidence_question},
            ]
            second = runtime.respond("A aká bola posledná predikcia?", history=history)

        self.assertEqual(calls, [initial, canonical])
        self.assertEqual(second.answer_source, "verified_forward_journal")
        self.assertEqual(second.evidence_question, canonical)
        self.assertEqual(second.text, "Syntetický posledný záznam NVDA.")


if __name__ == "__main__":
    unittest.main()
