"""Bounded Slovak explanations of the latest verified forward record.

All evidence is synthetic; this suite does not run a local Qwen benchmark.
"""
from __future__ import annotations

import unittest
from unittest.mock import patch

from ourai_language.forward_followups import (
    forward_latest_explanation_followup_question,
)
from ourai_language.ollama_runtime import LocalOllamaChatRuntime


def _history(
    prior: str = "Aká je posledná predikcia NVDA vo forward denníku?",
    *, source: str = "verified_forward_journal",
    evidence: str | None = None,
):
    return [
        {"role": "user", "content": prior},
        {"role": "assistant", "content": "Syntetický záznam.",
         "source": source,
         "evidence_question": prior if evidence is None else evidence},
    ]


class VerifiedLatestExplanationTests(unittest.TestCase):
    def test_natural_slovak_phrasings_preserve_one_verified_ticker(self):
        expected = "Aká je posledná predikcia NVDA vo forward denníku?"
        for prompt in (
            "Čo táto predikcia znamená?",
            "Čo táto predikcia vlastne znamená?",
            "A čo táto predikcia znamená?",
            "Vysvetli mi túto predikciu.",
            "A vysvetli mi túto predikciu!",
            "Čo znamená tá posledná predikcia?",
            "Co tato predikcia vlastne znamena?",
        ):
            with self.subTest(prompt=prompt):
                self.assertEqual(
                    forward_latest_explanation_followup_question(
                        prompt, _history(),
                    ), expected,
                )
        self.assertEqual(
            forward_latest_explanation_followup_question(
                "Čo táto predikcia znamená?",
                _history("A čo pre Teslu?", evidence=(
                    "Aká je posledná predikcia TSLA vo forward denníku?"
                )),
            ),
            "Aká je posledná predikcia TSLA vo forward denníku?",
        )

    def test_requires_immediate_verified_latest_single_ticker_provenance(self):
        prompt = "Čo táto predikcia znamená?"
        histories = (
            [], _history()[:1],
            _history(source="model"),
            _history(source="historical_v3_reports"),
            _history(evidence=""),
            _history(evidence="Aká je presnosť predikcií NVDA vo forward denníku?"),
            _history(evidence="Aké sú predikcie NVDA vo forward denníku?"),
            _history(evidence="Aká je posledná predikcia vo forward denníku?"),
            _history(prior="Aká je posledná predikcia NVDA a TSLA?"),
            _history(prior="Aká je posledná predikcia NVDA a ibm?"),
            _history(prior="Aká je posledná predikcia NVDA?",
                     evidence="Aká je posledná predikcia TSLA vo forward denníku?"),
            _history() + [{"role": "user", "content": "Iná téma."}],
        )
        for history in histories:
            with self.subTest(history=history):
                self.assertIsNone(
                    forward_latest_explanation_followup_question(prompt, history),
                )

    def test_rejects_causal_historical_trading_and_ticker_switch_requests(self):
        for prompt in (
            "Prečo model očakáva taký pohyb?",
            "Prečo je táto predikcia nesprávna?",
            "Čo znamená posledná historická predikcia?",
            "Čo táto predikcia znamená pre TSLA?",
            "Čo táto predikcia znamená pre nákup?",
            "Čo táto predikcia znamená pre ziskovosť?",
            "Čo táto predikcia znamená v lockboxe?",
            "Čo táto predikcia znamená a mám kúpiť NVDA?",
            "Čo táto predikcia znamená pre NVDA a TSLA?",
        ):
            with self.subTest(prompt=prompt):
                self.assertIsNone(
                    forward_latest_explanation_followup_question(
                        prompt, _history(),
                    ),
                )

    def test_runtime_rereads_verified_latest_before_general_forward_routing(self):
        runtime = LocalOllamaChatRuntime(
            forward_journal="synthetic-journal", knowledge_enabled=False,
            semantic_tool_routing=True,
        )
        original = "Aká je posledná predikcia NVDA vo forward denníku?"
        followup = "Čo táto predikcia vlastne znamená?"
        calls = []

        def reader(question, journal):
            self.assertEqual(journal, "synthetic-journal")
            calls.append(question)
            if question == original:
                return "Prvý syntetický záznam."
            raise AssertionError(f"Unexpected journal question: {question}")

        with patch(
            "ourai_language.ollama_runtime.maybe_forward_answer",
            side_effect=reader,
        ), patch.object(
            runtime, "_generate",
            side_effect=AssertionError("Qwen must not run"),
        ):
            first = runtime.respond(original)
            history = [
                {"role": "user", "content": original},
                {"role": "assistant", "content": first.text,
                 "source": first.answer_source,
                 "evidence_question": first.evidence_question},
            ]
            second = runtime.respond(followup, history=history)

        self.assertEqual(calls, [original, original])
        self.assertEqual(second.answer_source, "verified_forward_journal")
        self.assertEqual(second.evidence_question, original)
        self.assertIn("Prvý syntetický záznam.", second.text)
        self.assertIn("absolútna 15-minútová zmena", second.text)
        self.assertIn("bez určenia smeru", second.text)
        self.assertIn("nie či predikcia bola presná", second.text)
        self.assertNotIn("presnosť smeru", second.text)


if __name__ == "__main__":
    unittest.main()
