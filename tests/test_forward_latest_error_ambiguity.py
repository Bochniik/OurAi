"""Clarify ambiguous single-record error vs aggregate MAE, without Qwen.

All history and forward-reader values are synthetic. No local model, journal
file, lockbox, prediction model or Orion integration is accessed.
"""
from __future__ import annotations

import unittest
from unittest.mock import patch

from ourai_language.forward_followups import (
    forward_latest_error_ambiguity_clarification,
)
from ourai_language.ollama_runtime import LocalOllamaChatRuntime


LATEST = "Aká je posledná predikcia NVDA vo forward denníku?"


def history(
    question=LATEST, *, source="verified_forward_journal",
    evidence=LATEST, assistant="Syntetická posledná predikcia.",
):
    return [
        {"role": "user", "content": question},
        {"role": "assistant", "content": assistant, "source": source,
         "evidence_question": evidence},
    ]


class LatestErrorAmbiguityTests(unittest.TestCase):
    def test_short_real_slovak_phrasings_clarify_without_metrics(self):
        for question in (
            "A aká bola jej chyba?",
            "Aká bola jej chyba?",
            "A aká je jej chyba?",
            "A čo jej chyba?",
            "A aka bola jej chyba?",
            "A aká bola jej odchýlka?",
        ):
            with self.subTest(question=question):
                answer = forward_latest_error_ambiguity_clarification(
                    question, history(),
                )
                self.assertIsNotNone(answer)
                self.assertIn("NVDA", answer)
                self.assertIn("konkrétnej", answer)
                self.assertIn("MAE", answer)
                self.assertNotIn("17 bps", answer)

    def test_ticker_switch_must_be_verified_before_inheritance(self):
        answer = forward_latest_error_ambiguity_clarification(
            "A aká bola jej chyba?",
            history("A čo TSLA?", evidence=(
                "Aká je posledná predikcia TSLA vo forward denníku?"
            )),
        )
        self.assertIsNotNone(answer)
        self.assertIn("TSLA", answer)
        self.assertNotIn("NVDA", answer)

    def test_rejects_unverified_stale_conflicting_and_non_latest_context(self):
        for prior in (
            [],
            history()[:1],
            history(source="model"),
            history(source="historical_v3_reports"),
            history(evidence="Aké máš výsledky shadow pre NVDA?"),
            history(evidence="Aká je presnosť predikcií NVDA vo forward denníku?"),
            history(evidence="Aká je posledná predikcia TSLA vo forward denníku?"),
            history(evidence="Aká je posledná predikcia NVDA v lockboxe?"),
            history(question="Aká je posledná predikcia NVDA a TSLA?"),
            history(question="Aká je posledná predikcia NVDA a IBM?"),
            history(question="Aká bola cena NVDA?"),
            history() + [{"role": "user", "content": "Iná otázka"}],
        ):
            with self.subTest(prior=prior):
                self.assertIsNone(
                    forward_latest_error_ambiguity_clarification(
                        "A aká bola jej chyba?", prior,
                    )
                )

    def test_rejects_explicit_metrics_source_switch_and_trading(self):
        for question in (
            "A aká bola jej chyba v lockboxe?",
            "A aká bola jej chyba pri TSLA?",
            "A aká bola jej smerová chyba?",
            "A aká bola jej chyba a mám kúpiť NVDA?",
            "A aká bola jej MAE?",
            "A čo chyba poslednej predikcie a výnos?",
            "Prečo sa tá predikcia pomýlila?",
            "Čo znamená MAE?",
        ):
            with self.subTest(question=question):
                self.assertIsNone(
                    forward_latest_error_ambiguity_clarification(
                        question, history(),
                    )
                )

    def test_runtime_clarifies_without_reader_or_qwen_and_without_numbers(self):
        runtime = LocalOllamaChatRuntime(
            forward_journal="synthetic-journal", knowledge_enabled=False,
            semantic_tool_routing=True,
        )
        with patch(
            "ourai_language.ollama_runtime.maybe_forward_answer",
            return_value="Syntetický posledný záznam.",
        ) as reader, patch.object(
            runtime, "_generate",
            side_effect=AssertionError("Qwen must not run"),
        ):
            first = runtime.respond(LATEST)
            self.assertEqual(first.answer_source, "verified_forward_journal")
            prior = history(assistant=first.text)
            second = runtime.respond("A aká bola jej chyba?", history=prior)

        reader.assert_called_once_with(LATEST, "synthetic-journal")
        self.assertEqual(second.answer_source, "conversation_clarification")
        self.assertIsNone(second.evidence_question)
        self.assertIn("MAE", second.text)
        self.assertIn("NVDA", second.text)
        self.assertNotIn("Syntetický", second.text)


if __name__ == "__main__":
    unittest.main()
