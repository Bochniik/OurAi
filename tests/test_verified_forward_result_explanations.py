"""Source-bound forward MAE explanations, using synthetic readers only."""
from __future__ import annotations

import unittest
from unittest.mock import patch

from ourai_language.forward_followups import (
    forward_results_explanation_followup_question,
)
from ourai_language.ollama_runtime import LocalOllamaChatRuntime


def _history(
    question: str = "Aké máš výsledky shadow pre NVDA?",
    *, source: str = "verified_forward_journal",
    evidence: str | None = None,
):
    return [
        {"role": "user", "content": question},
        {"role": "assistant", "content": "Syntetické výsledky.",
         "source": source,
         "evidence_question": question if evidence is None else evidence},
    ]


class VerifiedForwardExplanationTests(unittest.TestCase):
    def test_natural_slovak_explanations_reread_same_ticker(self):
        expected = "Aká je presnosť predikcií NVDA vo forward denníku?"
        for prompt in (
            "Čo tieto výsledky znamenajú?",
            "A čo tieto výsledky znamenajú?",
            "Čo znamenajú tieto výsledky?",
            "Vysvetli mi tieto výsledky.",
            "A vysvetli mi tieto výsledky!",
            "Co tieto vysledky znamenaju?",
        ):
            with self.subTest(prompt=prompt):
                self.assertEqual(
                    forward_results_explanation_followup_question(
                        prompt, _history(),
                    ), expected,
                )

    def test_rejects_wrong_source_metric_or_topic(self):
        prompt = "Čo tieto výsledky znamenajú?"
        histories = (
            [], _history()[:1],
            _history(source="model"),
            _history(source="historical_v3_reports"),
            _history(evidence=""),
            _history(evidence="Ako sa ti darí v shadow režime?"),
            _history(question="Aká je posledná predikcia NVDA vo forward denníku?"),
            _history(question="Aké máš výsledky shadow pre NVDA a TSLA?"),
            _history(question="Aké máš výsledky shadow pre NVDA a ibm?"),
            _history(question="Aké máš výsledky shadow pre NVDA?",
                     evidence="Aká je presnosť predikcií TSLA vo forward denníku?"),
            _history() + [{"role": "user", "content": "Nová téma."}],
        )
        for history in histories:
            with self.subTest(history=history):
                self.assertIsNone(
                    forward_results_explanation_followup_question(prompt, history),
                )
        for prompt in (
            "Prečo je chyba taká veľká?",
            "Čo znamená MAE?",
            "Vysvetli mi historické výsledky.",
            "Čo tieto výsledky znamenajú pre nákup TSLA?",
            "A čo tieto výsledky znamenajú pre ziskovosť?",
            "Čo tieto výsledky znamenajú pre NVDA a TSLA?",
        ):
            with self.subTest(prompt=prompt):
                self.assertIsNone(
                    forward_results_explanation_followup_question(prompt, _history()),
                )

    def test_runtime_reloads_verified_metrics_without_qwen(self):
        runtime = LocalOllamaChatRuntime(
            forward_journal="synthetic-journal", knowledge_enabled=False,
            semantic_tool_routing=True,
        )
        original = "Aké máš výsledky shadow pre NVDA?"
        canonical = "Aká je presnosť predikcií NVDA vo forward denníku?"
        followup = "Čo tieto výsledky znamenajú?"
        calls = []

        def reader(question, journal):
            self.assertEqual(journal, "synthetic-journal")
            calls.append(question)
            if question == original:
                return "Prvé syntetické údaje."
            if question == canonical:
                return "Nové overené údaje."
            return None

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

        self.assertEqual(calls, [original, followup, canonical])
        self.assertEqual(second.answer_source, "verified_forward_journal")
        self.assertEqual(second.evidence_question, canonical)
        self.assertIn("Nové overené údaje.", second.text)
        self.assertNotIn("Prvé syntetické údaje.", second.text)
        self.assertIn("priemerná absolútna chyba", second.text)
        self.assertIn("nie je to presnosť smeru", second.text.lower())


if __name__ == "__main__":
    unittest.main()
