"""Fail-closed ticker-switch provenance and multi-turn Slovak regressions.

All journal responses are synthetic; no local Qwen benchmark is performed.
"""
from __future__ import annotations

import unittest
from unittest.mock import patch

from ourai_language.forward_followups import forward_ticker_followup_question
from ourai_language.ollama_runtime import LocalOllamaChatRuntime


def _history(user: str, evidence: str, source: str = "verified_forward_journal"):
    return [
        {"role": "user", "content": user},
        {"role": "assistant", "content": "Syntetické údaje.",
         "source": source, "evidence_question": evidence},
    ]


class ForwardTickerSwitchProvenanceTests(unittest.TestCase):
    def test_rejects_mismatched_unsupported_and_multi_ticker_provenance(self):
        question = "A čo TSLA?"
        cases = (
            # An authoritative-looking tag cannot excuse a wrong source ticker.
            ("Aké máš výsledky shadow pre NVDA?",
             "Aká je presnosť predikcií TSLA vo forward denníku?"),
            ("Aká je posledná predikcia NVDA a TSLA?",
             "Aká je posledná predikcia NVDA vo forward denníku?"),
            ("Aká je posledná predikcia NVDA a ibm?",
             "Aká je posledná predikcia NVDA vo forward denníku?"),
            ("Aké máš výsledky shadow pre NVDA?",
             "Aká je presnosť predikcií NVDA a TSLA vo forward denníku?"),
            ("Aké máš výsledky shadow pre NVDA?",
             "Aká je presnosť predikcií NVDA a ibm vo forward denníku?"),
            # An overall report cannot silently drop the named user ticker.
            ("Aké máš výsledky shadow pre NVDA?",
             "Aké sú výsledky vo forward denníku?"),
        )
        for user, evidence in cases:
            with self.subTest(user=user, evidence=evidence):
                self.assertIsNone(
                    forward_ticker_followup_question(
                        question, _history(user, evidence),
                    ),
                )

    def test_preserves_verified_overall_and_single_ticker_switches(self):
        self.assertEqual(
            forward_ticker_followup_question(
                "A čo TSLA?",
                _history(
                    "Ako sa ti darí v shadow režime?",
                    "Aké sú výsledky vo forward denníku?",
                ),
            ),
            "Aká je presnosť predikcií TSLA vo forward denníku?",
        )
        self.assertEqual(
            forward_ticker_followup_question(
                "A čo pre Teslu?",
                _history(
                    "Aká je posledná predikcia NVDA vo forward denníku?",
                    "Aká je posledná predikcia NVDA vo forward denníku?",
                ),
            ),
            "Aká je posledná predikcia TSLA vo forward denníku?",
        )
        # The previous user may have asked a short explanation; the
        # canonical verified evidence_question still anchors the source.
        self.assertEqual(
            forward_ticker_followup_question(
                "A čo TSLA?",
                _history(
                    "Čo táto predikcia znamená?",
                    "Aká je posledná predikcia NVDA vo forward denníku?",
                ),
            ),
            "Aká je posledná predikcia TSLA vo forward denníku?",
        )

    def test_runtime_chains_explanation_switch_and_accuracy_without_stale_values(self):
        runtime = LocalOllamaChatRuntime(
            forward_journal="synthetic-journal", knowledge_enabled=False,
            semantic_tool_routing=True,
        )
        nvda = "Aká je posledná predikcia NVDA vo forward denníku?"
        tsla = "Aká je posledná predikcia TSLA vo forward denníku?"
        accuracy = "Aká je presnosť predikcií TSLA vo forward denníku?"
        calls = []

        def reader(question, journal):
            self.assertEqual(journal, "synthetic-journal")
            calls.append(question)
            if question in (nvda, tsla, accuracy):
                return "Syntetický reader: " + question
            return None

        history = []
        turns = (
            nvda,
            "Čo táto predikcia znamená?",
            "A čo TSLA?",
            "A aká bola jej presnosť?",
        )
        with patch(
            "ourai_language.ollama_runtime.maybe_forward_answer",
            side_effect=reader,
        ), patch.object(
            runtime, "_generate",
            side_effect=AssertionError("Qwen must not run"),
        ):
            for user in turns:
                response = runtime.respond(user, history=history)
                self.assertEqual(response.answer_source, "verified_forward_journal")
                history.extend((
                    {"role": "user", "content": user},
                    {"role": "assistant", "content": response.text,
                     "source": response.answer_source,
                     "evidence_question": response.evidence_question},
                ))
        self.assertEqual(calls, [nvda, nvda, "A čo TSLA?", tsla, accuracy])
        self.assertEqual(history[-1]["evidence_question"], accuracy)
        self.assertIn("Syntetický reader: " + accuracy, history[-1]["content"])


if __name__ == "__main__":
    unittest.main()
