"""Verified forward performance follow-ups; mocked readers, no local Qwen."""
from __future__ import annotations

import unittest
from unittest.mock import patch

from ourai_language.forward_followups import (
    forward_latest_followup_question, forward_performance_followup_question,
)
from ourai_language.ollama_runtime import LocalOllamaChatRuntime


def _history(
    prior: str = "Aká je posledná predikcia NVDA vo forward denníku?",
    *, source: str = "verified_forward_journal",
    evidence: str | None = None,
):
    return [
        {"role": "user", "content": prior},
        {"role": "assistant", "content": "Syntetická odpoveď.",
         "source": source,
         "evidence_question": prior if evidence is None else evidence},
    ]


class VerifiedForwardPerformanceFollowupTests(unittest.TestCase):
    def test_real_slovak_phrasings_keep_verified_ticker(self):
        canonical = "Aká je presnosť predikcií NVDA vo forward denníku?"
        for question in (
            "A aká bola jej presnosť?",
            "A aká je jej presnosť?",
            "A aká bola presnosť predikcií?",
            "A čo ich presnosť?",
            "A čo presnosť predikcií?",
            "A presnosť predikcií?",
            "A aka bola jej presnost?",
        ):
            with self.subTest(question=question):
                self.assertEqual(
                    forward_performance_followup_question(question, _history()),
                    canonical,
                )
        self.assertEqual(
            forward_performance_followup_question(
                "A aká bola jej presnosť?",
                _history("A čo TSLA?", evidence=(
                    "Aká je posledná predikcia TSLA vo forward denníku?"
                )),
            ),
            "Aká je presnosť predikcií TSLA vo forward denníku?",
        )

    def test_rejects_missing_mismatched_and_historical_provenance(self):
        question = "A aká bola jej presnosť?"
        for history in (
            [], _history()[:1],
            _history(source="model"),
            _history(source="historical_v3_reports"),
            _history(evidence=""),
            _history(evidence="Aký bol historický lockbox test v3?"),
            _history(evidence="Aké sú výsledky forward denníka?"),
            _history(prior="Aká je posledná predikcia NVDA a TSLA?"),
            _history(prior="Aká je posledná predikcia NVDA a ibm?"),
            _history(prior="Aká je posledná predikcia NVDA?",
                     evidence="Aká je posledná predikcia TSLA vo forward denníku?"),
            _history() + [{"role": "user", "content": "Zmeňme tému."}],
        ):
            with self.subTest(history=history):
                self.assertIsNone(
                    forward_performance_followup_question(question, history),
                )

    def test_rejects_trading_historical_and_stock_switch_requests(self):
        for question in (
            "A aká bola jej presnosť v lockboxe?",
            "A aká bola jej presnosť pre TSLA?",
            "A aká bola jej smerová presnosť?",
            "A aká bola jej presnosť a mám kúpiť NVDA?",
            "A aká bola jej presnosť na dnešných dátach?",
            "A aká bola presnosť predikcií IBM?",
            "A čo presnosť Nvidie oproti Tesle?",
            "A čo výnosnosť?",
        ):
            with self.subTest(question=question):
                self.assertIsNone(
                    forward_performance_followup_question(question, _history()),
                )

    def test_latest_followups_remain_unchanged(self):
        self.assertEqual(
            forward_latest_followup_question(
                "A aká bola posledná predikcia?", _history(),
            ),
            "Aká je posledná predikcia NVDA vo forward denníku?",
        )

    def test_runtime_rereads_verified_journal_without_model_or_stale_numbers(self):
        runtime = LocalOllamaChatRuntime(
            forward_journal="synthetic-journal", knowledge_enabled=False,
            semantic_tool_routing=True,
        )
        first_question = "Aká je posledná predikcia NVDA vo forward denníku?"
        canonical = "Aká je presnosť predikcií NVDA vo forward denníku?"
        calls = []

        def fake_reader(question, journal):
            self.assertEqual(journal, "synthetic-journal")
            calls.append(question)
            if question == first_question:
                return "Syntetická posledná predikcia."
            if question == canonical:
                return "Syntetické agregované MAE."
            raise AssertionError(f"Unexpected forward reader question: {question}")

        with patch(
            "ourai_language.ollama_runtime.maybe_forward_answer",
            side_effect=fake_reader,
        ), patch.object(
            runtime, "_generate",
            side_effect=AssertionError("local Qwen must not run"),
        ):
            first = runtime.respond(first_question)
            history = [
                {"role": "user", "content": first_question},
                {"role": "assistant", "content": first.text,
                 "source": first.answer_source,
                 "evidence_question": first.evidence_question},
            ]
            second = runtime.respond("A aká bola jej presnosť?", history=history)

        self.assertEqual(calls, [first_question, canonical])
        self.assertEqual(second.answer_source, "verified_forward_journal")
        self.assertEqual(second.evidence_question, canonical)
        self.assertEqual(second.text, "Syntetické agregované MAE.")


if __name__ == "__main__":
    unittest.main()
