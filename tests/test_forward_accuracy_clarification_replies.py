"""Verified forward accuracy clarification replies; synthetic evidence only.

No local Qwen inference, benchmark, model weights or journal writes.
"""
from __future__ import annotations

import unittest
from unittest.mock import patch

from ourai_language.forward_answers import forward_question_kind
from ourai_language.forward_followups import (
    forward_accuracy_clarification_reply_question,
)
from ourai_language.ollama_runtime import LocalOllamaChatRuntime


QUESTION = "Aká je úspešnosť forward predikcií NVDA?"
CLARIFICATION = (
    "Otázka, ako často boli predikcie správne, je nejednoznačná. "
    "Myslíš úspešnosť odhadu smeru alebo MAE?"
)
MAE = "Aká je MAE predikcií NVDA vo forward denníku?"
DIRECTION = "Aká je smerová presnosť forward predikcií NVDA?"


def _history(
    question: str = QUESTION, *, evidence: str | None = None,
    source: str = "verified_forward_journal",
    answer: str = CLARIFICATION,
):
    return [
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer, "source": source,
         "evidence_question": question if evidence is None else evidence},
    ]


class ForwardAccuracyClarificationRepliesTests(unittest.TestCase):
    def test_bounded_slovak_replies_select_correct_existing_metric(self):
        for reply in ("Myslím MAE", "Zaujíma ma MAE.", "MAE", "Myslim MAE"):
            with self.subTest(reply=reply):
                self.assertEqual(
                    forward_accuracy_clarification_reply_question(reply, _history()),
                    MAE,
                )
        for reply in (
            "Myslím smer", "Zaujíma ma smer", "Smerovú úspešnosť",
            "Smer rastu alebo poklesu",
        ):
            with self.subTest(reply=reply):
                self.assertEqual(
                    forward_accuracy_clarification_reply_question(reply, _history()),
                    DIRECTION,
                )
        self.assertEqual(forward_question_kind(MAE), "performance")
        self.assertEqual(forward_question_kind(DIRECTION), "direction_unavailable")

    def test_overall_clarification_does_not_invent_a_ticker(self):
        history = _history("Ako často si sa trafil vo forward denníku?")
        self.assertEqual(
            forward_accuracy_clarification_reply_question("Myslím MAE", history),
            "Aká je MAE predikcií vo forward denníku?",
        )
        self.assertEqual(
            forward_accuracy_clarification_reply_question("Myslím smer", history),
            "Aká je smerová presnosť forward predikcií?",
        )

    def test_rejects_stale_unverified_or_mismatched_provenance(self):
        for history in (
            [],
            _history()[:1],
            _history(source="historical_v3_reports"),
            _history(source="model"),
            _history(evidence=""),
            _history(evidence="Aká je úspešnosť forward predikcií TSLA?"),
            _history(evidence="Aká je posledná predikcia NVDA vo forward denníku?"),
            _history(answer="Forward denník neexistuje."),
            _history(answer="Forward denník má 0 predikcií."),
            _history(question="Aká je úspešnosť historických predikcií NVDA?"),
            _history(question="Aká je úspešnosť forward predikcií NVDA a TSLA?"),
            _history(question="Aká je úspešnosť forward predikcií pre IBM?"),
            _history(question="Koľko stojí NVDA?"),
            _history() + [{"role": "user", "content": "Iná téma."}],
        ):
            with self.subTest(history=history):
                self.assertIsNone(
                    forward_accuracy_clarification_reply_question(
                        "Myslím MAE", history
                    )
                )

    def test_rejects_new_topic_negation_or_mixed_metric_reply(self):
        for reply in (
            "Myslím MAE historického lockboxu",
            "Myslím MAE pre TSLA",
            "Nie MAE, ale smer",
            "Myslím smer aj MAE",
            "Myslím MAE, kúp NVDA",
            "Čo je MAE?",
            "Ako často si sa trafil?",
        ):
            with self.subTest(reply=reply):
                self.assertIsNone(
                    forward_accuracy_clarification_reply_question(reply, _history())
                )

    def test_runtime_rereads_reader_without_qwen_and_records_provenance(self):
        runtime = LocalOllamaChatRuntime(
            forward_journal="synthetic", knowledge_enabled=False,
            semantic_tool_routing=True,
        )
        calls = []

        def fake_reader(question, journal):
            self.assertEqual(journal, "synthetic")
            calls.append(question)
            if question == QUESTION:
                return CLARIFICATION
            if question == MAE:
                return "Overená syntetická agregovaná MAE."
            if question == DIRECTION:
                return "Smerová úspešnosť nie je dostupná."
            raise AssertionError(f"Unexpected journal question: {question}")

        with patch(
            "ourai_language.ollama_runtime.maybe_forward_answer",
            side_effect=fake_reader,
        ), patch.object(
            runtime, "_generate",
            side_effect=AssertionError("Qwen must not run"),
        ):
            first = runtime.respond(QUESTION)
            history = _history(
                evidence=first.evidence_question,
                source=first.answer_source,
                answer=first.text,
            )
            mae = runtime.respond("Myslím MAE", history=history)
            direction = runtime.respond("Zaujíma ma smer", history=history)

        self.assertEqual(calls, [QUESTION, MAE, DIRECTION])
        self.assertEqual(mae.answer_source, "verified_forward_journal")
        self.assertEqual(mae.evidence_question, MAE)
        self.assertEqual(mae.text, "Overená syntetická agregovaná MAE.")
        self.assertEqual(direction.evidence_question, DIRECTION)
        self.assertNotIn("MAE", direction.text)

    def test_invalid_forward_journal_still_fails_closed(self):
        runtime = LocalOllamaChatRuntime(
            forward_journal="synthetic", knowledge_enabled=False,
        )
        with patch(
            "ourai_language.ollama_runtime.maybe_forward_answer",
            side_effect=ValueError("invalid synthetic chain"),
        ), patch.object(
            runtime, "_generate",
            side_effect=AssertionError("Qwen must not run"),
        ):
            with self.assertRaisesRegex(RuntimeError, "Forward denník"):
                runtime.respond("Myslím MAE", history=_history())


if __name__ == "__main__":
    unittest.main()
