"""Ambiguous Slovak correctness questions must not be converted into MAE.

Synthetic journal data only; no local Qwen inference or benchmark.
"""
from __future__ import annotations

import unittest
from unittest.mock import patch

from ourai_language.forward_answers import forward_question_kind, maybe_forward_answer
from ourai_language.ollama_runtime import LocalOllamaChatRuntime


class ForwardAmbiguousAccuracyTests(unittest.TestCase):
    def test_explicit_forward_correctness_questions_request_clarification(self):
        questions = (
            "Ako často si sa trafil vo forward denníku?",
            "Koľkokrát boli tvoje predikcie správne v shadow režime?",
            "Aká je úspešnosť forward predikcií NVDA?",
            "Ako úspešné boli tvoje forward predikcie?",
            "Kolkokrat boli tvoje predikcie spravne vo forward denniku?",
        )
        for question in questions:
            with self.subTest(question=question):
                self.assertEqual(
                    forward_question_kind(question), "accuracy_clarification"
                )

    def test_unanchored_historical_and_educational_requests_do_not_select_forward(self):
        questions = (
            "Ako často si sa trafil?",
            "Koľkokrát boli tvoje predikcie správne?",
            "Aká je úspešnosť historických predikcií?",
            "Koľkokrát boli tvoje predikcie správne v lockboxe?",
            "Koľkokrát boli tvoje predikcie správne pri backteste vo forward denníku?",
            "Čo znamená úspešnosť forward predikcií?",
        )
        with patch(
            "ourai_language.forward_answers.read_forward_report",
            side_effect=AssertionError("must not read forward journal"),
        ):
            for question in questions:
                with self.subTest(question=question):
                    self.assertIsNone(forward_question_kind(question))
                    self.assertIsNone(maybe_forward_answer(question, "synthetic"))

    def test_explicit_direction_and_magnitude_remain_distinct(self):
        self.assertEqual(
            forward_question_kind(
                "Ako často si správne predpovedal smer v shadow režime?"
            ),
            "direction_unavailable",
        )
        self.assertEqual(
            forward_question_kind("Aká bola presnosť forward predikcií NVDA?"),
            "performance",
        )
        self.assertEqual(
            forward_question_kind("Aká bola MAE vo forward denníku?"),
            "performance",
        )

    def test_verified_reader_only_returns_clarification_not_invented_success(self):
        report = {"journal_exists": True, "prediction_count": 2}
        question = "Koľkokrát boli tvoje predikcie správne v shadow režime?"
        with patch(
            "ourai_language.forward_answers.read_forward_report",
            return_value=report,
        ) as reader:
            answer = maybe_forward_answer(question, "synthetic")
        reader.assert_called_once_with("synthetic")
        self.assertIn("nejednoznačná", answer)
        self.assertIn("smeru rastu/poklesu", answer)
        self.assertIn("MAE", answer)
        self.assertIn("Bez upresnenia", answer)
        self.assertNotIn("2 predikci", answer)
        self.assertNotIn("percento úspešnosti: ", answer)

    def test_corrupt_journal_fails_closed(self):
        with patch(
            "ourai_language.forward_answers.read_forward_report",
            side_effect=ValueError("invalid synthetic chain"),
        ):
            with self.assertRaises(ValueError):
                maybe_forward_answer(
                    "Ako často si sa trafil vo forward denníku?", "synthetic"
                )

    def test_runtime_clarifies_without_generating_qwen_numbers(self):
        runtime = LocalOllamaChatRuntime(
            forward_journal="synthetic", knowledge_enabled=False,
            semantic_tool_routing=True,
        )
        question = "Ako často si sa trafil vo forward denníku?"
        with patch(
            "ourai_language.forward_answers.read_forward_report",
            return_value={"journal_exists": True, "prediction_count": 2},
        ), patch.object(
            runtime, "_generate",
            side_effect=AssertionError("local Qwen must not run"),
        ):
            result = runtime.respond(question)
        self.assertEqual(result.answer_source, "verified_forward_journal")
        self.assertEqual(result.evidence_question, question)
        self.assertIn("nejednoznačná", result.text)


if __name__ == "__main__":
    unittest.main()
