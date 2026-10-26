"""Forward direction hit-rate questions must never be answered with magnitude MAE.

All evidence below is synthetic. No local Qwen inference or benchmark.
"""
from __future__ import annotations

import unittest
from unittest.mock import patch

from ourai_language.forward_answers import (
    forward_question_kind, maybe_forward_answer,
)
from ourai_language.ollama_runtime import LocalOllamaChatRuntime


def _synthetic_report():
    row = {
        "predictions": 3, "resolved": 2, "pending": 1,
        "overdue_pending": 0,
        "champion_mae_bps": 12.5,
        "linear_mae_bps": 13.0,
        "null_mae_bps": 14.0,
    }
    return {
        "journal_exists": True,
        "prediction_count": 3,
        "resolution_count": 2,
        "pending_count": 1,
        "overdue_pending_count": 0,
        "champion_mae_bps": 12.5,
        "linear_mae_bps": 13.0,
        "null_mae_bps": 14.0,
        "symbols": {"NVDA": row},
        "latest_by_symbol": {"NVDA": {}},
    }


class ForwardDirectionalAccuracyBoundaryTests(unittest.TestCase):
    def test_explicit_forward_direction_hit_rate_is_unavailable(self):
        questions = (
            "Aká bola smerová presnosť forward predikcií NVDA?",
            "Ako často si trafil smer predikcií vo forward denníku?",
            "Koľkokrát si správne predpovedal rast/pokles NVDA v shadow režime?",
            "Aka je smerova uspesnost forward predikcii?",
            "Ako casto si spravne predpovedal UP/DOWN v shadow rezime?",
        )
        for question in questions:
            with self.subTest(question=question):
                self.assertEqual(
                    forward_question_kind(question), "direction_unavailable",
                )

    def test_ambiguous_historical_and_educational_questions_do_not_select_forward(self):
        questions = (
            "Aká bola smerová presnosť predikcií NVDA?",
            "Aká bola smerová presnosť historických forward testov?",
            "Aká bola smerová presnosť v lockboxe?",
            "Čo znamená smerová presnosť vo forward denníku?",
            "Ako často si sa trafil?",
        )
        with patch(
            "ourai_language.forward_answers.read_forward_report",
            side_effect=AssertionError("must not read forward journal"),
        ):
            for question in questions:
                with self.subTest(question=question):
                    self.assertIsNone(forward_question_kind(question))
                    self.assertIsNone(maybe_forward_answer(question, "synthetic"))

    def test_verified_journal_does_not_turn_mae_into_directional_accuracy(self):
        question = "Aká bola smerová presnosť forward predikcií NVDA?"
        with patch(
            "ourai_language.forward_answers.read_forward_report",
            return_value=_synthetic_report(),
        ) as reader:
            answer = maybe_forward_answer(question, "synthetic")
        reader.assert_called_once_with("synthetic")
        self.assertIn("Smerovú úspešnosť", answer)
        self.assertIn("nie smer", answer)
        self.assertIn("MAE meria chybu veľkosti", answer)
        self.assertNotIn("12,50", answer)
        self.assertNotIn("3 predikcií", answer)

    def test_corrupt_journal_fails_closed(self):
        with patch(
            "ourai_language.forward_answers.read_forward_report",
            side_effect=ValueError("invalid synthetic chain"),
        ):
            with self.assertRaises(ValueError):
                maybe_forward_answer(
                    "Aká bola smerová presnosť forward predikcií NVDA?",
                    "synthetic",
                )

    def test_magnitude_mae_requests_remain_unchanged(self):
        for question in (
            "Aká bola presnosť forward predikcií NVDA?",
            "Ako veľmi sa tvoje forward odhady mýlia?",
        ):
            with self.subTest(question=question):
                self.assertEqual(forward_question_kind(question), "performance")

    def test_runtime_returns_bounded_answer_without_local_model(self):
        runtime = LocalOllamaChatRuntime(
            forward_journal="synthetic", knowledge_enabled=False,
            semantic_tool_routing=True,
        )
        question = "Aká bola smerová presnosť forward predikcií NVDA?"
        with patch(
            "ourai_language.forward_answers.read_forward_report",
            return_value=_synthetic_report(),
        ), patch.object(
            runtime, "_generate",
            side_effect=AssertionError("local Qwen must not run"),
        ):
            result = runtime.respond(question)
        self.assertEqual(result.answer_source, "verified_forward_journal")
        self.assertEqual(result.evidence_question, question)
        self.assertIn("Smerovú úspešnosť", result.text)
        self.assertNotIn("12,50", result.text)


if __name__ == "__main__":
    unittest.main()
