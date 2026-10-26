"""Slovak error-size phrasing must use verified forward MAE, not a latest record.

Synthetic report only. No journal files, model inference or Qwen benchmark.
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
        # The existing reader accesses this map before choosing the metric.
        "latest_by_symbol": {"NVDA": {}},
    }


class ColloquialForwardErrorTests(unittest.TestCase):
    def test_explicit_forward_error_size_is_performance(self):
        questions = (
            "Ako veľmi sa tvoje forward odhady mýlia?",
            "Ako veľmi sa tvoje odhady mýlia v shadow režime?",
            "Ako ďaleko boli predikcie vo forward denníku od skutočnosti?",
            "O koľko sa tvoje forward predikcie líšili od reality?",
            "Aká bola odchýlka forward predikcií NVDA?",
            "Aka bola odchylka predikcii NVDA v shadow rezime?",
        )
        for question in questions:
            with self.subTest(question=question):
                self.assertEqual(forward_question_kind(question), "performance")

    def test_ambiguous_retrospective_and_causal_questions_are_not_forward(self):
        questions = (
            "Ako veľmi sa tvoje odhady mýlia?",
            "Ako ďaleko boli predikcie od skutočnosti?",
            "Aká bola odchýlka historických predikcií NVDA?",
            "Ako veľmi sa tvoje odhady mýlia v lockboxe?",
            "Prečo sa tvoje forward predikcie mýlia?",
            "Z akého dôvodu majú forward predikcie chyby?",
            "Čo spôsobilo odchýlku forward predikcií?",
            "Čo znamená MAE vo forward denníku?",
            "Koľko stojí NVDA?",
        )
        with patch(
            "ourai_language.forward_answers.read_forward_report",
            side_effect=AssertionError("must not read forward journal"),
        ):
            for question in questions:
                with self.subTest(question=question):
                    self.assertIsNone(forward_question_kind(question))
                    self.assertIsNone(maybe_forward_answer(question, "synthetic"))

    def test_existing_verified_reader_supplies_aggregate_mae_not_latest_record(self):
        question = "Ako ďaleko boli predikcie vo forward denníku od skutočnosti?"
        with patch(
            "ourai_language.forward_answers.read_forward_report",
            return_value=_synthetic_report(),
        ) as reader:
            answer = maybe_forward_answer(question, "synthetic")
        reader.assert_called_once_with("synthetic")
        self.assertIn("Celková MAE magnitude:", answer)
        self.assertIn("12,50 bps", answer)
        self.assertNotIn("Posledný záznam:", answer)
        self.assertIn("neznamenajú ziskovosť", answer)

    def test_named_ticker_uses_only_its_aggregate_forward_error(self):
        question = "Aká bola odchýlka forward predikcií NVDA?"
        with patch(
            "ourai_language.forward_answers.read_forward_report",
            return_value=_synthetic_report(),
        ):
            answer = maybe_forward_answer(question, "synthetic")
        self.assertTrue(answer.startswith("NVDA: 3 predikcií"))
        self.assertIn("MAE magnitude:", answer)
        self.assertNotIn("Najnovšia zaznamenaná predikcia", answer)

    def test_runtime_reuses_reader_without_calling_local_qwen(self):
        runtime = LocalOllamaChatRuntime(
            forward_journal="synthetic", knowledge_enabled=False,
            semantic_tool_routing=True,
        )
        question = "Ako veľmi sa tvoje forward odhady mýlia?"
        with patch(
            "ourai_language.forward_answers.read_forward_report",
            return_value=_synthetic_report(),
        ), patch.object(
            runtime, "_generate",
            side_effect=AssertionError("local model must not run"),
        ):
            result = runtime.respond(question)
        self.assertEqual(result.answer_source, "verified_forward_journal")
        self.assertEqual(result.evidence_question, question)
        self.assertIn("MAE magnitude:", result.text)


if __name__ == "__main__":
    unittest.main()
