"""Finite, deterministic finance glossary; not an LLM capability test."""
from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from ourai_language.contracts import AnalysisSnapshot
from ourai_language.financial_knowledge import (
    lookup_financial_knowledge, supported_topics,
)
from ourai_language.ollama_runtime import LocalOllamaChatRuntime
from ourai_language.runtime import OurAIChatRuntime
from scripts.benchmark_ourai_language import CASES


class _Response:
    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def read(self):
        return json.dumps({
            "done_reason": "stop", "message": {"role": "assistant", "content": "Odpovedal model."}
        }).encode("utf-8")


class FinancialKnowledgeTests(unittest.TestCase):
    def snapshot(self) -> AnalysisSnapshot:
        return AnalysisSnapshot(
            ticker="NVDA", company_name="NVIDIA Corporation",
            horizon_minutes=15, p_up=0.68, p_down=0.32,
            expected_move_pct=0.41,
            uncertainty="medium", decision="LONG_BIAS",
        )

    def test_seven_existing_finance_benchmark_questions_are_scoped_and_handled(self):
        expected = (
            "fundamental_technical", "liquidity_spread", "low_volume",
            "calibration", "accuracy_profit", "overfitting",
            "volatility_trend",
        )
        for (case_id, prompt), topic in zip(CASES[:7], expected):
            with self.subTest(case_id=case_id):
                answer = lookup_financial_knowledge(prompt)
                self.assertIsNotNone(answer)
                self.assertEqual(answer.topic, topic)
                self.assertGreater(len(answer.text), 100)
                self.assertTrue(all(x.startswith("https://") for x in answer.references))
        self.assertIsNone(lookup_financial_knowledge(CASES[7][1]))
        self.assertGreaterEqual(len(supported_topics()), 9)

    def test_concept_content_prevents_observed_llm_failures(self):
        answer = lookup_financial_knowledge("Čo je preučenie modelu pri predpovedaní pohybu akcií?")
        self.assertIn("tréningovým dátam", answer.text)
        self.assertIn("náhodného šumu", answer.text)
        self.assertIn("nových", answer.text)
        self.assertNotIn("znamená overenie", answer.text)

        profitable = lookup_financial_knowledge("Prečo model so 70 % úspešnosťou nemusí byť ziskový?")
        self.assertIn("70 − 90 = −20 €", profitable.text)
        self.assertIn("poplatky", profitable.text)

        calibrated = lookup_financial_knowledge("Čo znamená kalibrácia pravdepodobností predikčného modelu?")
        self.assertIn("70 %", calibrated.text)
        self.assertIn("oddelených", calibrated.text)

    def test_brier_and_slippage_glossary(self):
        self.assertIn("štvorcov", lookup_financial_knowledge("Čo je Brier score?").text)
        self.assertIn("skutočne dosiahnutou", lookup_financial_knowledge("Čo je slippage?").text)

    def test_unrelated_and_specific_questions_are_not_hijacked(self):
        for query in (
            "Aké je hlavné mesto Francúzska?",
            "Vysvetli mi prosím fundamentálne pravidlá varenia.",
            "Chcem spraviť graf.",
            "Porovnaj NVDA a TSLA dnes.",
            "Aká je cena MSFT dnes?",
            "Dokážeš garantovať, že NVDA zajtra porastie?",
            "Čo je overfitting tohto konkrétneho modelu NVDA?",
        ):
            with self.subTest(query=query):
                self.assertIsNone(lookup_financial_knowledge(query, analysis=self.snapshot()))
        self.assertIsNone(
            lookup_financial_knowledge(
                "Čo je overfitting pri NVDA?",
                analysis=self.snapshot(),
            )
        )

    def test_curated_concept_works_with_snapshot_but_ticker_specific_remains_grounded(self):
        snapshot = self.snapshot()
        with patch("ourai_language.ollama_runtime.urlopen") as urlopen:
            runtime = LocalOllamaChatRuntime()
            concept = runtime.respond(
                "Čo znamená kalibrácia pravdepodobností predikčného modelu?",
                analysis=snapshot,
            )
            self.assertEqual(concept.answer_source, "curated_knowledge")
            self.assertIn("dlhodobej početnosti", concept.text)
            specific = runtime.respond("Prečo si pri NVDA bullish?", analysis=snapshot)
            self.assertNotEqual(specific.answer_source, "curated_knowledge")
            self.assertIn("NVDA", specific.text)
            urlopen.assert_not_called()

    def test_both_backends_return_same_curated_explanation_without_model_loading(self):
        query = "Čo je to preučenie modelu pri predpovedaní pohybu akcií?"
        with patch("ourai_language.ollama_runtime.urlopen") as urlopen:
            ollama_answer = LocalOllamaChatRuntime().respond(query)
            urlopen.assert_not_called()
        hf = object.__new__(OurAIChatRuntime)
        hf.knowledge_enabled = True
        hf_answer = hf.respond(query)
        self.assertEqual(ollama_answer.text, hf_answer.text)
        self.assertEqual(hf_answer.answer_source, "curated_knowledge")
        self.assertFalse(hf_answer.used_grounding_fallback)

    def test_disable_curated_answers_for_raw_llm_comparison(self):
        with patch("ourai_language.ollama_runtime.urlopen", return_value=_Response()) as urlopen:
            runtime = LocalOllamaChatRuntime(knowledge_enabled=False)
            result = runtime.respond("Čo je to preučenie modelu pri predpovedaní pohybu akcií?")
            self.assertEqual(result.answer_source, "model")
            self.assertEqual(result.text, "Odpovedal model.")
            urlopen.assert_called_once()


if __name__ == "__main__":
    unittest.main()
