"""Synthetic language-unit regressions; no Qwen, journal, or network required."""
import unittest
from unittest.mock import patch

from ourai_language.numeric_units import explicit_bps_conversion_answer
from ourai_language.ollama_runtime import LocalOllamaChatRuntime


class BasisPointUnitTests(unittest.TestCase):
    def test_exact_slovak_unit_requests(self):
        for question in (
            "Koľko percent je 17 bps?",
            "Koľko percentuálnych bodov predstavuje 17 bps?",
            "Preveď 17 bps na percentá.",
            "kolko percent je 17 bazickych bodov?",
        ):
            with self.subTest(question=question):
                reply = explicit_bps_conversion_answer(question)
                self.assertIsNotNone(reply)
                self.assertIn("17 bps = 0,17", reply)
                self.assertNotIn("0,017", reply)

    def test_decimal_bps_and_zero(self):
        self.assertIn("0,125", explicit_bps_conversion_answer(
            "Koľko percent je 12,5 bps?"
        ))
        self.assertIn("0 bps = 0", explicit_bps_conversion_answer(
            "Preveď 0 bps na percentá"
        ))

    def test_does_not_hijack_predictions_or_open_ended_questions(self):
        questions = (
            "Ak má model MAE 17 bps, znamená to 83 % úspešnosť?",
            "Koľko percent zarobila OurAI pri MAE 17 bps?",
            "Aké sú tvoje forward výsledky pri 17 bps?",
            "Preveď 17 bps na percentá pre NVDA a vyhodnoť zisk.",
            "Čo znamená MAE 17 bps?",
            "Koľko percent je -17 bps?",
            "Koľko percent je 17 bps a kúp NVDA",
        )
        for question in questions:
            with self.subTest(question=question):
                self.assertIsNone(explicit_bps_conversion_answer(question))

    def test_ollama_runtime_does_not_generate_for_bare_conversion(self):
        runtime = LocalOllamaChatRuntime(knowledge_enabled=False)
        with patch.object(runtime, "_generate", side_effect=AssertionError("LLM called")):
            answer = runtime.respond("Koľko percent je 17 bps?")
        self.assertIn("0,17", answer.text)
        self.assertFalse(answer.used_grounding_fallback)


if __name__ == "__main__":
    unittest.main()
