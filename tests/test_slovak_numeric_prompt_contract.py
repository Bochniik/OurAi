"""Regression for the local conversational prompt's Slovak and unit guidance.

These assertions validate the prompt contract, not model accuracy or a Qwen benchmark.
"""
import unittest

from ourai_language.prompting import (
    ANALYSIS_QA,
    GENERAL_CHAT,
    build_messages,
)


class SlovakNumericPromptTests(unittest.TestCase):
    def test_general_chat_contains_correct_bps_conversion_and_boundary(self):
        messages = build_messages(
            "Ak má model MAE 17 bps, znamená to 83 % úspešnosť?",
            mode=GENERAL_CHAT,
        )
        system = messages[0]["content"]
        self.assertIn("17 bps is 0.17 percentage points", system)
        self.assertIn("not 0.017%", system)
        self.assertIn("it is not a hit rate", system)
        self.assertIn("directional accuracy, or profit percentage", system)
        self.assertEqual(messages[-1]["role"], "user")

    def test_general_prompt_keeps_conversation_natural(self):
        messages = build_messages("Ahoj, ako sa ti dnes darí?", mode=GENERAL_CHAT)
        system = messages[0]["content"]
        self.assertIn("idiomatic sentences", system)
        self.assertIn("Do not pretend to have human", system)
        self.assertIn("Reply in the language used by the user", system)

    def test_no_analysis_source_is_invented_by_prompt(self):
        messages = build_messages(
            "Koľko OurAI zarobila minulý týždeň?",
            mode=GENERAL_CHAT,
        )
        system = messages[0]["content"]
        self.assertIn("Do not claim that you have live market data", system)
        self.assertIn("Never promise profit", system)
        self.assertEqual(len(messages), 2)


if __name__ == "__main__":
    unittest.main()
