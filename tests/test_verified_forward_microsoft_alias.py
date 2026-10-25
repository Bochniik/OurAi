"""Curated Slovak Microsoft alias routing; mocked proposals, not a Qwen benchmark."""
from __future__ import annotations

import json
import unittest

from ourai_language.verified_tool_router import select_verified_tool


def _proposal(action: str) -> str:
    return json.dumps({
        "tool": "forward_v2", "action": action,
        "split": "both", "baseline": "both",
    })


def _select(question: str, action: str = "performance", *, available: bool = True):
    return select_verified_tool(
        question, _proposal(action),
        historical_available=True, forward_available=available,
    )


class VerifiedMicrosoftAliasTests(unittest.TestCase):
    def test_curated_slovak_forms_select_only_frozen_msft(self):
        questions = (
            "Aké máš výsledky shadow pre akcie Microsoft?",
            "Ako sa darí forward predikciám Microsoftu?",
            "Aká je posledná forward predikcia pre Microsoft?",
            "Zhrň forward predikcie spoločnosti Microsoftom.",
            "Aké máš výsledky shadow pre ticker microsoft?",
            "Aké sú forward výsledky pre Microsofte?",
            "Aká je presnosť forward modelu Microsoftu?",
        )
        for question in questions:
            for action in ("summary", "performance", "latest"):
                with self.subTest(question=question, action=action):
                    selected = _select(question, action)
                    self.assertIsNotNone(selected)
                    self.assertEqual(selected[0], "forward_v2")
                    self.assertIn("MSFT", selected[1])
                    self.assertNotIn("NVDA", selected[1])
                    self.assertNotIn("TSLA", selected[1])

    def test_multi_company_and_unknown_comparisons_fail_closed(self):
        questions = (
            "Aké máš výsledky shadow pre Microsoft a Nvidia?",
            "Aké máš výsledky shadow pre Tesla a Microsoft?",
            "Aká je presnosť forward predikcií MSFT oproti Tesle?",
            "Aké máš výsledky shadow pre Microsoft a NVDA?",
            "Aké máš výsledky shadow pre Microsoft a IBM?",
            "Aké máš výsledky shadow pre IBM a Microsoft?",
            "Aké máš výsledky shadow pre Microsoft vs Alphabet?",
            "Aké máš výsledky shadow pre Alphabet oproti Microsoftu?",
            "Aké máš výsledky shadow pre Microsoft, Microsoft, IBM?",
            "Aké máš výsledky shadow pre IBM, Microsoft, Microsoft?",
        )
        for question in questions:
            for action in ("summary", "performance", "latest"):
                with self.subTest(question=question, action=action):
                    self.assertIsNone(_select(question, action))

    def test_repeated_alias_and_ticker_are_one_symbol(self):
        for question in (
            "Aké máš výsledky shadow pre Microsoft a MSFT?",
            "Aké máš výsledky shadow pre MSFT a Microsoftu?",
            "Aké máš výsledky shadow pre Microsoft, Microsoft?",
            "Aká je MAE vo forward denníku pre Microsoft a USD?",
        ):
            with self.subTest(question=question):
                selected = _select(question)
                self.assertIsNotNone(selected)
                self.assertIn("MSFT", selected[1])

    def test_unrelated_requests_and_missing_reader_do_not_gain_access(self):
        for question in (
            "Čo je Microsoft?",
            "Aké máš výsledky shadow pre Microsoft a nakúp akcie.",
            "Aké máš výsledky shadow pre akcie Alphabet?",
        ):
            with self.subTest(question=question):
                self.assertIsNone(_select(question))
        self.assertIsNone(_select(
            "Aké máš výsledky shadow pre Microsoft?", available=False,
        ))


if __name__ == "__main__":
    unittest.main()
