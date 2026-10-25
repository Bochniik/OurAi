"""Curated Slovak Tesla alias routing: mocked tool proposals, not Qwen tests."""
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


class VerifiedTeslaAliasTests(unittest.TestCase):
    def test_curated_slovak_forms_select_only_frozen_tsla(self):
        questions = (
            "Aké máš výsledky shadow pre akcie Tesla?",
            "Ako sa darí forward predikciám Tesly?",
            "Aká je posledná forward predikcia pre Teslu?",
            "Zhrň forward predikcie spoločnosti Teslou.",
            "Aké máš výsledky shadow pre ticker TESLA?",
            "Aké sú forward výsledky pre tesla?",
            "Aká je presnosť forward modelu Tesle?",
        )
        for question in questions:
            for action in ("summary", "performance", "latest"):
                with self.subTest(question=question, action=action):
                    selected = _select(question, action)
                    self.assertIsNotNone(selected)
                    self.assertEqual(selected[0], "forward_v2")
                    self.assertIn("TSLA", selected[1])
                    self.assertNotIn("NVDA", selected[1])

    def test_multi_company_or_unsupported_comparison_fails_closed(self):
        questions = (
            "Aké máš výsledky shadow pre Tesla a Nvidia?",
            "Aké máš výsledky shadow pre Nvidia a Tesla?",
            "Aká je presnosť forward predikcií NVDA oproti Tesle?",
            "Aké máš výsledky shadow pre Tesla a MSFT?",
            "Aké máš výsledky shadow pre Tesla a IBM?",
            "Aké máš výsledky shadow pre IBM a Tesla?",
            "Aké máš výsledky shadow pre Tesla vs Microsoft?",
            "Aké máš výsledky shadow pre Microsoft oproti Tesle?",
            "Aké máš výsledky shadow pre Tesla, Tesla, IBM?",
            "Aké máš výsledky shadow pre IBM, Tesla, Tesla?",
        )
        for question in questions:
            for action in ("summary", "performance", "latest"):
                with self.subTest(question=question, action=action):
                    self.assertIsNone(_select(question, action))

    def test_repeated_alias_and_ticker_stay_single_symbol(self):
        for question in (
            "Aké máš výsledky shadow pre Tesla a TSLA?",
            "Aké máš výsledky shadow pre TSLA a Tesly?",
            "Aké máš výsledky shadow pre Tesla, Tesla?",
            "Aká je MAE vo forward denníku pre Tesla a USD?",
        ):
            with self.subTest(question=question):
                selected = _select(question)
                self.assertIsNotNone(selected)
                self.assertIn("TSLA", selected[1])

    def test_no_extra_permissions_or_unverified_names(self):
        for question in (
            "Čo je Tesla?",
            "Aké máš výsledky shadow pre Tesla a nakúp akcie.",
            "Aké máš výsledky shadow pre akcie Microsoft?",
        ):
            with self.subTest(question=question):
                self.assertIsNone(_select(question))
        self.assertIsNone(_select(
            "Aké máš výsledky shadow pre Tesla?", available=False,
        ))


if __name__ == "__main__":
    unittest.main()
