"""Curated Slovak Nvidia alias routing: mocked tool proposals, not Qwen tests."""
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


class VerifiedNvidiaAliasTests(unittest.TestCase):
    def test_curated_name_forms_select_only_frozen_nvda(self):
        questions = (
            "Aké máš výsledky shadow pre akcie Nvidia?",
            "Ako sa darí forward predikciám Nvidie?",
            "Aká je posledná forward predikcia pre Nvidiu?",
            "Zhrň forward predikcie spoločnosti Nvidiou.",
            "Aké máš výsledky shadow pre ticker NVIDIA?",
            "Aké sú forward výsledky pre nvidia?",
        )
        for question in questions:
            for action in ("summary", "performance", "latest"):
                with self.subTest(question=question, action=action):
                    selected = _select(question, action)
                    self.assertIsNotNone(selected)
                    self.assertEqual(selected[0], "forward_v2")
                    self.assertIn("NVDA", selected[1])

    def test_alias_with_unknown_company_or_ticker_fails_closed(self):
        questions = (
            "Aké máš výsledky shadow pre Nvidia a IBM?",
            "Aké máš výsledky shadow pre IBM a Nvidia?",
            "Aké máš výsledky shadow pre Nvidia a Tesla?",
            "Aké máš výsledky shadow pre Tesla a Nvidia?",
            "Aké máš výsledky shadow pre Nvidia vs Microsoft?",
            "Aké máš výsledky shadow pre Microsoft oproti Nvidii?",
            "Aké máš výsledky shadow pre Nvidia a TSLA?",
            "Aké máš výsledky shadow pre akcie Nvidia a IBM?",
        )
        for question in questions:
            for action in ("summary", "performance", "latest"):
                with self.subTest(question=question, action=action):
                    self.assertIsNone(_select(question, action))

    def test_repeated_alias_and_ticker_do_not_create_fake_comparison(self):
        for question in (
            "Aké máš výsledky shadow pre Nvidia a NVDA?",
            "Aké máš výsledky shadow pre NVDA a Nvidie?",
            "Aké máš výsledky shadow pre Nvidia, Nvidia?",
            "Aká je MAE vo forward denníku pre Nvidia a MAE?",
        ):
            with self.subTest(question=question):
                selected = _select(question)
                self.assertIsNotNone(selected)
                self.assertIn("NVDA", selected[1])

    def test_no_new_tool_permissions_or_unsupported_name_mapping(self):
        for question in (
            "Čo je Nvidia?",
            "Aké máš výsledky shadow pre Nvidia a nakúp akcie.",
            "Aké máš výsledky shadow pre akcie Microsoft?",
        ):
            with self.subTest(question=question):
                self.assertIsNone(_select(question))
        self.assertIsNone(_select(
            "Aké máš výsledky shadow pre Nvidia?", available=False,
        ))


if __name__ == "__main__":
    unittest.main()
