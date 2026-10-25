"""Slovak company-cue guard for opt-in verified forward selection.

Only synthetic classifier JSON is used; this is not a local Qwen benchmark.
"""
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


class ForwardCompanyCueTests(unittest.TestCase):
    def test_unverified_company_names_after_slovak_cues_fail_closed(self):
        questions = (
            "Aké máš výsledky shadow spoločnosti Alphabet?",
            "Aká je presnosť forward predikcií firmy Alphabet?",
            "Zhrň forward predikcie pre spoločnosť Apple.",
            "Aké máš výsledky shadow pre firmu ibm?",
            "Aké máš výsledky shadow pre spoločnosť IBM?",
            "Aké máš výsledky shadow pre spolocnosti Alphabet?",
            "Aké máš výsledky shadow spoločnosťou Alphabet?",
            "Aké máš výsledky shadow pre firmou Alphabet?",
        )
        for question in questions:
            for action in ("summary", "performance", "latest"):
                with self.subTest(question=question, action=action):
                    self.assertIsNone(_select(question, action))

    def test_verified_company_aliases_and_tickers_still_work(self):
        cases = (
            ("Aké máš výsledky shadow firmy Nvidia?", "NVDA"),
            ("Aké máš výsledky shadow spoločnosti Nvidie?", "NVDA"),
            ("Aké máš výsledky shadow spoločnosťou Teslou?", "TSLA"),
            ("Aké máš výsledky shadow firmy Tesla?", "TSLA"),
            ("Aké máš výsledky shadow spoločnosti Microsoftu?", "MSFT"),
            ("Aké máš výsledky shadow firmy Microsoft?", "MSFT"),
            ("Aké máš výsledky shadow firmy nvda?", "NVDA"),
        )
        for question, ticker in cases:
            for action in ("summary", "performance", "latest"):
                with self.subTest(question=question, action=action):
                    selected = _select(question, action)
                    self.assertIsNotNone(selected)
                    self.assertEqual(selected[0], "forward_v2")
                    self.assertIn(ticker, selected[1])

    def test_unknown_company_cannot_be_ignored_in_named_comparison(self):
        for question in (
            "Aké máš výsledky shadow spoločnosti Alphabet a NVDA?",
            "Aké máš výsledky shadow firmy IBM oproti NVDA?",
            "Aké máš výsledky shadow spoločnosti Microsoft a Alphabet?",
        ):
            with self.subTest(question=question):
                self.assertIsNone(_select(question))

    def test_no_new_permissions_or_general_company_lookup(self):
        self.assertIsNone(_select("Čo je spoločnosť Nvidia?"))
        self.assertIsNone(_select(
            "Aké máš výsledky shadow spoločnosti Nvidia?", available=False,
        ))
        self.assertEqual(
            _select("Ako sa ti darí v shadow režime?")[0], "forward_v2",
        )


if __name__ == "__main__":
    unittest.main()
