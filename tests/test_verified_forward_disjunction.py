"""Disjunctive Slovak company questions must not silently narrow forward evidence.

Uses synthetic tool JSON proposals only; no local Qwen benchmark is run.
"""
from __future__ import annotations

import json
import unittest

from ourai_language.verified_tool_router import select_verified_tool


def _select(question: str, action: str = "performance", *, available: bool = True):
    proposal = json.dumps({
        "tool": "forward_v2", "action": action,
        "split": "both", "baseline": "both",
    })
    return select_verified_tool(
        question, proposal,
        historical_available=True, forward_available=available,
    )


class ForwardDisjunctionTests(unittest.TestCase):
    def test_unsupported_alternatives_fail_closed_for_all_forward_actions(self):
        questions = (
            "Aké máš výsledky shadow pre Nvidia alebo Alphabet?",
            "Aké máš výsledky shadow pre Alphabet alebo Nvidia?",
            "Aké máš výsledky shadow pre NVDA alebo Alphabet?",
            "Aké máš výsledky shadow pre Alphabet alebo NVDA?",
            "Aké máš výsledky shadow pre NVDA či ibm?",
            "Aké máš výsledky shadow pre ibm či NVDA?",
            "Aké máš výsledky shadow pre Nvidie či IBM?",
            "Aké máš výsledky shadow pre Microsoftu alebo Alphabet?",
            "Aké máš výsledky shadow pre Tesla či Alphabet?",
            "Aké máš výsledky shadow pre Nvidia alebo Tesla?",
        )
        for question in questions:
            for action in ("summary", "performance", "latest"):
                with self.subTest(question=question, action=action):
                    self.assertIsNone(_select(question, action))

    def test_same_symbol_alternatives_and_metric_words_still_work(self):
        questions = (
            ("Aké máš výsledky shadow pre Nvidia alebo NVDA?", "NVDA"),
            ("Aké máš výsledky shadow pre NVDA či NVDA?", "NVDA"),
            ("Aké máš výsledky shadow pre NVDA alebo MAE?", "NVDA"),
            ("Aké máš výsledky shadow pre Nvidia či Nvidie?", "NVDA"),
            ("Aké máš výsledky shadow pre Teslu alebo TSLA?", "TSLA"),
            ("Aké máš výsledky shadow pre Microsoft či MSFT?", "MSFT"),
        )
        for question, ticker in questions:
            for action in ("summary", "performance", "latest"):
                with self.subTest(question=question, action=action):
                    result = _select(question, action)
                    self.assertIsNotNone(result)
                    self.assertEqual(result[0], "forward_v2")
                    self.assertIn(ticker, result[1])

    def test_no_new_reader_permissions_or_educational_misrouting(self):
        self.assertIsNone(_select(
            "Aké máš výsledky shadow pre NVDA?", available=False,
        ))
        self.assertIsNone(_select("Čo znamená forward?"))
        self.assertIsNotNone(_select("Ako sa ti darí v shadow režime?"))


if __name__ == "__main__":
    unittest.main()
