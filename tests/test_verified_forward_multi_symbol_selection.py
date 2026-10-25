"""Guard opt-in forward tool selection against silent ticker narrowing.

These are mocked JSON classifications, not a local Qwen benchmark.
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


class ForwardMultiSymbolSelectionTests(unittest.TestCase):
    def test_two_named_frozen_tickers_never_select_only_first(self):
        questions = (
            "Aké máš výsledky shadow pre NVDA a TSLA?",
            "Aká je posledná forward predikcia AAPL vs NVDA?",
            "Zhrň predikcie vo forward denníku pre tsla, nvda.",
            "Koľko forward predikcií máš pre NVDA a AAPL?",
        )
        for question in questions:
            for action in ("summary", "performance", "latest"):
                with self.subTest(question=question, action=action):
                    self.assertIsNone(select_verified_tool(
                        question, _proposal(action),
                        historical_available=True, forward_available=True,
                    ))

    def test_repeated_single_ticker_is_not_a_multi_ticker_request(self):
        question = "Aké máš výsledky shadow pre NVDA, len NVDA?"
        for action in ("summary", "performance", "latest"):
            with self.subTest(action=action):
                selected = select_verified_tool(
                    question, _proposal(action),
                    historical_available=False, forward_available=True,
                )
                self.assertIsNotNone(selected)
                self.assertEqual(selected[0], "forward_v2")
                self.assertIn("NVDA", selected[1])

    def test_single_ticker_and_overall_queries_still_work(self):
        for question, ticker in (
            ("Aké máš výsledky shadow pre NVDA?", "NVDA"),
            ("Ako sa ti darí v shadow režime?", None),
        ):
            with self.subTest(question=question):
                selected = select_verified_tool(
                    question, _proposal("performance"),
                    historical_available=False, forward_available=True,
                )
                self.assertIsNotNone(selected)
                self.assertEqual(selected[0], "forward_v2")
                self.assertEqual("NVDA" in selected[1], ticker == "NVDA")


if __name__ == "__main__":
    unittest.main()
