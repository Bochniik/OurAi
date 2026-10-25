"""Fail-closed opt-in tool routing for explicitly unsupported stock symbols.

Mocked classifier JSON only; this is not a local Qwen benchmark.
"""
from __future__ import annotations

import json
import unittest

from ourai_language.verified_tool_router import select_verified_tool
from trading.forward_analytics import FROZEN_SYMBOLS


def _proposal(action: str) -> str:
    return json.dumps({
        "tool": "forward_v2", "action": action,
        "split": "both", "baseline": "both",
    })


class UnsupportedForwardTickerTests(unittest.TestCase):
    def test_explicit_unknown_ticker_cannot_be_silently_dropped(self):
        self.assertNotIn("IBM", FROZEN_SYMBOLS)
        questions = (
            "Aké máš výsledky shadow pre NVDA a IBM?",
            "Aká je presnosť forward predikcií IBM oproti NVDA?",
            "Aké sú posledné forward predikcie NVDA vs IBM?",
            "Zhrň predikcie vo forward denníku pre IBM.",
            "Aké máš výsledky shadow pre $IBM?",
            "Zhrň forward predikcie pre NVDA/IBM.",
            "Aké máš výsledky shadow pre NVDA, IBM?",
            "Aké máš výsledky shadow pre IBM a NVDA?",
            "Aké máš výsledky shadow pre ticker IBM?",
        )
        for question in questions:
            for action in ("summary", "performance", "latest"):
                with self.subTest(question=question, action=action):
                    self.assertIsNone(select_verified_tool(
                        question, _proposal(action),
                        historical_available=True, forward_available=True,
                    ))

    def test_known_ticker_metrics_and_overall_questions_remain_allowed(self):
        questions = (
            ("Aké máš výsledky shadow pre NVDA?", "NVDA"),
            ("Aká je MAE pre NVDA vo forward denníku?", "NVDA"),
            ("Aké máš výsledky shadow pre NVDA a MAE?", "NVDA"),
            ("Ako sa ti darí v shadow režime?", None),
            ("Aká je MAE vo forward denníku?", None),
        )
        for question, ticker in questions:
            for action in ("summary", "performance", "latest"):
                with self.subTest(question=question, action=action):
                    result = select_verified_tool(
                        question, _proposal(action),
                        historical_available=False, forward_available=True,
                    )
                    self.assertIsNotNone(result)
                    self.assertEqual(result[0], "forward_v2")
                    self.assertEqual("NVDA" in result[1], ticker == "NVDA")

    def test_guard_does_not_grant_permission_without_forward_reader(self):
        self.assertIsNone(select_verified_tool(
            "Aké máš výsledky shadow pre NVDA?",
            _proposal("performance"),
            historical_available=True, forward_available=False,
        ))


if __name__ == "__main__":
    unittest.main()
