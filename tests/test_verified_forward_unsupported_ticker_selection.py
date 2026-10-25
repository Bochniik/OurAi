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

    def test_unsupported_ticker_in_longer_lists_cannot_be_skipped(self):
        # A non-overlapping pair regex used to see only the first pair.
        # It could silently answer for NVDA while ignoring a third IBM.
        questions = (
            "Aké máš výsledky shadow pre NVDA, NVDA, IBM?",
            "Aké máš výsledky shadow pre NVDA, MAE, IBM?",
            "Aké máš výsledky shadow pre NVDA, TSLA, IBM?",
            "Aké máš výsledky shadow pre NVDA a NVDA a IBM?",
        )
        for question in questions:
            for action in ("summary", "performance", "latest"):
                with self.subTest(question=question, action=action):
                    self.assertIsNone(select_verified_tool(
                        question, _proposal(action),
                        historical_available=True, forward_available=True,
                    ))

    def test_lowercase_unsupported_symbols_after_explicit_ticker_cues(self):
        # The old uppercase-only capture silently answered overall or for
        # NVDA when a user explicitly named an unsupported lowercase ticker.
        questions = (
            "Aké máš výsledky shadow pre ticker ibm?",
            "Aká je presnosť forward predikcií pre symbol ibm?",
            "Zhrň predikcie vo forward denníku pre akciu ibm.",
            "Aké máš výsledky shadow pre stock ibm a NVDA?",
            "Aké máš výsledky shadow pre ticker nvda a symbol ibm?",
            "Aké máš výsledky shadow pre akcie Tesla?",
        )
        for question in questions:
            for action in ("summary", "performance", "latest"):
                with self.subTest(question=question, action=action):
                    self.assertIsNone(select_verified_tool(
                        question, _proposal(action),
                        historical_available=True, forward_available=True,
                    ))

    def test_lowercase_supported_tickers_and_unmarked_prose_still_work(self):
        for question, ticker in (
            ("Aké máš výsledky shadow pre ticker nvda?", "NVDA"),
            ("Aké máš výsledky shadow pre symbol tsla?", "TSLA"),
            ("Aké máš výsledky shadow pre akcie meta?", "META"),
            ("Aké máš výsledky shadow pre nvda?", "NVDA"),
            ("Aká je MAE vo forward denníku pre symbol usd a NVDA?", "NVDA"),
            ("Ako sa ti darí v shadow režime?", None),
        ):
            with self.subTest(question=question):
                result = select_verified_tool(
                    question, _proposal("performance"),
                    historical_available=False, forward_available=True,
                )
                self.assertIsNotNone(result)
                self.assertEqual(result[0], "forward_v2")
                if ticker:
                    self.assertIn(ticker, result[1])
                else:
                    self.assertFalse(any(
                        symbol in result[1] for symbol in FROZEN_SYMBOLS
                    ))

    def test_repeated_supported_tickers_and_metric_words_still_work(self):
        for question in (
            "Aké máš výsledky shadow pre NVDA, NVDA, NVDA?",
            "Aké máš výsledky shadow pre NVDA, MAE, NVDA?",
            "Aké máš výsledky shadow pre NVDA, NVDA, USD?",
        ):
            with self.subTest(question=question):
                selected = select_verified_tool(
                    question, _proposal("performance"),
                    historical_available=False, forward_available=True,
                )
                self.assertIsNotNone(selected)
                self.assertEqual(selected[0], "forward_v2")
                self.assertIn("NVDA", selected[1])

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
