"""Reject ungrounded named prior turns before forward ticker inheritance.

All source tags and reader answers are synthetic; no local Qwen benchmark.
"""
from __future__ import annotations

import unittest

from ourai_language.forward_followups import (
    forward_latest_explanation_followup_question,
    forward_latest_followup_question,
    forward_performance_followup_question,
    forward_results_explanation_followup_question,
    forward_ticker_followup_question,
)

LATEST = "Aká je posledná predikcia NVDA vo forward denníku?"
PERFORMANCE = "Aká je presnosť predikcií NVDA vo forward denníku?"


def history(user: str, evidence: str):
    return [
        {"role": "user", "content": user},
        {"role": "assistant", "content": "Syntetická overená odpoveď.",
         "source": "verified_forward_journal", "evidence_question": evidence},
    ]


class NamedTurnForwardIntentTests(unittest.TestCase):
    def test_named_non_forward_turn_cannot_authorize_any_followup(self):
        # A source tag alone cannot convert a stock-news, price, trading,
        # educational or social turn into an eligible forward conversation.
        for user in (
            "Koľko stojí NVDA?",
            "Aké sú novinky o NVDA?",
            "Mám kúpiť NVDA?",
            "Ahoj NVDA.",
            "Povedz mi vtip o NVDA.",
            "Čo znamená forward predikcia NVDA?",
        ):
            for evidence in (LATEST, PERFORMANCE):
                with self.subTest(user=user, evidence=evidence):
                    rows = history(user, evidence)
                    self.assertIsNone(forward_ticker_followup_question(
                        "A čo TSLA?", rows,
                    ))
                    self.assertIsNone(forward_latest_followup_question(
                        "A aká bola jej posledná predikcia?", rows,
                    ))
                    self.assertIsNone(forward_performance_followup_question(
                        "A aká bola jej presnosť?", rows,
                    ))
                    self.assertIsNone(forward_latest_explanation_followup_question(
                        "Čo táto predikcia znamená?", rows,
                    ))
                    self.assertIsNone(forward_results_explanation_followup_question(
                        "Čo tieto výsledky znamenajú?", rows,
                    ))

    def test_real_forward_questions_remain_eligible(self):
        self.assertEqual(forward_ticker_followup_question(
            "A čo TSLA?", history(
                "Aké máš výsledky shadow pre NVDA?", PERFORMANCE,
            ),
        ), "Aká je presnosť predikcií TSLA vo forward denníku?")
        self.assertEqual(forward_performance_followup_question(
            "A aká bola jej presnosť?", history(
                "Aká je posledná predikcia NVDA vo forward denníku?", LATEST,
            ),
        ), PERFORMANCE)

    def test_bounded_named_switches_remain_eligible(self):
        for user in ("A čo NVDA?", "A čo pre NVDA?", "A pre NVDA?"):
            with self.subTest(user=user):
                self.assertEqual(forward_ticker_followup_question(
                    "A čo TSLA?", history(user, LATEST),
                ), "Aká je posledná predikcia TSLA vo forward denníku?")
                self.assertEqual(forward_performance_followup_question(
                    "A aká bola jej presnosť?", history(user, PERFORMANCE),
                ), PERFORMANCE)


if __name__ == "__main__":
    unittest.main()
