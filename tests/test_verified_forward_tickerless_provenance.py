"""Reject invented ticker context from unrelated tickerless user turns.

Synthetic source tags and readers only; no local Qwen benchmark is run.
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
        {"role": "assistant", "content": "Syntetický forward report.",
         "source": "verified_forward_journal", "evidence_question": evidence},
    ]


class TickerlessForwardProvenanceTests(unittest.TestCase):
    def test_unrelated_or_overall_question_cannot_introduce_named_ticker(self):
        for user in (
            "Ahoj.",
            "Ako sa máš?",
            "Aké máš výsledky shadow?",
            "Aká je posledná predikcia vo forward denníku?",
            "Ako sa ti darí v shadow režime?",
            "Vysvetli mi model.",
            "Ďakujem.",
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

    def test_tickerless_followup_must_match_recorded_metric(self):
        self.assertIsNone(forward_ticker_followup_question(
            "A čo TSLA?", history("Čo táto predikcia znamená?", PERFORMANCE),
        ))
        self.assertIsNone(forward_performance_followup_question(
            "A aká bola jej presnosť?",
            history("Čo tieto výsledky znamenajú?", LATEST),
        ))
        self.assertIsNone(forward_ticker_followup_question(
            "A čo TSLA?", history("A aká bola jej presnosť?", LATEST),
        ))

    def test_existing_bounded_slovak_chains_remain_valid(self):
        self.assertEqual(forward_ticker_followup_question(
            "A čo TSLA?", history("Čo táto predikcia znamená?", LATEST),
        ), "Aká je posledná predikcia TSLA vo forward denníku?")
        self.assertEqual(forward_ticker_followup_question(
            "A čo TSLA?", history("A aká bola jej presnosť?", PERFORMANCE),
        ), "Aká je presnosť predikcií TSLA vo forward denníku?")
        self.assertEqual(forward_latest_followup_question(
            "A aká bola jej posledná predikcia?",
            history("Čo tieto výsledky znamenajú?", PERFORMANCE),
        ), LATEST)
        self.assertEqual(forward_performance_followup_question(
            "A aká bola jej presnosť?",
            history("A aká bola jej posledná predikcia?", LATEST),
        ), PERFORMANCE)
        self.assertEqual(forward_results_explanation_followup_question(
            "Čo tieto výsledky znamenajú?",
            history("A aka bola jej presnost?", PERFORMANCE),
        ), PERFORMANCE)
        self.assertEqual(forward_latest_explanation_followup_question(
            "Čo táto predikcia znamená?",
            history("A čo jej posledná predikcia?", LATEST),
        ), LATEST)
        self.assertEqual(forward_performance_followup_question(
            "A aká bola jej presnosť?",
            history("A čo TSLA?",
                    "Aká je posledná predikcia TSLA vo forward denníku?"),
        ), "Aká je presnosť predikcií TSLA vo forward denníku?")


if __name__ == "__main__":
    unittest.main()
