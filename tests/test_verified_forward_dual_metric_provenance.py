"""Reject ambiguous mixed latest/performance provenance in verified forward chat.

Synthetic source tags only: no journal, model weights, or local Qwen benchmark.
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
        {
            "role": "assistant",
            "content": "Syntetická odpoveď, nie skutočné výsledky.",
            "source": "verified_forward_journal",
            "evidence_question": evidence,
        },
    ]


class DualMetricForwardProvenanceTests(unittest.TestCase):
    def test_mixed_named_requests_cannot_authorize_one_metric(self):
        for user in (
            "Aká je posledná predikcia NVDA a aká je jej presnosť?",
            "Ukáž mi najnovšiu predikciu NVDA a MAE.",
            "Aké boli výsledky poslednej predikcie NVDA?",
            "Aká je chyba poslednej predikcie NVDA?",
            "Ukáž poslednú predikciu NVDA aj jej presnosť.",
        ):
            for evidence in (LATEST, PERFORMANCE):
                with self.subTest(user=user, evidence=evidence):
                    prior = history(user, evidence)
                    self.assertIsNone(
                        forward_ticker_followup_question("A čo TSLA?", prior)
                    )
                    self.assertIsNone(
                        forward_latest_followup_question(
                            "A aká bola jej posledná predikcia?", prior
                        )
                    )
                    self.assertIsNone(
                        forward_performance_followup_question(
                            "A aká bola jej presnosť?", prior
                        )
                    )
                    self.assertIsNone(
                        forward_results_explanation_followup_question(
                            "Čo tieto výsledky znamenajú?", prior
                        )
                    )
                    self.assertIsNone(
                        forward_latest_explanation_followup_question(
                            "Čo táto predikcia znamená?", prior
                        )
                    )

    def test_single_metric_named_requests_remain_eligible(self):
        self.assertEqual(
            forward_ticker_followup_question(
                "A čo TSLA?", history(LATEST, LATEST)
            ),
            "Aká je posledná predikcia TSLA vo forward denníku?",
        )
        self.assertEqual(
            forward_ticker_followup_question(
                "A čo TSLA?", history(PERFORMANCE, PERFORMANCE)
            ),
            "Aká je presnosť predikcií TSLA vo forward denníku?",
        )
        self.assertEqual(
            forward_performance_followup_question(
                "A aká bola jej presnosť?", history(LATEST, LATEST)
            ),
            PERFORMANCE,
        )
        self.assertEqual(
            forward_latest_followup_question(
                "A aká bola jej posledná predikcia?",
                history(PERFORMANCE, PERFORMANCE),
            ),
            LATEST,
        )


if __name__ == "__main__":
    unittest.main()
