"""Reject mismatched metric provenance in named verified-forward turns.

The previous user's requested metric must match the recorded reader question;
source tags, assistant prose and synthetic figures alone are not authorization.
No local Qwen benchmark or live journal access is performed.
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


def _history(user: str, evidence: str):
    return [
        {"role": "user", "content": user},
        {
            "role": "assistant",
            "content": "Syntetická odpoveď; nie skutočné metriky.",
            "source": "verified_forward_journal",
            "evidence_question": evidence,
        },
    ]


class NamedMetricProvenanceTests(unittest.TestCase):
    def test_latest_user_request_cannot_authorize_performance_evidence(self):
        for user in (
            LATEST,
            "Aká bola posledná predikcia NVDA?",
            "Ukáž mi najnovšiu predikciu NVDA.",
        ):
            with self.subTest(user=user):
                history = _history(user, PERFORMANCE)
                self.assertIsNone(forward_ticker_followup_question("A čo TSLA?", history))
                self.assertIsNone(forward_latest_followup_question(
                    "A aká bola jej posledná predikcia?", history,
                ))
                self.assertIsNone(forward_performance_followup_question(
                    "A aká bola jej presnosť?", history,
                ))
                self.assertIsNone(forward_results_explanation_followup_question(
                    "Čo tieto výsledky znamenajú?", history,
                ))

    def test_performance_user_request_cannot_authorize_latest_evidence(self):
        for user in (
            PERFORMANCE,
            "Aká je MAE predikcií NVDA?",
            "Aké máš výsledky shadow pre NVDA?",
        ):
            with self.subTest(user=user):
                history = _history(user, LATEST)
                self.assertIsNone(forward_ticker_followup_question("A čo TSLA?", history))
                self.assertIsNone(forward_latest_followup_question(
                    "A aká bola posledná predikcia?", history,
                ))
                self.assertIsNone(forward_performance_followup_question(
                    "A aká bola jej presnosť?", history,
                ))
                self.assertIsNone(forward_latest_explanation_followup_question(
                    "Čo táto predikcia znamená?", history,
                ))

    def test_matching_metrics_and_bounded_ticker_switches_still_work(self):
        self.assertEqual(
            forward_ticker_followup_question(
                "A čo TSLA?", _history(LATEST, LATEST),
            ),
            "Aká je posledná predikcia TSLA vo forward denníku?",
        )
        self.assertEqual(
            forward_ticker_followup_question(
                "A čo TSLA?", _history(PERFORMANCE, PERFORMANCE),
            ),
            "Aká je presnosť predikcií TSLA vo forward denníku?",
        )
        self.assertEqual(
            forward_performance_followup_question(
                "A aká bola jej presnosť?", _history(LATEST, LATEST),
            ),
            PERFORMANCE,
        )
        self.assertEqual(
            forward_latest_followup_question(
                "A aká bola jej posledná predikcia?",
                _history("A čo NVDA?", PERFORMANCE),
            ),
            LATEST,
        )
        self.assertEqual(
            forward_ticker_followup_question(
                "A čo TSLA?", _history("A pre NVDA?", PERFORMANCE),
            ),
            "Aká je presnosť predikcií TSLA vo forward denníku?",
        )


if __name__ == "__main__":
    unittest.main()
