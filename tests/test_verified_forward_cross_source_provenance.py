"""Reject contradictory historical/lockbox provenance in forward follow-ups.

All history and evidence are synthetic; no local Qwen benchmark is run.
"""
from __future__ import annotations

import unittest

from ourai_language.forward_followups import (
    forward_latest_explanation_followup_question,
    forward_latest_followup_question,
    forward_performance_followup_question,
    forward_ticker_followup_question,
)


LATEST = "Aká je posledná predikcia NVDA vo forward denníku?"
PERFORMANCE = "Aká je presnosť predikcií NVDA vo forward denníku?"


def history(user: str, evidence: str = LATEST):
    return [
        {"role": "user", "content": user},
        {"role": "assistant", "content": "Syntetický záznam.",
         "source": "verified_forward_journal", "evidence_question": evidence},
    ]


class ForwardCrossSourceProvenanceTests(unittest.TestCase):
    def test_retrospective_original_user_never_authorizes_forward_inheritance(self):
        # A forged/malformed source tag cannot override the user's source.
        for user in (
            "Aké boli historické výsledky NVDA?",
            "Čo ukázal lockbox NVDA?",
            "Aká bola MAE NVDA vo validácii?",
            "Čo ukázal NVDA backtest?",
            "Aké sú retrospektívne výsledky NVDA?",
            "NVDA validation výsledky?",
            "Porovnaj forward NVDA a lockboxe.",
        ):
            with self.subTest(user=user):
                rows = history(user)
                self.assertIsNone(forward_latest_followup_question(
                    "A aká bola jej posledná predikcia?", rows,
                ))
                self.assertIsNone(forward_performance_followup_question(
                    "A aká bola jej presnosť?", rows,
                ))
                self.assertIsNone(forward_ticker_followup_question(
                    "A čo TSLA?", rows,
                ))
                self.assertIsNone(forward_latest_explanation_followup_question(
                    "Čo táto predikcia znamená?", rows,
                ))

    def test_retrospective_stored_question_cannot_authorize_forward_switch(self):
        for evidence in (
            "Aká je posledná predikcia NVDA v lockboxe?",
            "Aká je presnosť predikcií NVDA v historickom teste?",
        ):
            with self.subTest(evidence=evidence):
                rows = history(LATEST, evidence)
                self.assertIsNone(forward_ticker_followup_question(
                    "A čo TSLA?", rows,
                ))
                self.assertIsNone(forward_performance_followup_question(
                    "A aká bola jej presnosť?", rows,
                ))

    def test_legitimate_forward_chains_still_work(self):
        self.assertEqual(
            forward_latest_followup_question(
                "A aká bola jej posledná predikcia?", history(LATEST),
            ), LATEST,
        )
        self.assertEqual(
            forward_performance_followup_question(
                "A aká bola jej presnosť?", history(LATEST),
            ), PERFORMANCE,
        )
        self.assertEqual(
            forward_ticker_followup_question(
                "A čo TSLA?", history("Čo táto predikcia znamená?"),
            ), "Aká je posledná predikcia TSLA vo forward denníku?",
        )
        self.assertEqual(
            forward_latest_followup_question(
                "A aká bola jej posledná predikcia?",
                history("A aká bola jej presnosť?", PERFORMANCE),
            ), LATEST,
        )


if __name__ == "__main__":
    unittest.main()
