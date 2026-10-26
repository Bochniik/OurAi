"""Synthetic regressions for MAE reply after latest-record error clarification.

No local Qwen, real journal, lockbox, market data or trading is used.
"""
from __future__ import annotations

import unittest
from unittest.mock import patch

from ourai_language.forward_followups import (
    forward_latest_error_ambiguity_clarification,
    forward_latest_error_mae_reply_question,
)
from ourai_language.ollama_runtime import LocalOllamaChatRuntime


LATEST = "Aká je posledná predikcia NVDA vo forward denníku?"
AMBIGUOUS = "A aká bola jej chyba?"
MAE = "Aká je MAE predikcií NVDA vo forward denníku?"


def history(*, latest=LATEST, evidence=LATEST,
            source="verified_forward_journal", ambiguity=AMBIGUOUS):
    prior = [
        {"role": "user", "content": latest},
        {"role": "assistant", "content": "Syntetický overený záznam.",
         "source": source, "evidence_question": evidence},
    ]
    clarification = forward_latest_error_ambiguity_clarification(
        ambiguity, prior,
    )
    return prior + [
        {"role": "user", "content": ambiguity},
        {"role": "assistant", "content": clarification,
         "source": "conversation_clarification"},
    ]


class LatestErrorMaeReplyTests(unittest.TestCase):
    def test_explicit_mae_replies_inherit_verified_ticker_only(self):
        for reply in (
            "MAE", "Myslím MAE.", "Zaujíma ma MAE?",
            "Priemernú MAE.", "Myslím priemernú chybu.",
            "Zaujíma ma priemerná chyba.", "Myslím tú priemernú chybu.",
        ):
            with self.subTest(reply=reply):
                self.assertEqual(
                    forward_latest_error_mae_reply_question(reply, history()),
                    MAE,
                )

    def test_rejects_other_metrics_mixed_intents_and_trading(self):
        for reply in (
            "Myslím chybu tej konkrétnej predikcie.",
            "Nemyslím MAE", "MAE aj smer", "MAE TSLA",
            "MAE z lockboxu", "MAE a kúp NVDA",
            "Aká bola MAE za celý rok?", "Koľko zarobila?",
            "MAE a čo konkrétna chyba?", "Čo znamená MAE?",
        ):
            with self.subTest(reply=reply):
                self.assertIsNone(
                    forward_latest_error_mae_reply_question(reply, history())
                )

    def test_rejects_untrusted_mismatched_or_interrupted_provenance(self):
        cases = [
            [],
            history()[:-1],
            history(source="historical_v3_reports"),
            history(source="model"),
            history(latest="Aká je posledná predikcia TSLA vo forward denníku?"),
            history(evidence="Aká je posledná predikcia TSLA vo forward denníku?"),
            history(latest="Aká je MAE predikcií NVDA vo forward denníku?",
                    evidence="Aká je MAE predikcií NVDA vo forward denníku?"),
            history(latest="Aká je posledná historická predikcia NVDA?"),
            history(latest="Aká je posledná predikcia NVDA a TSLA vo forward denníku?"),
            history() + [{"role": "user", "content": "Iná téma"}],
        ]
        wrong_source = history()
        wrong_source[-1]["source"] = "verified_forward_journal"
        cases.append(wrong_source)
        tampered = history()
        tampered[-1]["content"] = "Myslíš MAE?"
        cases.append(tampered)
        for index, case in enumerate(cases):
            with self.subTest(index=index):
                self.assertIsNone(
                    forward_latest_error_mae_reply_question("Myslím MAE", case)
                )

    def test_runtime_rereads_verified_forward_reader_without_qwen(self):
        runtime = LocalOllamaChatRuntime(
            forward_journal="synthetic-journal", knowledge_enabled=False,
        )
        with (
            patch.object(runtime, "_generate", side_effect=AssertionError("Qwen called")),
            patch(
                "ourai_language.ollama_runtime.maybe_forward_answer",
                return_value="Syntetická odpoveď z overeného čítania.",
            ) as reader,
        ):
            result = runtime.respond("Myslím MAE.", history=history())
        reader.assert_called_once_with(MAE, "synthetic-journal")
        self.assertEqual(result.answer_source, "verified_forward_journal")
        self.assertEqual(result.evidence_question, MAE)
        self.assertNotIn("Syntetický overený záznam", result.text)

    def test_invalid_reader_fails_closed(self):
        runtime = LocalOllamaChatRuntime(
            forward_journal="synthetic-journal", knowledge_enabled=False,
        )
        with (
            patch.object(runtime, "_generate", side_effect=AssertionError("Qwen called")),
            patch("ourai_language.ollama_runtime.maybe_forward_answer",
                  side_effect=ValueError("bad journal")),
        ):
            with self.assertRaisesRegex(RuntimeError, "Forward denník"):
                runtime.respond("Myslím MAE.", history=history())


if __name__ == "__main__":
    unittest.main()
