"""Bounded indirect Slovak replies to verified forward accuracy clarification.

Synthetic reader only: no real journal, model invocation, trading or lockbox.
"""
from __future__ import annotations

import unittest
from unittest.mock import patch

from ourai_language.forward_followups import (
    forward_accuracy_clarification_reply_question,
)
from ourai_language.ollama_runtime import LocalOllamaChatRuntime


QUESTION = "Aká je úspešnosť forward predikcií NVDA?"
CLARIFICATION = (
    "Otázka, ako často boli predikcie správne, je nejednoznačná. "
    "Myslíš úspešnosť odhadu smeru alebo MAE?"
)
MAE = "Aká je MAE predikcií NVDA vo forward denníku?"
DIRECTION = "Aká je smerová presnosť forward predikcií NVDA?"


def history(*, question=QUESTION, evidence=None, source="verified_forward_journal",
            answer=CLARIFICATION):
    return [
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer, "source": source,
         "evidence_question": question if evidence is None else evidence},
    ]


class IndirectSlovakAccuracyReplyTests(unittest.TestCase):
    def test_indirect_magnitude_phrasings(self):
        for reply in (
            "Zaujíma ma skôr to, o koľko si sa pomýlil.",
            "Zaujima ma skor to o kolko si sa pomylil",
            "Chcem vedieť, o koľko si sa pomýlil.",
            "Skôr ma zaujíma, aká veľká bola chyba?",
            "Zaujíma ma, ako ďaleko boli odhady od reality.",
        ):
            with self.subTest(reply=reply):
                self.assertEqual(
                    forward_accuracy_clarification_reply_question(reply, history()),
                    MAE,
                )

    def test_indirect_direction_phrasings(self):
        for reply in (
            "Chcem vedieť, či si správne odhadol rast alebo pokles.",
            "Zaujima ma ci si spravne odhadol rast alebo pokles",
            "Skôr ma zaujíma, či si trafil smer?",
            "Chcem vediet ci si trafil smer",
        ):
            with self.subTest(reply=reply):
                self.assertEqual(
                    forward_accuracy_clarification_reply_question(reply, history()),
                    DIRECTION,
                )

    def test_overall_context_does_not_invent_symbol(self):
        overall = history(question="Ako často si sa trafil vo forward denníku?")
        self.assertEqual(
            forward_accuracy_clarification_reply_question(
                "Skôr ma zaujíma, aká veľká bola chyba?", overall
            ),
            "Aká je MAE predikcií vo forward denníku?",
        )
        self.assertEqual(
            forward_accuracy_clarification_reply_question(
                "Chcem vedieť, či si trafil smer.", overall
            ),
            "Aká je smerová presnosť forward predikcií?",
        )

    def test_rejects_negation_mixed_metrics_and_source_switch(self):
        for reply in (
            "Nie, nechcem vedieť, či si trafil smer",
            "Zaujíma ma skôr to, o koľko si sa pomýlil v lockboxe",
            "Chcem vedieť, či si správne odhadol rast alebo pokles TSLA",
            "Skôr ma zaujíma, či si trafil smer aj aká bola MAE",
            "Zaujíma ma skôr to, o koľko si sa pomýlil a kúp NVDA",
            "Čo znamená, že si trafil smer?",
        ):
            with self.subTest(reply=reply):
                self.assertIsNone(
                    forward_accuracy_clarification_reply_question(reply, history())
                )

    def test_rejects_unverified_or_interrupted_context(self):
        for prior in (
            [],
            history(source="historical_v3_reports"),
            history(answer="Forward denník neexistuje."),
            history(evidence="Aká je úspešnosť forward predikcií TSLA?"),
            history(question="Aká je úspešnosť historických predikcií NVDA?"),
            history() + [{"role": "user", "content": "Iná otázka"}],
        ):
            with self.subTest(prior=prior):
                self.assertIsNone(
                    forward_accuracy_clarification_reply_question(
                        "Zaujíma ma skôr to, o koľko si sa pomýlil", prior
                    )
                )

    def test_runtime_rereads_existing_verified_reader_without_qwen(self):
        runtime = LocalOllamaChatRuntime(
            forward_journal="synthetic", knowledge_enabled=False,
            semantic_tool_routing=True,
        )
        calls = []

        def reader(question, journal):
            self.assertEqual(journal, "synthetic")
            calls.append(question)
            if question == MAE:
                return "Overená syntetická MAE."
            if question == DIRECTION:
                return "Smerová úspešnosť nie je dostupná."
            raise AssertionError(f"Unexpected reader question: {question}")

        with patch(
            "ourai_language.ollama_runtime.maybe_forward_answer",
            side_effect=reader,
        ), patch.object(
            runtime, "_generate",
            side_effect=AssertionError("Local Qwen must not run"),
        ):
            mae = runtime.respond(
                "Zaujíma ma skôr to, o koľko si sa pomýlil.", history=history()
            )
            direction = runtime.respond(
                "Chcem vedieť, či si správne odhadol rast alebo pokles.",
                history=history(),
            )

        self.assertEqual(calls, [MAE, DIRECTION])
        self.assertEqual(mae.answer_source, "verified_forward_journal")
        self.assertEqual(mae.evidence_question, MAE)
        self.assertEqual(direction.evidence_question, DIRECTION)
        self.assertEqual(direction.text, "Smerová úspešnosť nie je dostupná.")


if __name__ == "__main__":
    unittest.main()
