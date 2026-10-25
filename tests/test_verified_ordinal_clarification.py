"""Ambiguous 'other model' questions ask for clarification, not a report."""
from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from ourai_language.conversation_followups import (
    historical_ambiguity_clarification, historical_followup_question,
)
from ourai_language.historical_answers import historical_question_kind
from ourai_language.ollama_runtime import LocalOllamaChatRuntime
from ourai_language.runtime import OurAIChatRuntime
from ourai_language.semantic_router import canonical_historical_question
from ourai_language.verified_tool_router import select_verified_tool


def _history(question: str, source: str = "historical_v3_reports"):
    return [
        {"role": "user", "content": question},
        {"role": "assistant", "content": "Overené historické výsledky.",
         "source": source, "evidence_question": question},
    ]


class VerifiedOrdinalClarificationTests(unittest.TestCase):
    def test_ambiguous_model_or_unnamed_reference_clarifies_only_with_verified_days(self):
        for metric in ("Koľko percent trhových dní", "Koľko trhových dní"):
            for scope in ("v historickej validácii", "na historickom lockbox teste"):
                previous = f"{metric} model v3 prekonal lineárny baseline {scope}?"
                for question in (
                    "A ako to dopadlo oproti druhému modelu?",
                    "A oproti druhému modelu?",
                    "A oproti druhému?",
                ):
                    with self.subTest(metric=metric, scope=scope, question=question):
                        clarification = historical_ambiguity_clarification(
                            question, _history(previous),
                        )
                        self.assertIsNotNone(clarification)
                        self.assertIn("lineárnym", clarification)
                        self.assertIn("nulovým", clarification)
                        self.assertNotIn("%", clarification)
                        self.assertIsNone(historical_followup_question(
                            question, _history(previous),
                        ))
                        self.assertIsNone(historical_question_kind(question))
                        self.assertIsNone(canonical_historical_question(
                            question, json.dumps({
                                "intent": "compare", "split": "test",
                                "baseline": "null",
                            }),
                        ))
                        self.assertIsNone(select_verified_tool(
                            question, json.dumps({
                                "tool": "historical_v3", "action": "compare",
                                "split": "test", "baseline": "null",
                            }), historical_available=True, forward_available=True,
                        ))

    def test_invalid_history_or_extended_request_never_clarifies(self):
        previous = (
            "Koľko percent trhových dní model v3 prekonal "
            "lineárny baseline v historickej validácii?"
        )
        question = "A oproti druhému modelu?"
        for invalid in (
            [], _history(previous)[:-1],
            _history(previous, "model"),
            _history(previous, "verified_forward_journal"),
            _history(previous) + [{"role": "user", "content": "Iná téma."}],
            _history("Aká bola MAE modelu v3 v historickej validácii?"),
            _history("Porovnaj model v3 s lineárnym baseline."),
            _history("Koľko dní model v3 prekonal lineárny baseline?"),
        ):
            with self.subTest(history=invalid):
                self.assertIsNone(historical_ambiguity_clarification(question, invalid))
        for unsafe in (
            "A oproti druhému baseline?",
            "A oproti druhému modelu a obchoduj?",
            "A oproti druhému modelu vo forward denníku?",
            "A oproti druhému modelu aj na lockboxe?",
        ):
            with self.subTest(question=unsafe):
                self.assertIsNone(historical_ambiguity_clarification(
                    unsafe, _history(previous),
                ))

    def test_ollama_runtime_clarifies_without_reader_or_generation(self):
        previous = (
            "Koľko trhových dní model v3 prekonal "
            "nulový baseline na historickom lockbox teste?"
        )
        runtime = LocalOllamaChatRuntime(
            historical_reports_dir="/nonexistent/ordinal-clarification-reports",
            semantic_tool_routing=True, knowledge_enabled=False,
        )
        with patch.object(runtime, "_generate",
                          side_effect=AssertionError("Qwen must not run")):
            result = runtime.respond(
                "A ako to dopadlo oproti druhému modelu?",
                history=_history(previous),
            )
        self.assertEqual(result.answer_source, "conversation_clarification")
        self.assertIsNone(result.evidence_question)
        self.assertIn("Upresni", result.text)
        self.assertIsNone(historical_followup_question(
            "A oproti druhému baseline?",
            _history(previous) + [
                {"role": "user", "content": "A oproti druhému modelu?"},
                {"role": "assistant", "content": result.text,
                 "source": result.answer_source},
            ],
        ))

    def test_transformers_runtime_clarifies_before_model_loading(self):
        previous = (
            "Koľko percent trhových dní model v3 prekonal "
            "lineárny baseline v historickej validácii?"
        )
        # No local model is initialized; this verifies only the deterministic
        # pre-generation branch, not any Qwen language-quality benchmark.
        runtime = OurAIChatRuntime.__new__(OurAIChatRuntime)
        runtime.knowledge_enabled = False
        runtime.historical_reports_dir = "/nonexistent/ordinal-clarification-reports"
        runtime.forward_journal = None
        result = runtime.respond("A oproti druhému?", history=_history(previous))
        self.assertEqual(result.answer_source, "conversation_clarification")
        self.assertIsNone(result.evidence_question)
        self.assertIn("lineárnym", result.text)
        self.assertIn("nulovým", result.text)


if __name__ == "__main__":
    unittest.main()
