"""Missing historical day-win split requires clarification, never inferred data."""
from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from ourai_language.conversation_followups import (
    historical_followup_question, historical_missing_split_clarification,
)
from ourai_language.historical_answers import (
    historical_day_wins_missing_split, historical_question_kind,
    maybe_historical_answer,
)
from ourai_language.ollama_runtime import LocalOllamaChatRuntime
from ourai_language.runtime import OurAIChatRuntime
from ourai_language.semantic_router import canonical_historical_question
from ourai_language.verified_tool_router import select_verified_tool


MISSING = (
    "Koľko dní model v3 prekonal lineárny baseline?",
    "Koľko trhových dní model v3 prekonal nulový baseline?",
    "Koľko percent trhových dní model v3 prekonal lineárny baseline?",
    "Aké percento dní model v3 prekonal nulový baseline?",
)
REPORTS = "/nonexistent/ourai-missing-split-reports"


class MissingDayWinSplitTests(unittest.TestCase):
    def test_explicit_but_incomplete_questions_never_open_reports(self):
        for question in MISSING:
            with self.subTest(question=question):
                self.assertTrue(historical_day_wins_missing_split(question))
                self.assertIsNone(historical_question_kind(question))
                self.assertIsNone(maybe_historical_answer(question, REPORTS))
                reply = historical_missing_split_clarification(question)
                self.assertIn("validáciu", reply)
                self.assertIn("lockbox", reply)
                self.assertNotIn("%", reply)
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

    def test_full_split_remains_authoritative_and_extras_do_not_clarify(self):
        for question in (
            "Koľko dní model v3 prekonal lineárny baseline v historickej validácii?",
            "Koľko percent trhových dní model v3 prekonal nulový baseline "
            "na historickom lockbox teste?",
        ):
            self.assertFalse(historical_day_wins_missing_split(question))
            self.assertIsNone(historical_missing_split_clarification(question))
            self.assertIsNotNone(historical_question_kind(question))
        for question in (
            "Čo je MAE?",
            "Koľko dní model v3 prekonal baseline?",
            "Koľko dní model v3 prekonal lineárny baseline vo forward denníku?",
            "Koľko dní model v3 prekonal lineárny baseline a obchoduj?",
            "Koľko dní model v3 prekonal lineárny baseline na lockboxe?",
            "Koľko dní model v3 prekonal lineárny baseline vo validácii?",
        ):
            with self.subTest(question=question):
                self.assertFalse(historical_day_wins_missing_split(question))
                self.assertIsNone(historical_missing_split_clarification(question))

    def test_both_runtimes_clarify_without_qwen_or_report(self):
        ollama = LocalOllamaChatRuntime(
            historical_reports_dir=REPORTS,
            semantic_tool_routing=True, knowledge_enabled=False,
        )
        transformers = OurAIChatRuntime.__new__(OurAIChatRuntime)
        transformers.knowledge_enabled = False
        transformers.historical_reports_dir = REPORTS
        transformers.forward_journal = None
        for runtime in (ollama, transformers):
            for question in MISSING:
                with self.subTest(runtime=type(runtime).__name__, question=question):
                    with patch.object(ollama, "_generate",
                                      side_effect=AssertionError("Qwen must not run")):
                        result = runtime.respond(question)
                    self.assertEqual(result.answer_source, "conversation_clarification")
                    self.assertIsNone(result.evidence_question)
                    self.assertIn("Upresni zdroj", result.text)
                    self.assertIsNone(historical_followup_question(
                        "A oproti druhému baseline?", [
                            {"role": "user", "content": question},
                            {"role": "assistant", "content": result.text,
                             "source": result.answer_source},
                        ],
                    ))


if __name__ == "__main__":
    unittest.main()
