"""Bounded ordinal-baseline follow-ups must use verified historical context."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ourai_language.conversation_followups import historical_followup_question
from ourai_language.historical_answers import (
    historical_question_kind, maybe_historical_answer,
    ordinal_other_comparison_kind,
)
from ourai_language.ollama_runtime import LocalOllamaChatRuntime
from ourai_language.semantic_router import canonical_historical_question
from ourai_language.verified_tool_router import select_verified_tool


def history(question: str, source: str = "historical_v3_reports"):
    return [
        {"role": "user", "content": question},
        {"role": "assistant", "content": "Overená historická odpoveď.",
         "source": source, "evidence_question": question},
    ]


class OrdinalBaselineFollowupTests(unittest.TestCase):
    def test_other_baseline_changes_only_verified_percentage_baseline(self):
        for scope in ("v historickej validácii", "na historickom lockbox teste"):
            for original, other in (
                ("lineárny", "nulový"), ("nulový", "lineárny"),
            ):
                previous = (
                    f"Koľko percent trhových dní model v3 prekonal "
                    f"{original} baseline {scope}?"
                )
                expected = (
                    f"Koľko percent trhových dní model v3 prekonal "
                    f"{other} baseline {scope}?"
                )
                for question in (
                    "A oproti druhému baseline?",
                    "A čo pri druhom baseline?",
                    "A ako to dopadlo oproti druhému baseline?",
                ):
                    with self.subTest(scope=scope, previous=original, question=question):
                        self.assertEqual(
                            ordinal_other_comparison_kind(question), "baseline",
                        )
                        self.assertEqual(
                            historical_followup_question(question, history(previous)),
                            expected,
                        )
                        self.assertIn(scope, expected)
                        self.assertIsNone(historical_question_kind(question))
                        self.assertIsNone(canonical_historical_question(
                            question, json.dumps({
                                "intent": "compare", "split": "test",
                                "baseline": "linear",
                            }),
                        ))
                        self.assertIsNone(select_verified_tool(
                            question, json.dumps({
                                "tool": "historical_v3", "action": "compare",
                                "split": "test", "baseline": "linear",
                            }), historical_available=True, forward_available=True,
                        ))

    def test_ambiguous_or_extended_other_model_never_inherits_a_baseline(self):
        previous = (
            "Koľko percent trhových dní model v3 prekonal "
            "lineárny baseline v historickej validácii?"
        )
        for question in (
            "A ako to dopadlo oproti druhému modelu?",
            "A oproti druhému modelu?",
            "A oproti druhému?",
            "A oproti druhému baseline a obchoduj?",
            "A oproti druhému baseline vo forward denníku?",
            "A čo pri druhom baseline aj vo validácii aj na lockboxe?",
        ):
            with self.subTest(question=question):
                self.assertIsNone(historical_followup_question(
                    question, history(previous),
                ))
                self.assertIsNone(historical_question_kind(question))
                self.assertIsNone(canonical_historical_question(
                    question, json.dumps({
                        "intent": "compare", "split": "validation",
                        "baseline": "null",
                    }),
                ))
                self.assertIsNone(select_verified_tool(
                    question, json.dumps({
                        "tool": "historical_v3", "action": "compare",
                        "split": "validation", "baseline": "null",
                    }), historical_available=True, forward_available=True,
                ))

    def test_unverified_or_wrong_metric_context_cannot_select_other_baseline(self):
        previous = (
            "Koľko percent trhových dní model v3 prekonal "
            "lineárny baseline v historickej validácii?"
        )
        question = "A oproti druhému baseline?"
        for invalid in (
            [],
            history(previous)[:-1],
            history(previous, "model"),
            history(previous, "verified_forward_journal"),
            history(previous) + [{"role": "user", "content": "Iná téma."}],
            history(previous.replace("Koľko percent trhových dní", "Koľko dní")),
            history("Aká bola MAE modelu v3 v historickej validácii?"),
            history("Porovnaj model v3 s lineárnym baseline."),
        ):
            with self.subTest(invalid=invalid):
                self.assertIsNone(historical_followup_question(question, invalid))

    def test_runtime_rereads_verified_counts_without_generating(self):
        previous = (
            "Koľko percent trhových dní model v3 prekonal "
            "lineárny baseline v historickej validácii?"
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = {
                "dataset_id": "synthetic-v3",
                "protocol_name": "ourai-v3-nonlinear-research-preregistration-v1",
                "protocol_sha256": "a" * 64,
                "horizon_minutes": 15,
                "test_lockbox_accessed": False,
                "original_validation_used": True,
                "validation_supported": True,
                "validation": {
                    "overall": {
                        "evaluated_count": 15,
                        "nonlinear_mae_bps": 8.0,
                        "linear_mae_bps": 10.0,
                        "null_mae_bps": 12.0,
                    },
                    "market_date_count": 3,
                    "market_date_wins_vs_linear": 2,
                    "market_date_wins_vs_null": 3,
                },
            }
            (root / "nonlinear_magnitude_validation_v1.json").write_text(
                json.dumps(report), encoding="utf-8",
            )
            runtime = LocalOllamaChatRuntime(
                historical_reports_dir=root, semantic_tool_routing=True,
                knowledge_enabled=False,
            )
            with patch.object(runtime, "_generate",
                              side_effect=AssertionError("Qwen must not run")):
                first = runtime.respond(previous)
                self.assertEqual(first.answer_source, "historical_v3_reports")
                self.assertIn("66,67 %", first.text)
                result = runtime.respond(
                    "A oproti druhému baseline?",
                    history=[
                        {"role": "user", "content": previous},
                        {"role": "assistant", "content": first.text,
                         "source": first.answer_source,
                         "evidence_question": first.evidence_question},
                    ],
                )
                self.assertEqual(result.answer_source, "historical_v3_reports")
                self.assertIn("100,00 %", result.text)
                self.assertIn("v 3 z 3", result.text)
                self.assertIn("nulový baseline", result.evidence_question)
                self.assertIn("nie je presnosť smeru UP/DOWN", result.text)
                self.assertIsNone(maybe_historical_answer(
                    "A oproti druhému baseline?", root,
                ))


if __name__ == "__main__":
    unittest.main()
