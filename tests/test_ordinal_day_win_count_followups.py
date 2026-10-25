"""Verified historical day-win counts retain their metric on ordinal baseline changes."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ourai_language.conversation_followups import historical_followup_question
from ourai_language.historical_answers import historical_question_kind
from ourai_language.ollama_runtime import LocalOllamaChatRuntime
from ourai_language.semantic_router import canonical_historical_question
from ourai_language.verified_tool_router import select_verified_tool


def _history(question: str, source: str = "historical_v3_reports"):
    return [
        {"role": "user", "content": question},
        {"role": "assistant", "content": "Overená odpoveď.",
         "source": source, "evidence_question": question},
    ]


def _report(split: str):
    return {
        "dataset_id": "synthetic-ordinal-count",
        "protocol_name": "ourai-v3-nonlinear-research-preregistration-v1",
        "protocol_sha256": "a" * 64,
        "horizon_minutes": 15,
        "test_lockbox_accessed": split == "test",
        "original_validation_used": True,
        "post_test_tuning_allowed": False,
        f"{split}_supported": True,
        split: {
            "overall": {
                "evaluated_count": 15,
                "nonlinear_mae_bps": 8.0,
                "linear_mae_bps": 10.0,
                "null_mae_bps": 12.0,
            },
            "market_date_count": 3,
            "market_date_wins_vs_linear": 2 if split == "validation" else 1,
            "market_date_wins_vs_null": 1 if split == "validation" else 3,
        },
    }


class OrdinalDayWinCountFollowupTests(unittest.TestCase):
    def test_count_is_preserved_across_both_splits_and_baselines(self):
        for scope in ("v historickej validácii", "na historickom lockbox teste"):
            for original, target in (("lineárny", "nulový"), ("nulový", "lineárny")):
                previous = (
                    f"Koľko trhových dní model v3 prekonal "
                    f"{original} baseline {scope}?"
                )
                expected = (
                    f"Koľko trhových dní model v3 prekonal "
                    f"{target} baseline {scope}?"
                )
                for question in (
                    "A oproti druhému baseline?",
                    "A čo pri druhom baseline?",
                    "A ako to dopadlo oproti druhému baseline?",
                ):
                    with self.subTest(scope=scope, original=original, question=question):
                        self.assertEqual(
                            historical_followup_question(question, _history(previous)),
                            expected,
                        )

    def test_ambiguous_and_unverified_context_fails_closed(self):
        previous = (
            "Koľko trhových dní model v3 prekonal "
            "lineárny baseline v historickej validácii?"
        )
        question = "A oproti druhému baseline?"
        invalid_history = (
            [],
            _history(previous)[:-1],
            _history(previous, "model"),
            _history(previous, "verified_forward_journal"),
            _history(previous) + [{"role": "user", "content": "Nová téma."}],
            _history("Aká bola MAE modelu v3 v historickej validácii?"),
            _history("Porovnaj model v3 s lineárnym baseline."),
            _history("Aké boli historické výsledky modelu v3?"),
        )
        for history in invalid_history:
            with self.subTest(history=history):
                self.assertIsNone(historical_followup_question(question, history))
        for unsafe in (
            "A oproti druhému modelu?",
            "A oproti druhému?",
            "A oproti druhému baseline aj vo forward denníku?",
            "A oproti druhému baseline a obchoduj?",
            "A oproti druhému baseline vo validácii aj na lockboxe?",
        ):
            with self.subTest(question=unsafe):
                self.assertIsNone(historical_followup_question(unsafe, _history(previous)))
                self.assertIsNone(historical_question_kind(unsafe))
                self.assertIsNone(canonical_historical_question(
                    unsafe, json.dumps({
                        "intent": "compare", "split": "validation", "baseline": "null",
                    }),
                ))
                self.assertIsNone(select_verified_tool(
                    unsafe, json.dumps({
                        "tool": "historical_v3", "action": "compare",
                        "split": "validation", "baseline": "null",
                    }), historical_available=True, forward_available=True,
                ))
        self.assertIsNone(historical_question_kind(question))

    def test_runtime_rereads_count_without_llm_or_percentage_conversion(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for split, filename in (
                ("validation", "nonlinear_magnitude_validation_v1.json"),
                ("test", "nonlinear_magnitude_test_lockbox_v1.json"),
            ):
                (root / filename).write_text(
                    json.dumps(_report(split)), encoding="utf-8",
                )
            runtime = LocalOllamaChatRuntime(
                historical_reports_dir=root,
                semantic_tool_routing=True,
                knowledge_enabled=False,
            )
            with patch.object(runtime, "_generate",
                              side_effect=AssertionError("Qwen must not run")):
                for split, scope, expected_count in (
                    ("validation", "v historickej validácii", 1),
                    ("test", "na historickom lockbox teste", 3),
                ):
                    previous = (
                        f"Koľko trhových dní model v3 prekonal "
                        f"lineárny baseline {scope}?"
                    )
                    first = runtime.respond(previous)
                    self.assertEqual(first.answer_source, "historical_v3_reports")
                    result = runtime.respond(
                        "A oproti druhému baseline?",
                        history=[
                            {"role": "user", "content": previous},
                            {"role": "assistant", "content": first.text,
                             "source": first.answer_source,
                             "evidence_question": first.evidence_question},
                        ],
                    )
                    with self.subTest(split=split):
                        self.assertEqual(result.answer_source, "historical_v3_reports")
                        self.assertIn(f"v {expected_count} z 3 trhových dní", result.text)
                        self.assertNotIn("%", result.text)
                        self.assertEqual(
                            result.evidence_question,
                            f"Koľko trhových dní model v3 prekonal "
                            f"nulový baseline {scope}?",
                        )


if __name__ == "__main__":
    unittest.main()
