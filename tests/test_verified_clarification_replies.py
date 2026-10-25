"""A short clarification answer must re-read only verified historical day wins."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ourai_language.conversation_followups import (
    historical_ambiguity_clarification,
    historical_clarification_reply_question,
    historical_followup_question,
)
from ourai_language.historical_answers import historical_question_kind
from ourai_language.ollama_runtime import LocalOllamaChatRuntime
from ourai_language.runtime import OurAIChatRuntime
from ourai_language.semantic_router import canonical_historical_question
from ourai_language.verified_tool_router import select_verified_tool


def _history(previous: str, ambiguous: str = "A oproti druhému modelu?"):
    verified = [
        {"role": "user", "content": previous},
        {"role": "assistant", "content": "Overené historické výsledky.",
         "source": "historical_v3_reports", "evidence_question": previous},
    ]
    clarification = historical_ambiguity_clarification(ambiguous, verified)
    assert clarification is not None
    return verified + [
        {"role": "user", "content": ambiguous},
        {"role": "assistant", "content": clarification,
         "source": "conversation_clarification", "evidence_question": ""},
    ]


def _report(split: str):
    return {
        "dataset_id": "synthetic-clarification-replies",
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
            "market_date_wins_vs_null": 3 if split == "validation" else 2,
        },
    }


class VerifiedClarificationReplyTests(unittest.TestCase):
    def test_short_replies_preserve_metric_and_split_without_guessing(self):
        for metric in ("Koľko trhových dní", "Koľko percent trhových dní"):
            for scope in ("v historickej validácii", "na historickom lockbox teste"):
                for old, target in (("lineárny", "nulový"), ("nulový", "lineárny")):
                    previous = f"{metric} model v3 prekonal {old} baseline {scope}?"
                    history = _history(previous)
                    expected = f"{metric} model v3 prekonal {target} baseline {scope}?"
                    for reply in (target, f"{target} baseline",
                                  f"Myslel som {target}.", f"Ten {target}."):
                        with self.subTest(metric=metric, scope=scope,
                                          target=target, reply=reply):
                            self.assertEqual(
                                historical_clarification_reply_question(reply, history),
                                expected,
                            )
                            self.assertEqual(
                                historical_followup_question(reply, history), expected,
                            )
                            # The bare reply has no standalone tool permission.
                            self.assertIsNone(historical_question_kind(reply))
                            self.assertIsNone(canonical_historical_question(
                                reply, json.dumps({
                                    "intent": "compare", "split": "test",
                                    "baseline": "null",
                                }),
                            ))
                            self.assertIsNone(select_verified_tool(
                                reply, json.dumps({
                                    "tool": "historical_v3", "action": "compare",
                                    "split": "test", "baseline": "null",
                                }), historical_available=True, forward_available=True,
                            ))

    def test_invalid_clarification_chain_fails_closed(self):
        previous = (
            "Koľko percent trhových dní model v3 prekonal "
            "lineárny baseline v historickej validácii?"
        )
        good = _history(previous)
        bad = (
            [], good[:2], good[:3],
            good + [{"role": "user", "content": "Iná téma."}],
            [{**good[0]}, {**good[1], "source": "model"}, *good[2:]],
            [{**good[0]}, {**good[1], "source": "verified_forward_journal"}, *good[2:]],
            [*good[:3], {**good[3], "source": "model"}],
            [*good[:3], {**good[3], "content": "Nulový baseline."}],
            [*good[:3], {**good[3], "evidence_question": previous}],
            [*good[:2], {**good[2], "content": "A oproti druhému baseline?"}, good[3]],
            [
                {"role": "user", "content": "Aká bola MAE modelu v3 v historickej validácii?"},
                {**good[1], "evidence_question":
                 "Aká bola MAE modelu v3 v historickej validácii?"},
                *good[2:],
            ],
            [{**good[0]}, {**good[1], "evidence_question":
              "Koľko dní model v3 prekonal lineárny baseline?"}, *good[2:]],
        )
        for history in bad:
            with self.subTest(history=history):
                self.assertIsNone(historical_followup_question("Nulový", history))
        for unsafe in (
            "Nulový a obchoduj", "Nulový vo forward denníku",
            "Nulový na lockboxe", "Nulový alebo lineárny",
            "Nulový, kúp akcie", "Nulový a vo validácii",
        ):
            with self.subTest(reply=unsafe):
                self.assertIsNone(historical_followup_question(unsafe, good))

    def test_both_runtimes_reread_verified_report_without_qwen(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for split, filename in (
                ("validation", "nonlinear_magnitude_validation_v1.json"),
                ("test", "nonlinear_magnitude_test_lockbox_v1.json"),
            ):
                (root / filename).write_text(
                    json.dumps(_report(split)), encoding="utf-8",
                )
            ollama = LocalOllamaChatRuntime(
                historical_reports_dir=root, semantic_tool_routing=True,
                knowledge_enabled=False,
            )
            transformers = OurAIChatRuntime.__new__(OurAIChatRuntime)
            transformers.knowledge_enabled = False
            transformers.historical_reports_dir = root
            transformers.forward_journal = None
            for runtime in (ollama, transformers):
                for metric, scope, count in (
                    ("Koľko trhových dní", "v historickej validácii", 3),
                    ("Koľko percent trhových dní", "na historickom lockbox teste", 2),
                ):
                    previous = (
                        f"{metric} model v3 prekonal lineárny baseline {scope}?"
                    )
                    with self.subTest(runtime=type(runtime).__name__, metric=metric):
                        with patch.object(ollama, "_generate",
                                          side_effect=AssertionError("Qwen must not run")):
                            first = runtime.respond(previous)
                            verified = [
                                {"role": "user", "content": previous},
                                {"role": "assistant", "content": first.text,
                                 "source": first.answer_source,
                                 "evidence_question": first.evidence_question},
                            ]
                            question = "A oproti druhému modelu?"
                            clarify = runtime.respond(question, history=verified)
                            self.assertEqual(clarify.answer_source, "conversation_clarification")
                            self.assertIsNone(clarify.evidence_question)
                            history = verified + [
                                {"role": "user", "content": question},
                                {"role": "assistant", "content": clarify.text,
                                 "source": clarify.answer_source,
                                 "evidence_question": clarify.evidence_question or ""},
                            ]
                            result = runtime.respond("Nulový.", history=history)
                        self.assertEqual(result.answer_source, "historical_v3_reports")
                        self.assertIn(f"v {count} z 3 trhových dní", result.text)
                        self.assertIn("nulový baseline", result.evidence_question)
                        self.assertIn(scope, result.evidence_question)
                        self.assertEqual("%" in result.text, "percent" in metric)
                        self.assertIsNone(historical_followup_question("Nulový.", []))


if __name__ == "__main__":
    unittest.main()
