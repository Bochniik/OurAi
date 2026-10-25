"""An explicit historical source reply requires the exact prior clarification."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ourai_language.conversation_followups import (
    historical_followup_question, historical_missing_split_clarification,
    historical_missing_split_reply_question,
)
from ourai_language.historical_answers import (
    bare_historical_split_reply_target, historical_day_wins_missing_split_context,
    historical_question_kind, maybe_historical_answer,
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
REPLIES = (
    ("Validáciu.", "validation"),
    ("Historickú validáciu.", "validation"),
    ("Lockbox.", "test"),
    ("Lockbox test.", "test"),
    ("Historický lockbox test.", "test"),
)
REPORTS = "/nonexistent/ourai-split-replies"


def _history(question: str) -> list[dict[str, str]]:
    clarification = historical_missing_split_clarification(question)
    assert clarification is not None
    return [
        {"role": "user", "content": question},
        {"role": "assistant", "content": clarification,
         "source": "conversation_clarification", "evidence_question": ""},
    ]


def _report(split: str) -> dict:
    return {
        "dataset_id": "synthetic-source-clarification",
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


class VerifiedMissingSplitReplyTests(unittest.TestCase):
    def test_reconstruct_only_metric_baseline_and_explicit_split(self):
        for question in MISSING:
            context = historical_day_wins_missing_split_context(question)
            self.assertIsNotNone(context)
            metric, baseline = context
            history = _history(question)
            for reply, split in REPLIES:
                with self.subTest(question=question, reply=reply):
                    expected = (
                        f"{'Koľko percent trhových dní' if metric == 'share' else 'Koľko trhových dní'} "
                        f"model v3 prekonal {'lineárny' if baseline == 'linear' else 'nulový'} "
                        f"baseline {'v historickej validácii' if split == 'validation' else 'na historickom lockbox teste'}?"
                    )
                    self.assertEqual(
                        historical_missing_split_reply_question(reply, history), expected,
                    )
                    self.assertEqual(historical_followup_question(reply, history), expected)
                    self.assertEqual(bare_historical_split_reply_target(reply), split)
                    # Neither the deterministic nor model-proposed standalone
                    # router may turn the short reply into report access.
                    self.assertIsNone(historical_question_kind(reply))
                    self.assertIsNone(maybe_historical_answer(reply, REPORTS))
                    self.assertIsNone(canonical_historical_question(
                        reply, json.dumps({
                            "intent": "compare", "split": "test", "baseline": "null",
                        }),
                    ))
                    self.assertIsNone(select_verified_tool(
                        reply, json.dumps({
                            "tool": "historical_v3", "action": "compare",
                            "split": "test", "baseline": "null",
                        }), historical_available=True, forward_available=True,
                    ))

    def test_interrupted_or_forged_clarifications_fail_closed(self):
        good = _history(MISSING[0])
        invalid = (
            [], good[:1],
            [*good, {"role": "user", "content": "Nová otázka."}],
            [{**good[0], "content": "Koľko dní model v3 prekonal baseline?"}, good[1]],
            [good[0], {**good[1], "source": "model"}],
            [good[0], {**good[1], "source": "historical_v3_reports"}],
            [good[0], {**good[1], "content": "Validácia alebo lockbox?"}],
            [good[0], {**good[1], "evidence_question": MISSING[0]}],
            [{**good[0], "role": "assistant"}, good[1]],
            [good[0], {**good[1], "role": "user"}],
            [good[0], {**good[1], "content": good[1]["content"] + " "}],
        )
        for history in invalid:
            with self.subTest(history=history):
                self.assertIsNone(historical_followup_question("Validáciu.", history))
                self.assertIsNone(historical_followup_question("Lockbox.", history))
        for reply in (
            "Validáciu a obchoduj", "Lockbox vo forward denníku",
            "Lockbox alebo validáciu", "Validáciu, potom predaj",
            "Nie validáciu, ale lockbox", "Lockbox na NVDA",
            "Validáciu a koľko zarobím?", "Lockbox a zmeň model",
        ):
            with self.subTest(reply=reply):
                self.assertIsNone(bare_historical_split_reply_target(reply))
                self.assertIsNone(historical_followup_question(reply, good))

    def test_both_runtimes_reread_synthetic_reports_without_qwen(self):
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
            cases = (
                (MISSING[0], "Validáciu.", "validation", 2, False),
                (MISSING[1], "Lockbox.", "test", 2, False),
                (MISSING[2], "Lockbox test.", "test", 1, True),
                (MISSING[3], "Historickú validáciu.", "validation", 3, True),
            )
            for runtime in (ollama, transformers):
                for question, reply, split, wins, share in cases:
                    with self.subTest(runtime=type(runtime).__name__, question=question):
                        with patch.object(ollama, "_generate",
                                          side_effect=AssertionError("Qwen must not run")):
                            clarification = runtime.respond(question)
                            self.assertEqual(
                                clarification.answer_source, "conversation_clarification"
                            )
                            history = [
                                {"role": "user", "content": question},
                                {"role": "assistant", "content": clarification.text,
                                 "source": clarification.answer_source,
                                 "evidence_question": clarification.evidence_question or ""},
                            ]
                            answer = runtime.respond(reply, history=history)
                        self.assertEqual(answer.answer_source, "historical_v3_reports")
                        self.assertIn(f"v {wins} z 3 trhových dní", answer.text)
                        self.assertEqual("%" in answer.text, share)
                        self.assertIn(
                            "v historickej validácii" if split == "validation"
                            else "na historickom lockbox teste",
                            answer.evidence_question,
                        )
                        self.assertIn("nie je presnosť smeru", answer.text.lower())


if __name__ == "__main__":
    unittest.main()
