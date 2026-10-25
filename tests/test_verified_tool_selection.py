"""Experimental verified tool selection never trusts Qwen with financial facts."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ourai_language.conversation_followups import historical_followup_question
from ourai_language.forward_answers import forward_question_kind
from ourai_language.historical_answers import (
    historical_question_kind, historical_source_correction_target, maybe_historical_answer,
)
from ourai_language.semantic_router import canonical_historical_question
from ourai_language.ollama_runtime import LocalOllamaChatRuntime
from ourai_language.verified_tool_router import select_verified_tool
from scripts.chat_ourai import parser


def choice(tool="none", action="none", split="both", baseline="both"):
    return json.dumps({
        "tool": tool, "action": action, "split": split, "baseline": baseline,
    })


def validation_fixture(directory: Path):
    payload = {
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
    (directory / "nonlinear_magnitude_validation_v1.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )


class VerifiedToolSelectionTests(unittest.TestCase):
    def test_day_win_share_split_switch_is_verified_and_context_bound(self):
        cases = (
            ("v historickej validácii", "A na lockboxe?",
             "na historickom lockbox teste", "compare_test"),
            ("na historickom lockbox teste", "A vo validácii?",
             "v historickej validácii", "compare_validation"),
        )
        for old_scope, question, new_scope, kind in cases:
            for baseline in ("lineárny", "nulový"):
                previous = (
                    f"Koľko percent trhových dní model v3 prekonal "
                    f"{baseline} baseline {old_scope}?"
                )
                history = [
                    {"role": "user", "content": previous},
                    {"role": "assistant", "content": "Overená historická odpoveď.",
                     "source": "historical_v3_reports",
                     "evidence_question": previous},
                ]
                with self.subTest(question=question, baseline=baseline):
                    canonical = historical_followup_question(question, history)
                    self.assertEqual(
                        canonical,
                        f"Koľko percent trhových dní model v3 prekonal "
                        f"{baseline} baseline {new_scope}?",
                    )
                    self.assertEqual(historical_question_kind(canonical), kind)
                    # Neither deterministic nor opt-in semantic selection may
                    # open a report from an isolated elliptical question.
                    self.assertIsNone(historical_question_kind(question))
                    self.assertIsNone(maybe_historical_answer(question, "missing"))
                    self.assertIsNone(canonical_historical_question(
                        question, json.dumps({
                            "intent": "compare", "split": "test",
                            "baseline": "linear",
                        }),
                    ))
                    self.assertIsNone(select_verified_tool(
                        question, choice("historical_v3", "compare", "test", "linear"),
                        historical_available=True, forward_available=True,
                    ))
                    for invalid in (
                        [], history[:-1],
                        [history[0], {**history[1], "source": "model"}],
                        [history[0], {**history[1],
                                      "source": "verified_forward_journal"}],
                        history + [{"role": "user", "content": "Iná otázka."}],
                    ):
                        self.assertIsNone(historical_followup_question(question, invalid))
                    for unsafe in (
                        question[:-1] + " a obchoduj?",
                        question[:-1] + " vo forward denníku?",
                        "A na lockboxe aj vo validácii?",
                    ):
                        self.assertIsNone(historical_followup_question(unsafe, history))
                    # No redundant split switch and no inferred share after
                    # a prior day count (or a two-split percentage).
                    self.assertIsNone(historical_followup_question(
                        "A vo validácii?" if "validácii" in old_scope
                        else "A na lockboxe?", history,
                    ))
                    count = previous.replace("Koľko percent trhových dní", "Koľko dní")
                    count_history = [
                        {"role": "user", "content": count},
                        {**history[1], "evidence_question": count},
                    ]
                    self.assertIsNone(historical_followup_question(
                        question, count_history,
                    ))
                    both = previous.replace(old_scope, "v historickej validácii a lockboxe")
                    self.assertIsNone(historical_followup_question(
                        question, [
                            {"role": "user", "content": both},
                            {**history[1], "evidence_question": both},
                        ],
                    ))

    def test_day_win_share_split_switch_rereads_target_report(self):
        previous = (
            "Koľko percent trhových dní model v3 prekonal "
            "lineárny baseline v historickej validácii?"
        )
        history = [
            {"role": "user", "content": previous},
            {"role": "assistant", "content": "Overený podiel.",
             "source": "historical_v3_reports", "evidence_question": previous},
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            validation_fixture(root)
            source = json.loads(
                (root / "nonlinear_magnitude_validation_v1.json").read_text(
                    encoding="utf-8"
                )
            )
            test = {
                **source, "test_lockbox_accessed": True,
                "post_test_tuning_allowed": False, "test_supported": True,
                "test": {
                    **source["validation"], "market_date_count": 5,
                    "market_date_wins_vs_linear": 1,
                    "market_date_wins_vs_null": 4,
                },
            }
            test.pop("validation")
            (root / "nonlinear_magnitude_test_lockbox_v1.json").write_text(
                json.dumps(test), encoding="utf-8",
            )
            runtime = LocalOllamaChatRuntime(
                historical_reports_dir=root, semantic_tool_routing=True,
                knowledge_enabled=False,
            )
            with patch.object(runtime, "_generate",
                              side_effect=AssertionError("model must not run")):
                answer = runtime.respond("A na lockboxe?", history=history)
            self.assertEqual(answer.answer_source, "historical_v3_reports")
            self.assertIn("20,00 % sledovaných trhových dní", answer.text)
            self.assertIn("v 1 z 5 trhových dní", answer.text)
            self.assertNotIn("V historickej validácii", answer.text)
            self.assertNotIn("nulovému baseline", answer.text)
            self.assertIn("nie je presnosť smeru UP/DOWN", answer.text)
            self.assertIn("na historickom lockbox teste", answer.evidence_question)

            missing = LocalOllamaChatRuntime(
                historical_reports_dir=root / "missing",
                semantic_tool_routing=True, knowledge_enabled=False,
            )
            with patch.object(missing, "_generate",
                              side_effect=AssertionError("model must not run")):
                unavailable = missing.respond("A na lockboxe?", history=history)
            self.assertEqual(unavailable.answer_source, "historical_v3_reports")
            self.assertIn("nie je dostupný", unavailable.text)

    def test_day_wins_metric_switch_preserves_verified_historical_split(self):
        cases = (
            ("Dobre, ale teraz mi namiesto MAE ukáž, koľko dní model prekonal lineárny baseline.",
             "lineárny baseline"),
            ("A koľko dní model prekonal nulový baseline?", "nulový baseline"),
        )
        for previous, scope, kind in (
            ("Porovnaj historickú validáciu v3 s lineárnym modelom.",
             "v historickej validácii", "compare_validation"),
            ("Porovnaj historický lockbox test v3 s lineárnym modelom.",
             "na historickom lockbox teste", "compare_test"),
        ):
            history = [
                {"role": "user", "content": previous},
                {"role": "assistant", "content": "Overený historický report.",
                 "source": "historical_v3_reports", "evidence_question": previous},
            ]
            for question, baseline in cases:
                with self.subTest(previous=previous, question=question):
                    canonical = historical_followup_question(question, history)
                    self.assertIsNotNone(canonical)
                    self.assertIn(scope, canonical)
                    self.assertIn(baseline, canonical)
                    self.assertEqual(historical_question_kind(canonical), kind)
                    self.assertIsNone(historical_question_kind(question))
                    self.assertIsNone(select_verified_tool(
                        question, choice("historical_v3", "compare"),
                        historical_available=True, forward_available=True,
                    ))
                    for invalid in (
                        [], history[:-1],
                        [history[0], {**history[1], "source": "model"}],
                        [history[0], {**history[1], "source": "verified_forward_journal"}],
                        history + [{"role": "user", "content": "A čo Python?"}],
                    ):
                        self.assertIsNone(historical_followup_question(question, invalid))
        # A verified two-report overview is not permission to invent a split.
        both = [
            {"role": "user", "content": "Ako presný bol historicky model v3?"},
            {"role": "assistant", "content": "Overené výsledky.",
             "source": "historical_v3_reports",
             "evidence_question": "Ako presný bol historicky model v3?"},
        ]
        self.assertIsNone(historical_followup_question(cases[0][0], both))
        for unsafe in (
            "A koľko dní model prekonal nulový baseline vo forward denníku?",
            "Dobre, ale teraz mi namiesto MAE ukáž, koľko dní model prekonal lineárny baseline a obchoduj.",
        ):
            self.assertIsNone(historical_followup_question(unsafe, history))

    def test_day_wins_followup_reads_observed_counts_not_directional_accuracy(self):
        previous = "Porovnaj historickú validáciu v3 s lineárnym modelom."
        history = [
            {"role": "user", "content": previous},
            {"role": "assistant", "content": "Overené historické porovnanie.",
             "source": "historical_v3_reports", "evidence_question": previous},
        ]
        with tempfile.TemporaryDirectory() as directory:
            validation_fixture(Path(directory))
            runtime = LocalOllamaChatRuntime(
                historical_reports_dir=directory, semantic_tool_routing=True,
                knowledge_enabled=False,
            )
            for question, wins in (
                ("Dobre, ale teraz mi namiesto MAE ukáž, koľko dní model prekonal lineárny baseline.",
                 "v 2 z 3 trhových dní"),
                ("A koľko dní model prekonal nulový baseline?", "v 3 z 3 trhových dní"),
            ):
                with self.subTest(question=question):
                    with patch.object(runtime, "_generate",
                                      side_effect=AssertionError("model must not run")):
                        result = runtime.respond(question, history=history)
                    self.assertEqual(result.answer_source, "historical_v3_reports")
                    self.assertIn(wins, result.text)
                    self.assertNotIn("10,00 bps", result.text)
                    self.assertIn("nie je presnosť smeru UP/DOWN", result.text)

    def test_explicit_historical_day_win_percentages_are_not_directional_accuracy(self):
        # Synthetic verified report: 2/3 days vs linear, 3/3 vs null.
        # Neither fraction measures correct UP/DOWN predictions.
        with tempfile.TemporaryDirectory() as directory:
            validation_fixture(Path(directory))
            cases = (
                ("Koľko dní model v3 prekonal lineárny baseline v historickej validácii?",
                 "v 2 z 3 trhových dní", None),
                ("Koľko percent trhových dní model v3 prekonal lineárny baseline v historickej validácii?",
                 "66,67 % sledovaných trhových dní", "nulovému baseline"),
                ("Aké percento dní model v3 prekonal nulový baseline v historickej validácii?",
                 "100,00 % sledovaných trhových dní", "lineárnemu baseline"),
            )
            for question, expected, excluded in cases:
                with self.subTest(question=question):
                    self.assertEqual(historical_question_kind(question), "compare_validation")
                    result = maybe_historical_answer(question, directory)
                    self.assertIsNotNone(result)
                    self.assertIn(expected, result)
                    self.assertIn("nie je presnosť smeru UP/DOWN", result)
                    self.assertNotIn("8,00 bps", result)
                    if excluded:
                        self.assertNotIn(excluded, result)
            # A complete request is handled by the deterministic verified
            # reader without model generation or the opt-in classifier.
            runtime = LocalOllamaChatRuntime(
                historical_reports_dir=directory, semantic_tool_routing=True,
                knowledge_enabled=False,
            )
            with patch.object(runtime, "_generate",
                              side_effect=AssertionError("model must not run")):
                answer = runtime.respond(cases[1][0])
            self.assertEqual(answer.answer_source, "historical_v3_reports")
            self.assertIn("66,67 % sledovaných trhových dní", answer.text)
            # Without a report, never calculate a percentage from a guess.
            self.assertIn(
                "nie je dostupný",
                maybe_historical_answer(cases[1][0], Path(directory) / "missing"),
            )

    def test_day_win_share_followup_requires_verified_count_and_single_split(self):
        questions = (
            "A koľko je to percent?",
            "Koľko percent to je?",
            "A aké percento to je?",
            "Koľko je to v percentách?",
        )
        for scope, baseline, expected in (
            ("v historickej validácii", "lineárny baseline", "compare_validation"),
            ("v historickej validácii", "nulový baseline", "compare_validation"),
            ("na historickom lockbox teste", "lineárny baseline", "compare_test"),
            ("na historickom lockbox teste", "nulový baseline", "compare_test"),
        ):
            previous = f"Koľko dní model v3 prekonal {baseline} {scope}?"
            history = [
                {"role": "user", "content": previous},
                {"role": "assistant", "content": "Overený počet trhových dní.",
                 "source": "historical_v3_reports", "evidence_question": previous},
            ]
            for question in questions:
                with self.subTest(previous=previous, question=question):
                    canonical = historical_followup_question(question, history)
                    self.assertEqual(
                        canonical,
                        f"Koľko percent trhových dní model v3 prekonal {baseline} {scope}?",
                    )
                    self.assertEqual(historical_question_kind(canonical), expected)
                    # Neither deterministic nor optional model selection may
                    # open a report from the standalone elliptical question.
                    self.assertIsNone(historical_question_kind(question))
                    self.assertIsNone(select_verified_tool(
                        question, choice("historical_v3", "compare"),
                        historical_available=True, forward_available=True,
                    ))
                    self.assertIsNone(historical_followup_question(question, []))
                    self.assertIsNone(historical_followup_question(
                        question, [history[0], {**history[1], "source": "model"}],
                    ))
                    self.assertIsNone(historical_followup_question(
                        question, [history[0], {**history[1],
                                               "source": "verified_forward_journal"}],
                    ))
                    self.assertIsNone(historical_followup_question(
                        question, history + [{"role": "user", "content": "A čo Python?"}],
                    ))

        for previous in (
            "Porovnaj historickú validáciu v3 s lineárnym modelom.",
            "Aká bola MAE modelu v3 v historickej validácii?",
            "Ako presný bol historicky model v3?",
            "Koľko percent trhových dní model v3 prekonal lineárny baseline v historickej validácii?",
        ):
            history = [
                {"role": "user", "content": previous},
                {"role": "assistant", "content": "Overené historické výsledky.",
                 "source": "historical_v3_reports", "evidence_question": previous},
            ]
            self.assertIsNone(historical_followup_question("A koľko je to percent?", history))
        self.assertIsNone(historical_followup_question(
            "A koľko je to percent vo forward denníku?", history,
        ))
        self.assertIsNone(historical_followup_question(
            "A koľko je to percent a kúp NVDA?", history,
        ))

    def test_day_win_share_followup_reads_only_verified_daily_counts(self):
        previous = (
            "Koľko dní model v3 prekonal lineárny baseline "
            "v historickej validácii?"
        )
        history = [
            {"role": "user", "content": previous},
            {"role": "assistant", "content": "V 2 z 3 trhových dní.",
             "source": "historical_v3_reports", "evidence_question": previous},
        ]
        with tempfile.TemporaryDirectory() as directory:
            validation_fixture(Path(directory))
            runtime = LocalOllamaChatRuntime(
                historical_reports_dir=directory, semantic_tool_routing=True,
                knowledge_enabled=False,
            )
            with patch.object(runtime, "_generate",
                              side_effect=AssertionError("model must not run")):
                result = runtime.respond("A koľko je to percent?", history=history)
            self.assertEqual(result.answer_source, "historical_v3_reports")
            self.assertIn("66,67 % sledovaných trhových dní", result.text)
            self.assertIn("v 2 z 3 trhových dní", result.text)
            self.assertNotIn("nulovému baseline", result.text)
            self.assertIn("nie je presnosť smeru UP/DOWN", result.text)
            self.assertEqual(
                result.evidence_question,
                "Koľko percent trhových dní model v3 prekonal "
                "lineárny baseline v historickej validácii?",
            )
            missing = LocalOllamaChatRuntime(
                historical_reports_dir=Path(directory) / "missing",
                semantic_tool_routing=True, knowledge_enabled=False,
            )
            with patch.object(missing, "_generate",
                              side_effect=AssertionError("model must not run")):
                unavailable = missing.respond("A koľko je to percent?", history=history)
            self.assertEqual(unavailable.answer_source, "historical_v3_reports")
            self.assertIn("nie je dostupný", unavailable.text)

    def test_day_win_share_baseline_switch_requires_verified_percentage(self):
        for scope, kind in (
            ("v historickej validácii", "compare_validation"),
            ("na historickom lockbox teste", "compare_test"),
        ):
            for old, new, question in (
                ("lineárny", "nulový", "A pri nulovom?"),
                ("nulový", "lineárny", "A pri lineárnom modeli?"),
            ):
                previous = (
                    f"Koľko percent trhových dní model v3 prekonal "
                    f"{old} baseline {scope}?"
                )
                history = [
                    {"role": "user", "content": previous},
                    {"role": "assistant", "content": "Overené percento dní.",
                     "source": "historical_v3_reports", "evidence_question": previous},
                ]
                with self.subTest(scope=scope, question=question):
                    canonical = historical_followup_question(question, history)
                    self.assertEqual(
                        canonical,
                        f"Koľko percent trhových dní model v3 prekonal "
                        f"{new} baseline {scope}?",
                    )
                    self.assertEqual(historical_question_kind(canonical), kind)
                    self.assertIsNone(historical_question_kind(question))
                    self.assertIsNone(maybe_historical_answer(question, "missing"))
                    self.assertIsNone(select_verified_tool(
                        question, choice("historical_v3", "compare"),
                        historical_available=True, forward_available=True,
                    ))
                    for invalid in (
                        [], history[:-1],
                        [history[0], {**history[1], "source": "model"}],
                        [history[0], {**history[1], "source": "verified_forward_journal"}],
                        history + [{"role": "user", "content": "A čo Python?"}],
                    ):
                        self.assertIsNone(historical_followup_question(question, invalid))
                    for unsafe in (
                        "A pri nulovom a obchoduj?",
                        "A pri nulovom vo forward denníku?",
                        "A pri nulovom aj lineárnom?",
                    ):
                        self.assertIsNone(historical_followup_question(unsafe, history))

        for previous in (
            "Koľko dní model v3 prekonal lineárny baseline v historickej validácii?",
            "Porovnaj historickú validáciu v3 s lineárnym modelom.",
            "Ako presný bol historicky model v3?",
            "Koľko percent trhových dní model v3 prekonal lineárny baseline?",
        ):
            history = [
                {"role": "user", "content": previous},
                {"role": "assistant", "content": "Overená odpoveď.",
                 "source": "historical_v3_reports", "evidence_question": previous},
            ]
            self.assertIsNone(historical_followup_question("A pri nulovom?", history))

    def test_day_win_share_baseline_switch_rereads_verified_report(self):
        previous = (
            "Koľko percent trhových dní model v3 prekonal "
            "lineárny baseline v historickej validácii?"
        )
        history = [
            {"role": "user", "content": previous},
            {"role": "assistant", "content": "Overené percento dní.",
             "source": "historical_v3_reports", "evidence_question": previous},
        ]
        with tempfile.TemporaryDirectory() as directory:
            validation_fixture(Path(directory))
            runtime = LocalOllamaChatRuntime(
                historical_reports_dir=directory, semantic_tool_routing=True,
                knowledge_enabled=False,
            )
            with patch.object(runtime, "_generate",
                              side_effect=AssertionError("model must not run")):
                result = runtime.respond("A pri nulovom?", history=history)
            self.assertEqual(result.answer_source, "historical_v3_reports")
            self.assertIn("100,00 % sledovaných trhových dní", result.text)
            self.assertIn("v 3 z 3 trhových dní", result.text)
            self.assertNotIn("lineárnemu baseline", result.text)
            self.assertIn("nie je presnosť smeru UP/DOWN", result.text)
            self.assertEqual(
                result.evidence_question,
                "Koľko percent trhových dní model v3 prekonal "
                "nulový baseline v historickej validácii?",
            )
            missing = LocalOllamaChatRuntime(
                historical_reports_dir=Path(directory) / "missing",
                semantic_tool_routing=True, knowledge_enabled=False,
            )
            with patch.object(missing, "_generate",
                              side_effect=AssertionError("model must not run")):
                unavailable = missing.respond("A pri nulovom?", history=history)
            self.assertEqual(unavailable.answer_source, "historical_v3_reports")
            self.assertIn("nie je dostupný", unavailable.text)

    def test_now_baseline_followup_preserves_verified_historical_split(self):
        # "Teraz" is a discourse connector here, not a live-data request.
        for prior_question, expected_split in (
            ("Porovnaj historickú validáciu v3 s lineárnym modelom.",
             "v historickej validácii"),
            ("Porovnaj historický lockbox test v3 s lineárnym modelom.",
             "na historickom lockbox teste"),
        ):
            history = [
                {"role": "user", "content": prior_question},
                {"role": "assistant", "content": "Overené historické výsledky.",
                 "source": "historical_v3_reports",
                 "evidence_question": prior_question},
            ]
            for question, baseline in (
                ("A teraz to porovnaj s nulovým.", "nulovým baseline"),
                ("A teraz to porovnaj s nulovým baseline.", "nulovým baseline"),
                ("Teraz to porovnaj s lineárnym modelom.", "lineárnym modelom"),
            ):
                with self.subTest(prior=prior_question, question=question):
                    canonical = historical_followup_question(question, history)
                    self.assertIsNotNone(canonical)
                    self.assertIn(expected_split, canonical)
                    self.assertIn(baseline, canonical)
                    self.assertEqual(
                        historical_question_kind(canonical),
                        "compare_validation" if "validácii" in expected_split
                        else "compare_test",
                    )
                    # Standalone requests cannot use a historical reader.
                    self.assertIsNone(historical_question_kind(question))
                    self.assertIsNone(select_verified_tool(
                        question, choice("historical_v3", "compare"),
                        historical_available=True, forward_available=True,
                    ))
                    for invalid_history in (
                        [], history[:-1],
                        [history[0], {**history[1], "source": "model"}],
                        [history[0], {**history[1], "source": "verified_forward_journal"}],
                        history + [{"role": "user", "content": "A čo Python?"}],
                    ):
                        self.assertIsNone(historical_followup_question(
                            question, invalid_history,
                        ))
            for question in (
                "A teraz to porovnaj s nulovým na forward dátach.",
                "A teraz to porovnaj s nulovým pre live obchodovanie.",
                "A teraz ukáž aktuálnu cenu.",
                "A teraz to neporovnávaj s nulovým.",
            ):
                with self.subTest(rejected=question):
                    self.assertIsNone(historical_followup_question(question, history))

    def test_exclusive_baseline_corrections_require_verified_historical_context(self):
        cases = (
            ("Nie, teraz to porovnaj iba s nulovým modelom, nie s lineárnym.",
             "nulovým baseline", "lineárnym modelom"),
            ("Nie, teraz to porovnaj iba s lineárnym modelom, nie s nulovým.",
             "lineárnym modelom", "nulovým baseline"),
            ("A teraz to porovnaj len s nulovým baseline, nie s lineárnym.",
             "nulovým baseline", "lineárnym modelom"),
        )
        for previous, scope in (
            ("Porovnaj historickú validáciu v3 s lineárnym modelom.",
             "v historickej validácii"),
            ("Porovnaj historický lockbox test v3 s lineárnym modelom.",
             "na historickom lockbox teste"),
        ):
            history = [
                {"role": "user", "content": previous},
                {"role": "assistant", "content": "Overený historický report.",
                 "source": "historical_v3_reports", "evidence_question": previous},
            ]
            for question, wanted, excluded in cases:
                with self.subTest(previous=previous, question=question):
                    canonical = historical_followup_question(question, history)
                    self.assertIsNotNone(canonical)
                    self.assertIn(scope, canonical)
                    self.assertIn(wanted, canonical)
                    self.assertNotIn(excluded, canonical)
                    self.assertEqual(
                        historical_question_kind(canonical),
                        "compare_validation" if "validácii" in scope else "compare_test",
                    )
                    # No context may authorize the deterministic or opt-in
                    # verified report readers, even with a bad model proposal.
                    self.assertIsNone(historical_question_kind(question))
                    self.assertIsNone(select_verified_tool(
                        question, choice("historical_v3", "compare"),
                        historical_available=True, forward_available=True,
                    ))
                    for invalid in (
                        [], history[:-1],
                        [history[0], {**history[1], "source": "model"}],
                        [history[0], {**history[1], "source": "verified_forward_journal"}],
                        history + [{"role": "user", "content": "A čo Python?"}],
                    ):
                        self.assertIsNone(historical_followup_question(question, invalid))

            for unsafe in (
                "Nie, teraz to porovnaj iba s nulovým, nie s nulovým.",
                "Nie, teraz to porovnaj iba s nulovým, nie s lineárnym, a obchoduj.",
                "Nie, teraz to porovnaj iba s nulovým, nie s lineárnym na forward dátach.",
                "Nie, teraz to neporovnávaj s nulovým.",
            ):
                self.assertIsNone(historical_followup_question(unsafe, history))
                self.assertIsNone(historical_question_kind(unsafe))

    def test_exclusive_baseline_followup_reads_only_requested_baseline(self):
        previous = "Porovnaj historickú validáciu v3 s lineárnym modelom."
        history = [
            {"role": "user", "content": previous},
            {"role": "assistant", "content": "Overené historické porovnanie.",
             "source": "historical_v3_reports", "evidence_question": previous},
        ]
        with tempfile.TemporaryDirectory() as directory:
            validation_fixture(Path(directory))
            runtime = LocalOllamaChatRuntime(
                historical_reports_dir=directory, semantic_tool_routing=True,
                knowledge_enabled=False,
            )
            for question, included, excluded in (
                ("Nie, teraz to porovnaj iba s nulovým modelom, nie s lineárnym.",
                 "nulový baseline", "lineárny baseline"),
                ("Nie, teraz to porovnaj iba s lineárnym modelom, nie s nulovým.",
                 "lineárny baseline", "nulový baseline"),
            ):
                with self.subTest(question=question):
                    with patch.object(runtime, "_generate",
                                      side_effect=AssertionError("model must not run")):
                        result = runtime.respond(question, history=history)
                    self.assertEqual(result.answer_source, "historical_v3_reports")
                    self.assertIn(included, result.text)
                    self.assertNotIn(excluded, result.text)

    def test_historical_source_corrections_require_verified_immediate_context(self):
        # The old lockbox answer authorizes a correction, not reuse of its
        # metrics. The new question is read from the requested validation file.
        lockbox_history = [
            {"role": "user", "content": "Ukáž výsledky lockbox testu."},
            {"role": "assistant", "content": "Overený lockbox report.",
             "source": "historical_v3_reports",
             "evidence_question": "Aký bol historický lockbox test v3?"},
        ]
        validation_history = [
            {"role": "user", "content": "Ukáž výsledky validácie."},
            {"role": "assistant", "content": "Overený validačný report.",
             "source": "historical_v3_reports",
             "evidence_question": "Aké boli výsledky historickej validácie v3?"},
        ]
        cases = (
            ("Nie, myslel som validáciu.", lockbox_history,
             "Aké boli výsledky historickej validácie v3?", "validation"),
            ("Mala som na mysli historickú validáciu.", lockbox_history,
             "Aké boli výsledky historickej validácie v3?", "validation"),
            ("Nie, myslel som lockbox test.", validation_history,
             "Aký bol historický lockbox test v3?", "test"),
        )
        for question, history, canonical, split in cases:
            with self.subTest(question=question):
                self.assertEqual(historical_source_correction_target(question), split)
                self.assertEqual(historical_followup_question(question, history), canonical)
                self.assertEqual(historical_question_kind(canonical), split)
                self.assertIsNone(historical_question_kind(question))
                self.assertIsNone(select_verified_tool(
                    question, choice("historical_v3", split),
                    historical_available=True, forward_available=True,
                ))
                for invalid_history in (
                    [],
                    history[:-1],
                    [history[0], {**history[1], "source": "model"}],
                    [history[0], {**history[1], "source": "verified_forward_journal"}],
                    history + [{"role": "user", "content": "A čo Python?"}],
                ):
                    self.assertIsNone(historical_followup_question(
                        question, invalid_history,
                    ))

        # A correction naming multiple sources or changing topic is not an
        # authorization to select one report without explicit clarification.
        for ambiguous in (
            "Nie, myslel som validáciu aj lockbox.",
            "Nie, myslel som validáciu modelu v3 a forward denník.",
            "Nie, myslel som historické výsledky Pythonu.",
        ):
            with self.subTest(ambiguous=ambiguous):
                self.assertEqual(historical_source_correction_target(ambiguous), "ambiguous")
                self.assertIsNone(historical_followup_question(ambiguous, lockbox_history))
                self.assertIsNone(historical_question_kind(ambiguous))
                self.assertIsNone(canonical_historical_question(
                    ambiguous, json.dumps({
                        "intent": "validation", "split": "validation", "baseline": "both",
                    }),
                ))

    def test_corrected_validation_reads_only_verified_validation_report(self):
        prior = [
            {"role": "user", "content": "Ukáž výsledky lockbox testu."},
            {"role": "assistant", "content": "Historický test lockbox.",
             "source": "historical_v3_reports",
             "evidence_question": "Aký bol historický lockbox test v3?"},
        ]
        with tempfile.TemporaryDirectory() as directory:
            validation_fixture(Path(directory))
            runtime = LocalOllamaChatRuntime(
                historical_reports_dir=directory, semantic_tool_routing=True,
                knowledge_enabled=False,
            )
            # No Qwen model is involved: a verified source switch must happen
            # before the optional model-assisted router.
            with patch.object(runtime, "_generate",
                              side_effect=AssertionError("model must not run")):
                result = runtime.respond("Nie, myslel som validáciu.", history=prior)
            self.assertEqual(result.answer_source, "historical_v3_reports")
            self.assertEqual(result.evidence_question,
                             "Aké boli výsledky historickej validácie v3?")
            self.assertIn("Historická validácia v3", result.text)
            self.assertNotIn("Historický test lockbox", result.text)
            self.assertIsNone(maybe_historical_answer(
                "Nie, myslel som validáciu.", directory,
            ))

    def test_model_may_choose_only_preapproved_tool_and_operation(self):
        question = "Ako sa ti darí v shadow režime?"
        self.assertEqual(
            select_verified_tool(
                question, choice("forward_v2", "performance"),
                historical_available=True, forward_available=True,
            ),
            ("forward_v2", "Aká je presnosť predikcií vo forward denníku?"),
        )
        for suggestion in (
            "bad json",
            json.dumps({"tool": "forward_v2", "action": "performance",
                        "split": "both", "baseline": "both", "file": "/secrets"}),
            choice("execute_trade", "performance"),
            choice("forward_v2", "buy"),
            choice("forward_v2", "performance", "test"),
            choice("historical_v3", "performance"),
            choice("forward_v2", "performance") + " extra",
        ):
            with self.subTest(suggestion=suggestion):
                self.assertIsNone(select_verified_tool(
                    question, suggestion,
                    historical_available=True, forward_available=True,
                ))

    def test_forward_source_must_be_user_anchored_not_model_invented(self):
        candidate = choice("forward_v2", "performance")
        for user_question in (
            "Čo je paper trading?",
            "Ako sa máš?",
            "What is a forward test?",
            "Ako dopadla historická validácia?",
            "Porovnaj shadow s lockbox testom.",
            "Kúp NVDA, keď sa ti darí v shadow režime.",
            "Koľko je hodín?",
        ):
            with self.subTest(question=user_question):
                self.assertIsNone(select_verified_tool(
                    user_question, candidate,
                    historical_available=True, forward_available=True,
                ))
        self.assertIsNone(select_verified_tool(
            "Ako sa ti darí v shadow režime?", candidate,
            historical_available=True, forward_available=False,
        ))

    def test_polite_educational_forward_questions_never_open_reader(self):
        # The model's tool proposal cannot turn a definition into recorded
        # forward evidence just because the user mentions "shadow"/"model".
        proposal = choice("forward_v2", "performance")
        for question in (
            "Môžeš mi vysvetliť, čo je shadow model?",
            "Vieš mi vysvetliť, ako funguje forward model?",
            "Dokážeš vysvetliť, čo znamená paper trading model?",
            "Môžeš mi prosím vysvetliť, na čo slúži forward denník?",
        ):
            with self.subTest(question=question):
                self.assertIsNone(select_verified_tool(
                    question, proposal,
                    historical_available=True, forward_available=True,
                ))
        # A question about observed performance is different from a definition.
        self.assertEqual(
            select_verified_tool(
                "Vieš mi vysvetliť, ako sa ti darí v shadow režime?",
                proposal, historical_available=True, forward_available=True,
            ),
            ("forward_v2", "Aká je presnosť predikcií vo forward denníku?"),
        )

    def test_forward_definitions_do_not_open_deterministic_or_opt_in_reader(self):
        # The deterministic reader runs before the optional Qwen tool router.
        # Both must enforce the same educational boundary.
        proposal = choice("forward_v2", "performance")
        for question in (
            "Čo znamená MAE vo forward denníku?",
            "Ako funguje forward model?",
            "Prosím, vysvetli mi, čo je forward model.",
            "Môžeš mi vysvetliť, ako funguje forward model?",
            "Vieš mi prosím vysvetliť, na čo slúži forward denník?",
            "Explain what forward model means.",
            "Vysvetli prosím forward predikcie OurAI.",
            "Can you explain forward model performance?",
        ):
            with self.subTest(question=question):
                self.assertIsNone(forward_question_kind(question))
                self.assertIsNone(select_verified_tool(
                    question, proposal,
                    historical_available=True, forward_available=True,
                ))

    def test_explanation_of_observed_forward_results_is_not_definition(self):
        proposal = choice("forward_v2", "performance")
        for question in (
            "Vysvetli mi výsledky vo forward denníku.",
            "Môžeš mi vysvetliť, ako sa ti darí v shadow režime?",
        ):
            with self.subTest(question=question):
                self.assertEqual(
                    select_verified_tool(
                        question, proposal,
                        historical_available=True, forward_available=True,
                    )[0],
                    "forward_v2",
                )
        self.assertEqual(
            forward_question_kind("Vysvetli mi výsledky vo forward denníku."),
            "performance",
        )
        self.assertEqual(
            forward_question_kind("Aká je MAE vo forward denníku?"),
            "performance",
        )

    def test_educational_question_does_not_touch_missing_forward_journal(self):
        with tempfile.TemporaryDirectory() as temp:
            missing = Path(temp) / "missing.jsonl"
            runtime = LocalOllamaChatRuntime(
                forward_journal=missing, semantic_tool_routing=True,
                knowledge_enabled=False,
            )
            with patch.object(runtime, "_generate", side_effect=[
                choice("forward_v2", "performance"),
                "MAE je priemerná absolútna chyba.",
            ]) as generated:
                result = runtime.respond("Čo znamená MAE vo forward denníku?")
            self.assertEqual(result.answer_source, "model")
            self.assertIn("priemerná absolútna chyba", result.text)
            self.assertEqual(generated.call_count, 2)
            self.assertFalse(missing.exists())

    def test_historical_definitions_do_not_open_verified_reports(self):
        # Even a bad model proposal must not turn a definition into a
        # measured retrospective result, and the deterministic route must
        # apply the same boundary before the optional model router.
        proposal = choice("historical_v3", "overview")
        for question in (
            "Čo znamená historická MAE?",
            "Čo znamená MAE v historickej validácii modelu v3?",
            "Ako funguje lockbox test?",
            "Môžeš mi vysvetliť, čo znamená historická MAE modelu v3?",
            "Prosím, vysvetli mi, čo je historický baseline OurAI.",
            "Can you explain what historical MAE means for model v3?",
        ):
            with self.subTest(question=question):
                self.assertIsNone(historical_question_kind(question))
                self.assertIsNone(canonical_historical_question(
                    question, json.dumps({
                        "intent": "overview", "split": "both", "baseline": "both",
                    }),
                ))
                self.assertIsNone(select_verified_tool(
                    question, proposal,
                    historical_available=True, forward_available=True,
                ))

    def test_historical_measured_results_still_open_verified_reader(self):
        proposal = choice("historical_v3", "overview")
        for question in (
            "Aká je historická MAE modelu v3?",
            "Aké boli výsledky historickej validácie v3?",
            "What is the historical MAE of model v3?",
        ):
            with self.subTest(question=question):
                self.assertIsNotNone(historical_question_kind(question))
                self.assertEqual(select_verified_tool(
                    question, proposal,
                    historical_available=True, forward_available=True,
                )[0], "historical_v3")
        self.assertEqual(
            historical_question_kind("Vysvetli mi výsledky historickej validácie v3"),
            "validation",
        )

    def test_historical_definition_skips_missing_report_reader(self):
        with tempfile.TemporaryDirectory() as temp:
            missing = Path(temp) / "missing-reports"
            runtime = LocalOllamaChatRuntime(
                historical_reports_dir=missing, semantic_tool_routing=True,
                knowledge_enabled=False,
            )
            with patch.object(runtime, "_generate", side_effect=[
                choice("historical_v3", "overview"),
                "MAE znamená priemernú absolútnu chybu.",
            ]) as generated:
                result = runtime.respond(
                    "Čo znamená MAE v historickej validácii modelu v3?"
                )
            self.assertEqual(result.answer_source, "model")
            self.assertIn("priemernú absolútnu chybu", result.text)
            self.assertEqual(generated.call_count, 2)
            self.assertFalse(missing.exists())

    def test_user_named_historical_split_overrides_wrong_model_choice(self):
        # A classifier cannot turn a validation question into a lockbox
        # answer (or vice versa), including overview and comparison actions.
        cases = (
            (
                "Aké boli výsledky historickej validácie v3?",
                "test", "test", "validation", "validácia",
            ),
            (
                "Aká bola MAE na historickom lockbox teste modelu v3?",
                "validation", "validation", "test", "lockbox",
            ),
            (
                "Porovnaj MAE modelu v3 s lineárnym baseline v historickej validácii.",
                "compare", "test", "compare_validation", "validácii",
            ),
            (
                "Zhrň výsledky historickej validácie v3 jednoducho.",
                "simple", "test", "simple", "validácie",
            ),
        )
        for question, action, wrong_split, expected_kind, marker in cases:
            with self.subTest(question=question):
                selected = select_verified_tool(
                    question, choice("historical_v3", action, wrong_split),
                    historical_available=True, forward_available=True,
                )
                self.assertIsNotNone(selected)
                self.assertEqual(selected[0], "historical_v3")
                self.assertIn(marker, selected[1])
                self.assertEqual(historical_question_kind(selected[1]), expected_kind)

    def test_user_named_historical_baseline_overrides_wrong_model_choice(self):
        # The opt-in model may propose the wrong comparison baseline. The
        # user's explicitly named baseline(s) must win, including both.
        cases = (
            (
                "Porovnaj MAE modelu v3 s lineárnym baseline v historickej validácii.",
                "null", "lineárnym modelom", "nulovým baseline",
            ),
            (
                "Ako si obstál v historickej validácii v3 oproti nulovému baseline?",
                "linear", "nulovým baseline", "lineárnym modelom",
            ),
            (
                "Porovnaj historickú MAE modelu v3 s lineárnym aj nulovým baseline.",
                "linear", "lineárnym aj nulovým baseline", None,
            ),
            (
                "Porovnaj historickú MAE modelu v3 s lineárnym aj nulovým baseline.",
                "null", "lineárnym aj nulovým baseline", None,
            ),
        )
        for question, wrong_baseline, expected, excluded in cases:
            with self.subTest(question=question, proposed=wrong_baseline):
                proposal = choice("historical_v3", "compare", "both", wrong_baseline)
                selected = select_verified_tool(
                    question, proposal,
                    historical_available=True, forward_available=True,
                )
                self.assertIsNotNone(selected)
                self.assertEqual(selected[0], "historical_v3")
                self.assertIn(expected, selected[1])
                if excluded:
                    self.assertNotIn(excluded, selected[1])
                self.assertEqual(
                    selected[1],
                    canonical_historical_question(
                        question, json.dumps({
                            "intent": "compare", "split": "both",
                            "baseline": wrong_baseline,
                        }),
                    ),
                )

        # Without a named baseline, retain the existing model-selected
        # baseline rather than changing unrelated classification behavior.
        unspecified = select_verified_tool(
            "Porovnaj historické výsledky modelu v3 s benchmarkom.",
            choice("historical_v3", "compare", "both", "linear"),
            historical_available=True, forward_available=True,
        )
        self.assertIsNotNone(unspecified)
        self.assertIn("lineárnym modelom", unspecified[1])

    def test_negated_baseline_contrast_never_includes_excluded_model(self):
        # Synthetic validation fixture only; this does not test local Qwen.
        with tempfile.TemporaryDirectory() as directory:
            validation_fixture(Path(directory))
            cases = (
                ("Neporovnávaj historickú validáciu v3 s lineárnym modelom, ale s nulovým baseline.",
                 "linear", "nulový baseline", "lineárny baseline"),
                ("Porovnaj historickú MAE v3 nie s nulovým baseline, ale s lineárnym modelom.",
                 "null", "lineárny baseline", "nulový baseline"),
            )
            for question, wrong_proposal, expected, excluded in cases:
                with self.subTest(question=question):
                    # Deterministic reader runs before optional tool selection.
                    self.assertEqual(historical_question_kind(question), "compare_validation"
                                     if "validáciu" in question else "compare")
                    deterministic = maybe_historical_answer(question, directory)
                    self.assertIsNotNone(deterministic)
                    self.assertIn(expected, deterministic)
                    self.assertNotIn(excluded, deterministic)
                    for tool in ("historical_v3",):
                        selected = select_verified_tool(
                            question, choice(tool, "compare", "validation", wrong_proposal),
                            historical_available=True, forward_available=True,
                        )
                        self.assertIsNotNone(selected)
                        self.assertEqual(selected[0], "historical_v3")
                        self.assertIn(
                            "nulovým baseline" if expected == "nulový baseline"
                            else "lineárnym modelom",
                            selected[1],
                        )
                        self.assertNotIn(
                            "lineárnym aj nulovým baseline", selected[1],
                        )
                        answer = maybe_historical_answer(selected[1], directory)
                        self.assertIn(expected, answer)
                        self.assertNotIn(excluded, answer)

    def test_ambiguous_baseline_exclusions_fail_closed(self):
        for question in (
            "Neporovnávaj historickú MAE v3 s lineárnym baseline.",
            "Neporovnávaj historickú MAE v3 s lineárnym ani s nulovým baseline.",
            "Porovnaj historickú MAE v3 nie s lineárnym, ale s lineárnym.",
        ):
            with self.subTest(question=question):
                self.assertIsNone(historical_question_kind(question))
                self.assertIsNone(canonical_historical_question(
                    question, json.dumps({
                        "intent": "compare", "split": "validation",
                        "baseline": "both",
                    }),
                ))
                self.assertIsNone(select_verified_tool(
                    question, choice("historical_v3", "compare", "validation", "both"),
                    historical_available=True, forward_available=True,
                ))
        # Positive requests for both baselines must remain unchanged.
        both = "Porovnaj historickú MAE v3 s lineárnym aj nulovým baseline."
        self.assertIn("lineárnym aj nulovým baseline", select_verified_tool(
            both, choice("historical_v3", "compare", "both", "linear"),
            historical_available=True, forward_available=True,
        )[1])

    def test_named_baseline_is_honored_by_existing_read_only_report(self):
        # Synthetic fixture only: no real measured historical/forward values.
        with tempfile.TemporaryDirectory() as directory:
            validation_fixture(Path(directory))
            for question, wrong, expected, excluded in (
                (
                    "Porovnaj historickú validáciu modelu v3 s lineárnym baseline.",
                    "null", "lineárny baseline", "nulový baseline",
                ),
                (
                    "Porovnaj historickú validáciu modelu v3 s nulovým baseline.",
                    "linear", "nulový baseline", "lineárny baseline",
                ),
            ):
                with self.subTest(question=question):
                    selected = select_verified_tool(
                        question, choice("historical_v3", "compare", "test", wrong),
                        historical_available=True, forward_available=True,
                    )
                    self.assertIsNotNone(selected)
                    answer = maybe_historical_answer(selected[1], directory)
                    self.assertIsNotNone(answer)
                    self.assertIn(expected, answer)
                    self.assertNotIn(excluded, answer)

    def test_model_cannot_narrow_explicit_validation_and_lockbox_comparison(self):
        question = "Porovnaj historickú validáciu a lockbox test modelu v3."
        self.assertIsNone(select_verified_tool(
            question, choice("historical_v3", "compare", "validation"),
            historical_available=True, forward_available=True,
        ))
        self.assertIsNone(select_verified_tool(
            question, choice("historical_v3", "compare", "test"),
            historical_available=True, forward_available=True,
        ))

    def test_historical_source_never_steals_shadow_and_requires_report_config(self):
        classification = choice("historical_v3", "compare", "validation", "linear")
        result = select_verified_tool(
            "Ako sa darilo OurAI v3 oproti lineárnemu modelu?",
            classification, historical_available=True, forward_available=True,
        )
        self.assertIsNotNone(result)
        self.assertEqual(result[0], "historical_v3")
        self.assertIn("validácii", result[1])
        self.assertIn("lineárnym modelom", result[1])
        self.assertIsNone(select_verified_tool(
            "Ako sa ti darí v shadow režime?", classification,
            historical_available=True, forward_available=True,
        ))
        self.assertIsNone(select_verified_tool(
            "Ako sa darilo OurAI v3?", classification,
            historical_available=False, forward_available=True,
        ))

    def test_historical_reader_never_steals_forward_shadow_metrics(self):
        self.assertIsNone(historical_question_kind("Ako sa ti darí v shadow režime?"))
        self.assertIsNone(historical_question_kind("Aká je moja MAE v shadow režime?"))
        self.assertIsNone(historical_question_kind("Ako úspešný je paper trading?"))

    def test_ticker_can_only_come_from_user_and_frozen_universe(self):
        original = "Aké máš výsledky shadow pre NVDA?"
        for operation in ("summary", "latest", "performance"):
            result = select_verified_tool(
                original, choice("forward_v2", operation),
                historical_available=False, forward_available=True,
            )
            self.assertEqual(result[0], "forward_v2")
            self.assertIn("NVDA", result[1])
        self.assertNotIn("NVDA", select_verified_tool(
            "Ako sa ti darí v shadow režime?", choice("forward_v2", "performance"),
            historical_available=False, forward_available=True,
        )[1])

    def test_ollama_forward_routes_only_into_readonly_verified_journal(self):
        with tempfile.TemporaryDirectory() as temp:
            nonexistent = Path(temp) / "not-created.jsonl"
            runtime = LocalOllamaChatRuntime(
                forward_journal=nonexistent, semantic_tool_routing=True
            )
            with patch.object(
                runtime, "_generate", return_value=choice("forward_v2", "performance")
            ) as generate:
                result = runtime.respond("Ako sa ti darí v shadow režime?")
            self.assertEqual(generate.call_count, 1)
            self.assertTrue(generate.call_args.kwargs.get("json_mode"))
            self.assertEqual(result.answer_source, "verified_forward_journal")
            self.assertIn("zatiaľ neexistuje", result.text)
            self.assertNotIn("historická validácia", result.text)
            self.assertFalse(nonexistent.exists())

    def test_ollama_historical_routes_only_into_verified_report(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            validation_fixture(root)
            runtime = LocalOllamaChatRuntime(
                historical_reports_dir=root, semantic_tool_routing=True
            )
            with patch.object(
                runtime, "_generate",
                return_value=choice("historical_v3", "compare", "validation", "linear"),
            ) as generate:
                result = runtime.respond(
                    "Ako sa darilo OurAI v3 oproti lineárnemu modelu?"
                )
            self.assertEqual(generate.call_count, 1)
            self.assertEqual(result.answer_source, "historical_v3_reports")
            self.assertIn("8,00 bps", result.text)
            self.assertIn("10,00 bps", result.text)
            self.assertNotIn("forward v2", result.text.lower())
            self.assertIn("validácii", result.evidence_question)

    def test_unrelated_question_goes_to_normal_qwen(self):
        with tempfile.TemporaryDirectory() as temp:
            runtime = LocalOllamaChatRuntime(
                forward_journal=Path(temp) / "missing.jsonl",
                semantic_tool_routing=True,
            )
            with patch.object(runtime, "_generate", side_effect=[
                choice("forward_v2", "performance"),
                "Kvety získavajú energiu fotosyntézou.",
            ]) as generated:
                result = runtime.respond("Ako funguje fotosyntéza?")
            self.assertEqual(result.answer_source, "model")
            self.assertIn("fotosyntézou", result.text)
            self.assertEqual(generated.call_count, 2)

    def test_existing_deterministic_forward_path_is_not_affected(self):
        with tempfile.TemporaryDirectory() as temp:
            runtime = LocalOllamaChatRuntime(
                forward_journal=Path(temp) / "missing.jsonl",
                semantic_tool_routing=True,
            )
            with patch.object(runtime, "_generate") as generated:
                result = runtime.respond("Aké boli tvoje posledné predikcie?")
            self.assertEqual(result.answer_source, "verified_forward_journal")
            generated.assert_not_called()

    def test_cli_flag_and_existing_legacy_mode(self):
        options = parser().parse_args(["--backend", "ollama", "--semantic-tool-routing"])
        self.assertTrue(options.semantic_tool_routing)
        self.assertFalse(options.semantic_routing)
        self.assertFalse(LocalOllamaChatRuntime().semantic_tool_routing)
        self.assertFalse(LocalOllamaChatRuntime().semantic_routing)
        with self.assertRaises(ValueError):
            LocalOllamaChatRuntime(
                semantic_routing=True, semantic_tool_routing=True
            )


if __name__ == "__main__":
    unittest.main()
