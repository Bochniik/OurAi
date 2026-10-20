"""Constrained semantic routing: no financial numbers or filenames from an LLM."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ourai_language.ollama_runtime import LocalOllamaChatRuntime
from ourai_language.semantic_router import (
    canonical_historical_question, explicit_heldout_performance_question,
)
from scripts.chat_ourai import parser


def _classification(intent: str, split: str = "both", baseline: str = "both") -> str:
    return json.dumps({"intent": intent, "split": split, "baseline": baseline})


def _validation_report() -> dict:
    return {
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


class SemanticRouterTests(unittest.TestCase):
    def test_canonical_route_is_strict_and_sanitizes_model_arguments(self):
        cases = (
            ("overview", "Ako presný bol historicky model v3?"),
            ("validation", "historická validácia"),
            ("test", "lockbox"),
            ("symbols", "najväčšie chyby"),
            ("causes", "Prečo sa líšia"),
            ("simple", "jednoducho"),
            ("compare", "Porovnaj model v3"),
        )
        for intent, text in cases:
            with self.subTest(intent=intent):
                result = canonical_historical_question(
                    "Ako sa darilo OurAI v3?", _classification(intent)
                )
                self.assertIn(text, result)
        self.assertIsNone(canonical_historical_question("Ako sa darilo?", _classification("none")))
        self.assertIsNone(canonical_historical_question("Ahoj", '{"intent":"test","path":"/secrets"}'))
        self.assertIsNone(canonical_historical_question("Ahoj", '{"intent":"test","split":"both","baseline":"both","profit":999}'))
        self.assertIsNone(canonical_historical_question("Ahoj", '{"intent":"test","split":"bad","baseline":"both"}'))
        self.assertIsNone(canonical_historical_question("Ahoj", '{"intent":"test","split":"both","baseline":100}'))
        self.assertIsNone(canonical_historical_question("Ahoj", "not json"))

    def test_llm_misclassification_cannot_hijack_unrelated_or_definition_questions(self):
        for question in (
            "Aké je hlavné mesto Francúzska?",
            "Vysvetli mi fyziku hviezd.",
            "Čo je MAE?",
            "What is a backtest?",
            "Aký bol historický vývoj inflácie?",
            "What is the difference between a model and a theory?",
            "How do I improve my performance at chess?",
            "Ahoj, ako sa máš?",
        ):
            with self.subTest(question=question):
                self.assertIsNone(
                    canonical_historical_question(question, _classification("test"))
                )
                self.assertIsNone(
                    canonical_historical_question(question, _classification("compare"))
                )

    def test_evidence_anchor_accepts_flexible_retrospective_phrasings(self):
        for question in (
            "Ako si obstál na nevidených dátach?",
            "Ako sa darilo OurAI v3?",
            "Kto je historicky lepší?",
            "How well do you compare with a linear model?",
            "Why did v3 outperform the baseline historically?",
            "What is the historical MAE of model v3?",
        ):
            with self.subTest(question=question):
                self.assertIsNotNone(
                    canonical_historical_question(question, _classification("compare"))
                )

    def test_temporal_guard_matches_words_not_unrelated_substrings(self):
        self.assertIsNotNone(
            canonical_historical_question(
                "Did you deliver better historical model results?",
                _classification("compare"),
            )
        )
        self.assertIsNone(
            canonical_historical_question(
                "How did your model perform on live data?",
                _classification("compare"),
            )
        )
        self.assertIsNone(
            canonical_historical_question(
                "Ako presné sú tvoje dnešné predikcie?",
                _classification("compare"),
            )
        )

    def test_wrong_classifier_output_still_delegates_to_general_model(self):
        with tempfile.TemporaryDirectory() as temp:
            runtime = LocalOllamaChatRuntime(
                historical_reports_dir=temp, semantic_routing=True
            )
            with patch.object(
                runtime, "_generate",
                side_effect=[
                    _classification("test"),
                    "Hlavným mestom Francúzska je Paríž.",
                ],
            ) as generate:
                answer = runtime.respond("Aké je hlavné mesto Francúzska?")
            self.assertEqual(answer.answer_source, "model")
            self.assertIn("Paríž", answer.text)
            self.assertEqual(generate.call_count, 2)

    def test_current_and_forward_questions_never_hijacked(self):
        for question in (
            "Aké boli tvoje posledné predikcie?",
            "Čo si predpovedal dnes pri NVDA?",
            "What is the latest live forward forecast?",
            "Ako sa ti darí na aktuálnom trhu?",
        ):
            with self.subTest(question=question):
                self.assertIsNone(
                    canonical_historical_question(question, _classification("test"))
                )

    def test_comparison_preserves_selected_baseline_and_split(self):
        question = canonical_historical_question(
            "Kto je historicky lepší?", _classification("compare", "test", "linear")
        )
        self.assertIn("lockbox", question)
        self.assertIn("lineárnym modelom", question)
        self.assertNotIn("nulovým", question)
        null_question = canonical_historical_question(
            "Kto je historicky lepší?", _classification("compare", "validation", "null")
        )
        self.assertIn("validácii", null_question)
        self.assertIn("nulovým baseline", null_question)

    def test_actual_colloquial_unseen_dialogue_survives_wrong_classifier(self):
        # Regression for a real user transcript: first question was answered by
        # generic Qwen chat instead of local held-out evidence, losing follow-up.
        question = "Ako sa ti vlastne darilo na dátach, ktoré si predtým nevidel?"
        self.assertEqual(
            explicit_heldout_performance_question(question),
            "Aký bol historický lockbox test v3?",
        )
        with tempfile.TemporaryDirectory() as temp:
            data = _validation_report()
            data["test_lockbox_accessed"] = True
            data["post_test_tuning_allowed"] = False
            data["test_supported"] = True
            data["test"] = data.pop("validation")
            data.pop("validation_supported")
            data["test"]["overall"]["nonlinear_mae_bps"] = 17.19
            data["test"]["overall"]["linear_mae_bps"] = 17.65
            (Path(temp) / "nonlinear_magnitude_test_lockbox_v1.json").write_text(
                json.dumps(data), encoding="utf-8"
            )
            runtime = LocalOllamaChatRuntime(
                historical_reports_dir=temp, semantic_routing=True,
            )
            # Even an incorrectly returned 'none' cannot turn an explicit
            # self-performance/held-out question into unrelated chat.
            with patch.object(
                runtime, "_generate", return_value=_classification("none")
            ) as generated:
                first = runtime.respond(question)
                self.assertEqual(first.answer_source, "historical_v3_reports")
                self.assertIn("17,19 bps", first.text)
                history = [
                    {"role": "user", "content": question, "mode": first.mode},
                    {
                        "role": "assistant", "content": first.text,
                        "mode": first.mode, "source": first.answer_source,
                        "evidence_question": first.evidence_question,
                    },
                ]
                followup = runtime.respond(
                    "A ako si na tom bol oproti lineárnemu modelu?",
                    history=history,
                )
                self.assertEqual(followup.answer_source, "historical_v3_reports")
                self.assertIn("Lockbox:", followup.text)
                self.assertIn("17,65 bps", followup.text)
                self.assertIn("nižšia o 0,46 bps", followup.text)
                self.assertNotIn("Validácia:", followup.text)
                self.assertEqual(generated.call_count, 1)

    def test_unseen_fallback_does_not_hijack_generic_or_live_questions(self):
        questions = (
            "Vysvetli mi, čo je out-of-sample test.",
            "Ako sa ti darilo na dátach z dnešného trhu?",
            "Na skúške som nevidel otázky, ako sa ti darilo?",
            "Čo si myslíš o dátach, ktoré si predtým nevidel?",
            "Ako sa darilo modelom na nevidených dátach všeobecne?",
            "Ako sa ti darilo na živých forward dátach?",
        )
        for question in questions:
            with self.subTest(question=question):
                self.assertIsNone(explicit_heldout_performance_question(question))

    def test_ollama_semantic_route_uses_local_classifier_and_checked_metrics(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "nonlinear_magnitude_validation_v1.json"
            path.write_text(json.dumps(_validation_report()), encoding="utf-8")
            runtime = LocalOllamaChatRuntime(
                historical_reports_dir=temp, semantic_routing=True
            )
            with patch.object(
                runtime, "_generate", return_value=_classification("validation")
            ) as generate:
                answer = runtime.respond("Ako si obstál na nevidených dátach?")
            self.assertEqual(generate.call_count, 1)
            self.assertTrue(generate.call_args.kwargs.get("json_mode"))
            self.assertEqual(answer.answer_source, "historical_v3_reports")
            self.assertIn("8,00 bps", answer.text)
            self.assertNotIn("99", answer.text)
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), _validation_report())
            self.assertTrue(answer.evidence_question)

    def test_llm_none_delegates_to_general_conversation(self):
        with tempfile.TemporaryDirectory() as temp:
            runtime = LocalOllamaChatRuntime(
                historical_reports_dir=temp, semantic_routing=True
            )
            with patch.object(
                runtime, "_generate",
                side_effect=[_classification("none"), "Fyzika opisuje prírodné javy."],
            ) as generate:
                answer = runtime.respond("Vysvetli mi fyziku hviezd.")
            self.assertEqual(generate.call_count, 2)
            self.assertEqual(answer.answer_source, "model")
            self.assertIn("prírodné javy", answer.text)

    def test_explicit_recent_forward_question_never_calls_classifier(self):
        with tempfile.TemporaryDirectory() as temp:
            runtime = LocalOllamaChatRuntime(
                historical_reports_dir=temp, forward_journal=Path(temp) / "missing.jsonl",
                semantic_routing=True,
            )
            with patch.object(runtime, "_generate") as generated:
                answer = runtime.respond("Aké boli tvoje posledné predikcie?")
            generated.assert_not_called()
            self.assertEqual(answer.answer_source, "verified_forward_journal")

    def test_existing_deterministic_route_skips_classifier(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "nonlinear_magnitude_validation_v1.json"
            path.write_text(json.dumps(_validation_report()), encoding="utf-8")
            runtime = LocalOllamaChatRuntime(
                historical_reports_dir=temp, semantic_routing=True
            )
            with patch.object(runtime, "_generate") as generated:
                answer = runtime.respond("Ako dopadla historická validácia v3?")
            generated.assert_not_called()
            self.assertIn("8,00 bps", answer.text)

    def test_invalid_report_is_not_masked_by_model_generated_numbers(self):
        with tempfile.TemporaryDirectory() as temp:
            (Path(temp) / "nonlinear_magnitude_validation_v1.json").write_text(
                "{broken", encoding="utf-8"
            )
            runtime = LocalOllamaChatRuntime(
                historical_reports_dir=temp, semantic_routing=True
            )
            with patch.object(runtime, "_generate", return_value=_classification("validation")):
                with self.assertRaisesRegex(RuntimeError, "neplatný"):
                    runtime.respond("Ako si obstál na nevidených dátach?")

    def test_disabled_by_default_and_cli_flag_is_present(self):
        runtime = LocalOllamaChatRuntime()
        self.assertFalse(runtime.semantic_routing)
        args = parser().parse_args(["--backend", "ollama", "--semantic-routing"])
        self.assertTrue(args.semantic_routing)


if __name__ == "__main__":
    unittest.main()
