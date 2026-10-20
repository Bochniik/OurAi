"""Experimental verified tool selection never trusts Qwen with financial facts."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ourai_language.historical_answers import historical_question_kind
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
