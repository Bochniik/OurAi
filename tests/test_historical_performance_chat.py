"""Regression tests for read-only v3 historical chat interpretation."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ourai_language.historical_answers import (
    historical_question_kind, maybe_historical_answer,
)
from ourai_language.ollama_runtime import LocalOllamaChatRuntime
from ourai_language.runtime import OurAIChatRuntime


def report(split: str, *, mae: float = 8.5, dataset: str = "test-v3") -> dict:
    is_test = split == "test"
    return {
        "experiment": f"v3_hist_gradient_absolute_return_magnitude_{split}_v1",
        "dataset_id": dataset,
        "protocol_name": "ourai-v3-nonlinear-research-preregistration-v1",
        "protocol_sha256": "a" * 64,
        "horizon_minutes": 15,
        "test_lockbox_accessed": is_test,
        "original_validation_used": not is_test,
        "post_test_tuning_allowed": False,
        f"{split}_supported": True,
        split: {
            "overall": {
                "evaluated_count": 150,
                "nonlinear_mae_bps": mae,
                "linear_mae_bps": 11.0,
                "null_mae_bps": 12.0,
            },
            "market_date_count": 5,
            "market_date_wins_vs_linear": 4,
            "market_date_wins_vs_null": 5,
        },
    }


def write_report(directory: Path, split: str, payload: dict | None = None) -> Path:
    filename = (
        "nonlinear_magnitude_validation_v1.json"
        if split == "validation" else "nonlinear_magnitude_test_lockbox_v1.json"
    )
    path = directory / filename
    path.write_text(json.dumps(payload or report(split)), encoding="utf-8")
    return path


class HistoricalAnswersTests(unittest.TestCase):
    def test_routing_avoids_recent_forward_and_unrelated_questions(self):
        self.assertIsNone(historical_question_kind("Ako presné boli tvoje posledné predikcie?"))
        self.assertIsNone(historical_question_kind("Aká je posledná predikcia NVDA?"))
        self.assertIsNone(historical_question_kind("Ahoj"))
        self.assertIsNone(historical_question_kind("Čo je volatilita?"))
        self.assertEqual(historical_question_kind("Ako dopadla historická validácia v3?"), "validation")
        self.assertEqual(historical_question_kind("Aké sú výsledky lockbox testu?"), "test")
        self.assertEqual(historical_question_kind("Pri ktorých akciách robíš najväčšie chyby?"), "symbols")

    def test_validation_numbers_from_local_file_and_source_limits(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = write_report(root, "validation")
            original = path.read_bytes()
            answer = maybe_historical_answer("Ako dopadla historická validácia?", root)
            self.assertIn("8,50 bps", answer)
            self.assertIn("11,00 bps", answer)
            self.assertIn("150 vyhodnotených", answer)
            self.assertIn("4/5", answer)
            self.assertIn("nie je presnosť smeru UP/DOWN", answer)
            self.assertEqual(path.read_bytes(), original)

    def test_test_lockbox_is_separate_and_comparison_uses_both(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_report(root, "validation")
            write_report(root, "test", report("test", mae=9.25))
            lockbox = maybe_historical_answer("Čo ukázal lockbox?", root)
            self.assertIn("9,25 bps", lockbox)
            self.assertNotIn("8,50 bps", lockbox)
            comparison = maybe_historical_answer("Porovnaj model v3", root)
            self.assertIn("8,50 bps", comparison)
            self.assertIn("9,25 bps", comparison)
            self.assertIn("nie", comparison)

    def test_missing_report_is_explicit_and_never_fabricated(self):
        with tempfile.TemporaryDirectory() as temp:
            answer = maybe_historical_answer("Ako presný si bol?", temp)
            self.assertIn("nie je dostupný", answer)
            self.assertNotIn("bps", answer)
            self.assertIsNone(maybe_historical_answer("Ahoj", temp))

    def test_no_per_symbol_statistics_are_invented(self):
        with tempfile.TemporaryDirectory() as temp:
            write_report(Path(temp), "validation")
            answer = maybe_historical_answer("Pri ktorých akciách robíš najväčšie chyby?", temp)
            self.assertIn("per-symbol", answer)
            self.assertNotIn("NVDA", answer)
            self.assertNotIn("8,50", answer)

    def test_fail_closed_on_invalid_values_and_provenance(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            altered = report("validation")
            altered["validation"]["overall"]["nonlinear_mae_bps"] = float("nan")
            write_report(root, "validation", altered)
            with self.assertRaisesRegex(ValueError, "invalid historical metric"):
                maybe_historical_answer("Historická validácia?", root)

            write_report(root, "validation")
            altered_test = report("test", dataset="unrelated")
            write_report(root, "test", altered_test)
            with self.assertRaisesRegex(ValueError, "provenance mismatch"):
                maybe_historical_answer("Porovnaj model v3", root)

            altered_test["dataset_id"] = "test-v3"
            altered_test["test_lockbox_accessed"] = False
            write_report(root, "test", altered_test)
            with self.assertRaisesRegex(ValueError, "provenance mismatch"):
                maybe_historical_answer("Čo ukázal lockbox?", root)

    def test_ollama_and_transformers_path_use_no_generation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_report(root, "validation")
            with patch("ourai_language.ollama_runtime.urlopen") as mocked:
                ollama = LocalOllamaChatRuntime(historical_reports_dir=root)
                actual = ollama.respond("Aká bola historická validácia v3?")
                mocked.assert_not_called()
            hf = object.__new__(OurAIChatRuntime)
            hf.knowledge_enabled = True
            hf.historical_reports_dir = root
            hf.forward_journal = None
            alternate = hf.respond("Aká bola historická validácia v3?")
            self.assertEqual(actual.text, alternate.text)
            self.assertEqual(actual.answer_source, "historical_v3_reports")
            self.assertEqual(alternate.answer_source, "historical_v3_reports")

    def test_bad_report_blocks_llm_guessing(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_report(root, "validation").write_text("{broken json", encoding="utf-8")
            with patch("ourai_language.ollama_runtime.urlopen") as mocked:
                runtime = LocalOllamaChatRuntime(historical_reports_dir=root)
                with self.assertRaisesRegex(RuntimeError, "neplatný"):
                    runtime.respond("Ako dopadla historická validácia?")
                mocked.assert_not_called()


if __name__ == "__main__":
    unittest.main()
