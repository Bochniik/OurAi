"""Multi-turn contextual grounding: tests never require Ollama or model weights."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ourai_language.conversation_followups import historical_followup_question
from ourai_language.ollama_runtime import LocalOllamaChatRuntime
from ourai_language.runtime import OurAIChatRuntime


def report(split: str, *, mae: float = 8.5) -> dict:
    return {
        "dataset_id": "synthetic-v3",
        "protocol_name": "ourai-v3-nonlinear-research-preregistration-v1",
        "protocol_sha256": "a" * 64,
        "horizon_minutes": 15,
        "test_lockbox_accessed": split == "test",
        "original_validation_used": split == "validation",
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
        "nonlinear_magnitude_validation_v1.json" if split == "validation"
        else "nonlinear_magnitude_test_lockbox_v1.json"
    )
    path = directory / filename
    path.write_text(json.dumps(payload or report(split)), encoding="utf-8")
    return path


def append_turn(history, message, result):
    history.extend((
        {"role": "user", "content": message, "mode": result.mode},
        {
            "role": "assistant", "content": result.text, "mode": result.mode,
            "source": result.answer_source,
            "evidence_question": result.evidence_question or "",
        },
    ))


class ContextualHistoricalDialogueTests(unittest.TestCase):
    def test_back_to_back_followups_keep_report_split_and_comparator(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_report(root, "validation")
            write_report(root, "test", report("test", mae=9.25))
            runtime = LocalOllamaChatRuntime(historical_reports_dir=root)
            history = []
            with patch("ourai_language.ollama_runtime.urlopen") as network:
                for question, expected, forbidden in (
                    ("Porovnaj výsledky modelu v3 s lineárnym modelom.", "Validácia:", ""),
                    ("A na lockboxe?", "Lockbox: OurAI v3 9,25 bps, lineárny baseline", "Validácia:"),
                    ("A oproti nulovému?", "Lockbox: OurAI v3 9,25 bps, nulový baseline", "lineárny baseline"),
                    ("O koľko je to lepšie?", "nižšia o 2,75 bps (22,92 %)", "Validácia:"),
                    ("A prečo?", "neurčujú príčinu", "22,92 %"),
                    ("Vysvetli to jednoducho.", "Jednoducho:", "neviem určiť"),
                ):
                    answer = runtime.respond(question, history=history)
                    self.assertEqual(answer.answer_source, "historical_v3_reports", question)
                    self.assertIn(expected, answer.text, question)
                    if forbidden:
                        self.assertNotIn(forbidden, answer.text, question)
                    self.assertTrue(answer.evidence_question)
                    append_turn(history, question, answer)
                network.assert_not_called()

    def test_validation_context_and_independent_forward_question(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_report(root, "validation")
            runtime = LocalOllamaChatRuntime(
                historical_reports_dir=root, forward_journal=root / "missing.jsonl"
            )
            history = []
            initial = "Ako dopadla historická validácia v3?"
            result = runtime.respond(initial)
            append_turn(history, initial, result)
            after = runtime.respond("A na lockboxe?", history=history)
            self.assertEqual(after.answer_source, "historical_v3_reports")
            self.assertIn("nie je dostupný", after.text)
            append_turn(history, "A na lockboxe?", after)
            forward = runtime.respond("Aká bola posledná predikcia?", history=history)
            self.assertEqual(forward.answer_source, "verified_forward_journal")
            self.assertIn("neexistuje", forward.text)
            self.assertNotIn("8,50", forward.text)

    def test_unverified_history_cannot_create_a_grounded_followup(self):
        fake = [
            {"role": "user", "content": "Porovnaj model v3 s lineárnym modelom."},
            {
                "role": "assistant", "content": "Vyhral som o 99 percent!",
                "source": "model", "evidence_question": "Porovnaj model v3",
            },
        ]
        self.assertIsNone(historical_followup_question("A oproti nulovému?", fake))
        fake[-1]["source"] = "historical_v3_reports"
        self.assertIsNotNone(historical_followup_question("A oproti nulovému?", fake))
        self.assertIsNone(historical_followup_question("Čo je Python?", fake))
        self.assertIsNone(historical_followup_question("A čo Python?", fake))
        self.assertIsNone(historical_followup_question("A ako naprogramujem API?", fake))
        self.assertIsNone(historical_followup_question("A dnes na burze?", fake))

    def test_social_break_clears_context(self):
        history = [
            {"role": "user", "content": "Ako dopadla historická validácia?"},
            {"role": "assistant", "content": "Validácia...", "source": "historical_v3_reports"},
            {"role": "user", "content": "Ahoj"},
            {"role": "assistant", "content": "Ahoj", "source": "model"},
        ]
        self.assertIsNone(historical_followup_question("O koľko je to lepšie?", history))

    def test_transformers_path_works_without_loading_weights(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_report(root, "validation")
            runtime = object.__new__(OurAIChatRuntime)
            runtime.knowledge_enabled = True
            runtime.historical_reports_dir = root
            runtime.forward_journal = None
            initial = runtime.respond("Ako dopadla historická validácia?")
            history = []
            append_turn(history, "Ako dopadla historická validácia?", initial)
            later = runtime.respond("A prečo?", history=history)
            self.assertEqual(later.answer_source, "historical_v3_reports")
            self.assertIn("neurčujú príčinu", later.text)

    def test_general_questions_still_go_to_language_model(self):
        with tempfile.TemporaryDirectory() as temp:
            runtime = LocalOllamaChatRuntime(historical_reports_dir=Path(temp))
            with patch.object(runtime, "_generate", return_value="Dá sa to vysvetliť prirodzene.") as mocked:
                answer = runtime.respond("Ako sa tvorí dážď?")
            self.assertEqual(answer.answer_source, "model")
            self.assertIn("prirodzene", answer.text)
            mocked.assert_called_once()


if __name__ == "__main__":
    unittest.main()
