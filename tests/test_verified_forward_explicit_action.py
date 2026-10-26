"""Regression: explicit Slovak forward intent outranks an incorrect Qwen action.

Synthetic classifier output only; no local model, journal, or network.
"""
import json
import unittest

from ourai_language.forward_answers import forward_question_kind
from ourai_language.verified_tool_router import select_verified_tool


def proposal(action: str) -> str:
    return json.dumps({
        "tool": "forward_v2", "action": action,
        "split": "both", "baseline": "both",
    })


def choose(question: str, action: str):
    return select_verified_tool(
        question, proposal(action),
        historical_available=True, forward_available=True,
    )


class ExplicitForwardActionTests(unittest.TestCase):
    def test_explicit_latest_record_cannot_be_changed_to_aggregate(self):
        for question in (
            "Aká je posledná predikcia v shadow režime?",
            "Ukáž poslednú predikciu pre NVDA vo forward denníku.",
            "A najnovšiu predikciu Tesly vo forward denníku?",
            "Aký je posledný záznam v shadow denníku?",
        ):
            for wrong_action in ("summary", "performance"):
                with self.subTest(question=question, wrong_action=wrong_action):
                    selected = choose(question, wrong_action)
                    self.assertIsNotNone(selected)
                    self.assertEqual(selected[0], "forward_v2")
                    self.assertEqual(forward_question_kind(selected[1]), "latest")
                    if "NVDA" in question:
                        self.assertIn("NVDA", selected[1])
                    if "Tesly" in question:
                        self.assertIn("TSLA", selected[1])

    def test_explicit_mae_or_accuracy_cannot_be_changed_to_latest(self):
        for question in (
            "Aká je MAE v shadow režime?",
            "Akú presnosť majú predikcie NVDA vo forward denníku?",
            "Aká je chybovosť OurAI v shadow režime?",
            "Aká je úspešnosť predikcií vo forward denníku?",
        ):
            for wrong_action in ("summary", "latest"):
                with self.subTest(question=question, wrong_action=wrong_action):
                    selected = choose(question, wrong_action)
                    self.assertIsNotNone(selected)
                    self.assertEqual(selected[0], "forward_v2")
                    self.assertEqual(forward_question_kind(selected[1]), "performance")

    def test_mixed_explicit_metrics_fail_closed(self):
        for question in (
            "Ukáž poslednú predikciu a MAE vo forward denníku.",
            "Aká je posledná predikcia a jej presnosť v shadow režime?",
            "Nie MAE, ale poslednú predikciu v shadow režime.",
        ):
            for action in ("latest", "performance", "summary"):
                with self.subTest(question=question, action=action):
                    self.assertIsNone(choose(question, action))

    def test_broad_results_keep_existing_model_choice(self):
        question = "Aké máš výsledky shadow pre NVDA?"
        for action, kind in (
            ("latest", "latest"), ("performance", "performance"),
            ("summary", "summary"),
        ):
            with self.subTest(action=action):
                selected = choose(question, action)
                self.assertIsNotNone(selected)
                self.assertEqual(forward_question_kind(selected[1]), kind)

    def test_evidence_and_symbol_boundaries_remain(self):
        for question in (
            "Aká je posledná predikcia pre IBM vo forward denníku?",
            "Aká je MAE NVDA a TSLA vo forward denníku?",
            "Aká je posledná predikcia na historickom lockbox teste?",
            "Čo je posledná predikcia v shadow režime?",
        ):
            with self.subTest(question=question):
                self.assertIsNone(choose(question, "summary"))
        self.assertIsNone(select_verified_tool(
            "Aká je posledná predikcia v shadow režime?",
            proposal("latest"), historical_available=True,
            forward_available=False,
        ))


if __name__ == "__main__":
    unittest.main()
