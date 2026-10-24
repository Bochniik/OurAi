"""Regression tests: educational questions cannot trigger verified evidence readers."""
import json
import unittest

from ourai_language.verified_tool_router import select_verified_tool


def proposal(tool, action):
    return json.dumps({
        "tool": tool, "action": action, "split": "both", "baseline": "both",
    })


class EducationalToolGuardTests(unittest.TestCase):
    def test_forward_definitions_never_open_journal_even_if_model_misroutes(self):
        questions = (
            "Vysvetli mi, čo je forward model.",
            "Vysvetli prosím forward predikcie OurAI.",
            "Ako funguje forward model?",
            "Ako vlastne funguje forward denník?",
            "Na čo slúži shadow report?",
            "Aký je rozdiel medzi forward a historickým testom?",
            "Can you explain forward model performance?",
            "How does the forward model work?",
        )
        for question in questions:
            with self.subTest(question=question):
                self.assertIsNone(select_verified_tool(
                    question, proposal("forward_v2", "performance"),
                    historical_available=True, forward_available=True,
                ))

    def test_actual_forward_metrics_question_still_routes(self):
        selected = select_verified_tool(
            "Ako sa ti darí vo forward teste?",
            proposal("forward_v2", "performance"),
            historical_available=True, forward_available=True,
        )
        self.assertIsNotNone(selected)
        self.assertEqual(selected[0], "forward_v2")

    def test_general_education_cannot_open_historical_report(self):
        self.assertIsNone(select_verified_tool(
            "Vysvetli mi historickú validáciu modelu v3.",
            proposal("historical_v3", "overview"),
            historical_available=True, forward_available=True,
        ))


if __name__ == "__main__":
    unittest.main()
