from __future__ import annotations

import unittest

from ourai_language.analysis_answers import deterministic_analysis_answer
from ourai_language.contracts import AnalysisSnapshot
from ourai_language.grounding import find_unsupported_percentages


class DeterministicAnalysisAnswerTests(unittest.TestCase):
    def snapshot(self) -> AnalysisSnapshot:
        return AnalysisSnapshot(
            ticker="NVDA",
            company_name="NVIDIA Corporation",
            horizon_minutes=15,
            p_up=0.68,
            p_down=0.32,
            expected_move_pct=0.41,
            confidence=0.64,
            uncertainty="medium",
            decision="LONG_BIAS",
            market_regime="bullish",
            relative_strength="positive vs QQQ",
            volume_state="elevated",
            news_context="no strong catalyst supplied",
            reasons_for=(
                "positive short-term momentum",
                "relative strength versus QQQ",
                "elevated volume",
            ),
            reasons_against=("medium uncertainty",),
        )

    def test_why_answer_is_grounded_and_natural(self) -> None:
        snapshot = self.snapshot()
        answer = deterministic_analysis_answer("Prečo si pri NVDA bullish?", snapshot)
        self.assertIsNotNone(answer)
        self.assertIn("NVDA (NVIDIA Corporation)", answer)
        self.assertIn("68,0 %", answer)
        self.assertIn("32,0 %", answer)
        self.assertIn("pozitívne krátkodobé momentum", answer)
        self.assertIn("relatívna sila voči QQQ", answer)
        self.assertIn("zvýšený objem obchodovania", answer)
        self.assertNotIn("medium", answer)
        self.assertNotIn("positive short-term momentum", answer)
        self.assertEqual(find_unsupported_percentages(answer or "", snapshot), ())

    def test_risk_answer_does_not_repeat_uncertainty(self) -> None:
        answer = deterministic_analysis_answer(
            "Aké sú najväčšie riziká tejto predikcie?",
            self.snapshot(),
        )
        self.assertIsNotNone(answer)
        self.assertIn("stredná neistota modelu", answer)
        self.assertIn("v dodaných údajoch nie je potvrdený výrazný katalyzátor", answer)
        self.assertEqual((answer or "").count("stredná neistota"), 1)
        self.assertEqual((answer or "").count("katalyzátor"), 1)
        self.assertNotIn("no strong", answer)
        self.assertEqual(find_unsupported_percentages(answer or "", self.snapshot()), ())

    def test_action_answer_is_bias_not_trade_order(self) -> None:
        answer = deterministic_analysis_answer(
            "Povedz mi jednoducho, čo by si na základe tejto analýzy robil.",
            self.snapshot(),
        )
        self.assertIsNotNone(answer)
        self.assertIn("LONG_BIAS", answer)
        self.assertIn("automatický príkaz na obchod", answer)
        self.assertIn("68,0 %", answer)
        self.assertEqual(find_unsupported_percentages(answer or "", self.snapshot()), ())


    def test_probability_answer_explains_absolute_move(self) -> None:
        snapshot = self.snapshot()
        answer = deterministic_analysis_answer("Aká je pravdepodobnosť rastu?", snapshot)
        self.assertIsNotNone(answer)
        self.assertIn("68,0 %", answer)
        self.assertIn("0,41 %", answer)
        self.assertIn("nie", answer)
        self.assertEqual(find_unsupported_percentages(answer or "", snapshot), ())

    def test_unknown_descriptors_are_quoted_not_reinterpreted(self) -> None:
        from dataclasses import replace
        snapshot = replace(self.snapshot(), reasons_for=("custom-feature-42",))
        answer = deterministic_analysis_answer("Prečo si bullish?", snapshot)
        self.assertIn("„custom-feature-42“", answer or "")

    def test_no_trade_is_not_a_trade_instruction(self) -> None:
        from dataclasses import replace
        snapshot = replace(self.snapshot(), decision="NO_TRADE")
        answer = deterministic_analysis_answer("Čo by si robil?", snapshot)
        self.assertIn("NO_TRADE", answer or "")
        self.assertIn("Nie je to automatický príkaz na obchod", answer or "")

    def test_unknown_open_ended_question_falls_through(self) -> None:
        answer = deterministic_analysis_answer(
            "Vysvetli mi rozdiel medzi betou a alfou.",
            self.snapshot(),
        )
        self.assertIsNone(answer)


if __name__ == "__main__":
    unittest.main()
