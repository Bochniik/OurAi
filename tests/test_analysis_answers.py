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
        self.assertIn("68.0 %", answer)
        self.assertIn("32.0 %", answer)
        self.assertIn("positive short-term momentum", answer)
        self.assertEqual(find_unsupported_percentages(answer or "", snapshot), ())

    def test_risk_answer_does_not_repeat_uncertainty(self) -> None:
        answer = deterministic_analysis_answer(
            "Aké sú najväčšie riziká tejto predikcie?",
            self.snapshot(),
        )
        self.assertIsNotNone(answer)
        self.assertIn("medium uncertainty", answer)
        self.assertIn("chýba výrazný potvrdený katalyzátor", answer)
        self.assertEqual((answer or "").count("medium uncertainty"), 1)

    def test_action_answer_is_bias_not_trade_order(self) -> None:
        answer = deterministic_analysis_answer(
            "Povedz mi jednoducho, čo by si na základe tejto analýzy robil.",
            self.snapshot(),
        )
        self.assertIsNotNone(answer)
        self.assertIn("LONG_BIAS", answer)
        self.assertIn("automatický príkaz na obchod", answer)

    def test_unknown_open_ended_question_falls_through(self) -> None:
        answer = deterministic_analysis_answer(
            "Vysvetli mi rozdiel medzi betou a alfou.",
            self.snapshot(),
        )
        self.assertIsNone(answer)


if __name__ == "__main__":
    unittest.main()
