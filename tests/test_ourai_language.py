from __future__ import annotations

import unittest

from ourai_language.contracts import AnalysisSnapshot
from ourai_language.grounding import (
    canonical_percentage_values,
    find_unsupported_percentages,
    grounded_fallback,
)
from ourai_language.prompting import build_messages


class OurAILanguageTests(unittest.TestCase):
    def snapshot(self) -> AnalysisSnapshot:
        return AnalysisSnapshot(
            ticker="nvda",
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
            reasons_for=("positive momentum", "relative strength"),
            reasons_against=("medium uncertainty",),
        )

    def test_snapshot_normalizes_and_validates(self) -> None:
        snapshot = self.snapshot()
        self.assertEqual(snapshot.ticker, "NVDA")
        self.assertEqual(snapshot.decision, "LONG_BIAS")
        self.assertAlmostEqual(snapshot.p_up + snapshot.p_down, 1.0)
        self.assertEqual(snapshot.reasons_for, ("positive momentum", "relative strength"))

    def test_probability_sum_is_enforced(self) -> None:
        with self.assertRaises(ValueError):
            AnalysisSnapshot(
                ticker="AAPL",
                horizon_minutes=15,
                p_up=0.70,
                p_down=0.40,
                expected_move_pct=0.20,
                uncertainty="low",
                decision="LONG_BIAS",
            )

    def test_prompt_contains_canonical_snapshot(self) -> None:
        messages = build_messages(
            "Prečo si bullish?",
            analysis=self.snapshot(),
        )
        self.assertEqual(messages[0]["role"], "system")
        self.assertIn("CANONICAL_ANALYSIS_JSON", messages[0]["content"])
        self.assertIn('"ticker":"NVDA"', messages[0]["content"])
        self.assertEqual(messages[-1], {"role": "user", "content": "Prečo si bullish?"})

    def test_grounding_guard_accepts_only_snapshot_percentages(self) -> None:
        snapshot = self.snapshot()
        self.assertEqual(
            canonical_percentage_values(snapshot),
            (68.0, 32.0, 0.41, 64.0),
        )
        reply = "P(UP) je 68 %, P(DOWN) 32 % a očakávaný pohyb 0,41 %."
        self.assertEqual(find_unsupported_percentages(reply, snapshot), ())

        bad_reply = "P(UP) je 68 %, ale úspešnosť je 91 %."
        self.assertEqual(find_unsupported_percentages(bad_reply, snapshot), (91.0,))

    def test_fallback_uses_only_canonical_values(self) -> None:
        snapshot = self.snapshot()
        fallback = grounded_fallback(snapshot)
        self.assertIn("P(UP)=68.0%", fallback)
        self.assertIn("P(DOWN)=32.0%", fallback)
        self.assertIn("expected move=0.41%", fallback)
        self.assertIn("decision=LONG_BIAS", fallback)
        self.assertEqual(find_unsupported_percentages(fallback, snapshot), ())


if __name__ == "__main__":
    unittest.main()
