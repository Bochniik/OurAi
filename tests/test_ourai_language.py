from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

from ourai_language.ollama_runtime import LocalOllamaChatRuntime
from ourai_language.runtime import OurAIChatRuntime, snapshot_mismatch_result
from ourai_language.contracts import AnalysisSnapshot
from ourai_language.grounding import (
    canonical_percentage_values,
    find_unsupported_percentages,
    grounded_fallback,
)
from ourai_language.prompting import (
    ANALYSIS_QA,
    GENERAL_CHAT,
    build_messages,
    explicit_snapshot_ticker_mismatch,
    select_conversation_mode,
)


class OurAILanguageTests(unittest.TestCase):
    def snapshot(self) -> AnalysisSnapshot:
        return AnalysisSnapshot(
            ticker="nvda",
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
            reasons_for=("positive momentum", "relative strength", "positive momentum"),
            reasons_against=("no strong catalyst",),
        )

    def test_snapshot_normalizes_validates_and_dedupes(self) -> None:
        snapshot = self.snapshot()
        self.assertEqual(snapshot.ticker, "NVDA")
        self.assertEqual(snapshot.company_name, "NVIDIA Corporation")
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

    def test_analysis_prompt_contains_identity_fact_sheet_and_snapshot(self) -> None:
        messages = build_messages(
            "Prečo si pri NVDA bullish?",
            analysis=self.snapshot(),
        )
        self.assertEqual(messages[0]["role"], "system")
        self.assertIn("CANONICAL_ANALYSIS_FACTS", messages[0]["content"])
        self.assertIn("company_name: NVIDIA Corporation", messages[0]["content"])
        self.assertIn("CANONICAL_ANALYSIS_JSON", messages[0]["content"])
        self.assertIn('"ticker":"NVDA"', messages[0]["content"])
        self.assertEqual(messages[-1], {"role": "user", "content": "Prečo si pri NVDA bullish?"})

    def test_general_chat_does_not_receive_analysis_snapshot(self) -> None:
        messages = build_messages(
            "Ako sa máš?",
            analysis=self.snapshot(),
        )
        self.assertNotIn("CANONICAL_ANALYSIS_JSON", messages[0]["content"])
        self.assertNotIn("NVIDIA Corporation", messages[0]["content"])

    def test_router_separates_social_and_analysis_turns(self) -> None:
        snapshot = self.snapshot()
        self.assertEqual(
            select_conversation_mode("Prečo si pri NVDA bullish?", analysis=snapshot),
            ANALYSIS_QA,
        )
        self.assertEqual(
            select_conversation_mode("Aké sú riziká tejto predikcie?", analysis=snapshot),
            ANALYSIS_QA,
        )
        self.assertEqual(
            select_conversation_mode("Ako sa máš?", analysis=snapshot),
            GENERAL_CHAT,
        )
        self.assertEqual(
            select_conversation_mode("Aké je hlavné mesto Francúzska?", analysis=snapshot),
            GENERAL_CHAT,
        )

    def test_short_followup_can_inherit_analysis_mode_but_social_overrides(self) -> None:
        snapshot = self.snapshot()
        history = [
            {"role": "user", "content": "Prečo si bullish?", "mode": ANALYSIS_QA},
            {"role": "assistant", "content": "Kvôli momentu.", "mode": ANALYSIS_QA},
        ]
        self.assertEqual(
            select_conversation_mode("Prečo?", analysis=snapshot, history=history),
            ANALYSIS_QA,
        )
        self.assertEqual(
            select_conversation_mode("Ako sa máš?", analysis=snapshot, history=history),
            GENERAL_CHAT,
        )

    def test_short_unrelated_topic_breaks_snapshot_inheritance(self) -> None:
        snapshot = self.snapshot()
        history = [
            {"role": "user", "content": "Prečo si bullish?", "mode": ANALYSIS_QA},
            {"role": "assistant", "content": "Podľa snapshotu.", "mode": ANALYSIS_QA},
        ]
        for question in (
            "A čo Python?",
            "Ako funguje fotosyntéza?",
            "Kto bol Newton?",
            "A čo počasie?",
        ):
            with self.subTest(question=question):
                self.assertEqual(
                    select_conversation_mode(question, analysis=snapshot, history=history),
                    GENERAL_CHAT,
                )
                messages = build_messages(question, analysis=snapshot, history=history)
                self.assertNotIn("CANONICAL_ANALYSIS_FACTS", messages[0]["content"])
        for question in ("Prečo?", "A ako?", "A čo to znamená?"):
            with self.subTest(question=question):
                self.assertEqual(
                    select_conversation_mode(question, analysis=snapshot, history=history),
                    ANALYSIS_QA,
                )

    def test_multiturn_topic_break_does_not_inherit_old_analysis(self) -> None:
        snapshot = self.snapshot()
        history = [
            {"role": "user", "content": "Prečo si bullish?", "mode": ANALYSIS_QA},
            {"role": "assistant", "content": "Podľa snapshotu.", "mode": ANALYSIS_QA},
            {"role": "user", "content": "A čo Python?", "mode": GENERAL_CHAT},
            {"role": "assistant", "content": "Python je jazyk.", "mode": GENERAL_CHAT},
        ]
        self.assertEqual(
            select_conversation_mode("Prečo?", analysis=snapshot, history=history),
            GENERAL_CHAT,
        )
        self.assertEqual(
            select_conversation_mode("Aké sú riziká?", analysis=snapshot, history=history),
            ANALYSIS_QA,
        )

    def test_declared_general_mode_breaks_analysis_inheritance(self) -> None:
        snapshot = self.snapshot()
        # The vocabulary 'cena' is broadly finance-related, but this was
        # explicitly routed as ordinary chat (e.g. price of a textbook).
        history = [
            {"role": "user", "content": "Aká je cena učebnice?", "mode": GENERAL_CHAT},
            {"role": "assistant", "content": "Neviem bez aktuálneho cenníka.", "mode": GENERAL_CHAT},
        ]
        self.assertEqual(
            select_conversation_mode("Prečo?", analysis=snapshot, history=history),
            GENERAL_CHAT,
        )
        messages = build_messages("Prečo?", analysis=snapshot, history=history)
        self.assertNotIn("CANONICAL_ANALYSIS_FACTS", messages[0]["content"])

    def test_recorded_analytical_mode_supports_short_followup(self) -> None:
        snapshot = self.snapshot()
        # Not every meaningful analytical question uses a keyword from the
        # heuristic list. Recorded mode retains source continuity.
        history = [
            {"role": "user", "content": "Čím to vysvetľuješ?", "mode": ANALYSIS_QA},
            {"role": "assistant", "content": "Podľa snapshotu.", "mode": ANALYSIS_QA},
        ]
        self.assertEqual(
            select_conversation_mode("Prečo?", analysis=snapshot, history=history),
            ANALYSIS_QA,
        )

    def test_unlabelled_history_keeps_conservative_legacy_fallback(self) -> None:
        snapshot = self.snapshot()
        history = [
            {"role": "user", "content": "Aké sú riziká tejto predikcie?"},
            {"role": "assistant", "content": "Riziká existujú."},
        ]
        self.assertEqual(
            select_conversation_mode("Prečo?", analysis=snapshot, history=history),
            ANALYSIS_QA,
        )

    def test_history_is_filtered_by_mode(self) -> None:
        snapshot = self.snapshot()
        history = [
            {"role": "user", "content": "Prečo si bullish?", "mode": ANALYSIS_QA},
            {"role": "assistant", "content": "Analysis answer", "mode": ANALYSIS_QA},
            {"role": "user", "content": "Ahoj", "mode": GENERAL_CHAT},
            {"role": "assistant", "content": "Ahoj!", "mode": GENERAL_CHAT},
        ]
        messages = build_messages(
            "Ako sa máš?",
            analysis=snapshot,
            history=history,
        )
        contents = [item["content"] for item in messages]
        self.assertIn("Ahoj!", contents)
        self.assertNotIn("Analysis answer", contents)

    def test_return_to_current_snapshot_filters_previous_instrument(self) -> None:
        snapshot = self.snapshot()
        history = [
            {"role": "user", "content": "Analýza AAPL?", "mode": ANALYSIS_QA,
             "analysis_ticker": "AAPL"},
            {"role": "assistant", "content": "STARÁ AAPL PROGNÓZA", "mode": ANALYSIS_QA,
             "analysis_ticker": "AAPL"},
            {"role": "user", "content": "Ahoj", "mode": GENERAL_CHAT},
            {"role": "assistant", "content": "Ahoj!", "mode": GENERAL_CHAT},
            {"role": "user", "content": "Prečo je NVDA bullish?", "mode": ANALYSIS_QA,
             "analysis_ticker": "NVDA"},
            {"role": "assistant", "content": "AKTUÁLNY NVDA KONTEXT",
             "mode": ANALYSIS_QA, "analysis_ticker": "NVDA"},
        ]
        messages = build_messages("Vráťme sa k analýze NVDA.",
                                  analysis=snapshot, history=history)
        content = "\n".join(item["content"] for item in messages)
        self.assertIn("AKTUÁLNY NVDA KONTEXT", content)
        self.assertIn("CANONICAL_ANALYSIS_FACTS", content)
        self.assertNotIn("STARÁ AAPL PROGNÓZA", content)
        self.assertNotIn("Analýza AAPL?", content)
        self.assertNotIn("Ahoj!", content)

    def test_legacy_untagged_analysis_turns_remain_supported(self) -> None:
        snapshot = self.snapshot()
        history = [
            {"role": "user", "content": "Prečo si bullish?", "mode": ANALYSIS_QA},
            {"role": "assistant", "content": "Neoznačený starší kontext.",
             "mode": ANALYSIS_QA},
        ]
        messages = build_messages("Prečo?", analysis=snapshot, history=history)
        self.assertIn(
            "Neoznačený starší kontext.",
            [item["content"] for item in messages],
        )

    def test_short_followup_does_not_inherit_different_tagged_ticker(self) -> None:
        snapshot = self.snapshot()
        history = [
            {"role": "user", "content": "Prečo je AAPL bullish?",
             "mode": ANALYSIS_QA, "analysis_ticker": "AAPL"},
            {"role": "assistant", "content": "Analýza AAPL.",
             "mode": ANALYSIS_QA, "analysis_ticker": "AAPL"},
        ]
        self.assertEqual(
            select_conversation_mode("Prečo?", analysis=snapshot, history=history),
            GENERAL_CHAT,
        )
        messages = build_messages("Prečo?", analysis=snapshot, history=history)
        self.assertNotIn("CANONICAL_ANALYSIS_FACTS", messages[0]["content"])
        self.assertNotIn("Analýza AAPL.", [item["content"] for item in messages])

    def test_short_followup_keeps_matching_tagged_ticker(self) -> None:
        snapshot = self.snapshot()
        history = [
            {"role": "user", "content": "Prečo je NVDA bullish?",
             "mode": ANALYSIS_QA, "analysis_ticker": " nvda "},
            {"role": "assistant", "content": "Podľa snapshotu.",
             "mode": ANALYSIS_QA, "analysis_ticker": "NVDA"},
        ]
        self.assertEqual(
            select_conversation_mode("Prečo?", analysis=snapshot, history=history),
            ANALYSIS_QA,
        )
        messages = build_messages("Prečo?", analysis=snapshot, history=history)
        self.assertIn("CANONICAL_ANALYSIS_FACTS", messages[0]["content"])

    def test_short_followup_preserves_untagged_legacy_mode(self) -> None:
        snapshot = self.snapshot()
        history = [
            {"role": "user", "content": "Prečo je to bullish?", "mode": ANALYSIS_QA},
            {"role": "assistant", "content": "Podľa snapshotu.", "mode": ANALYSIS_QA},
        ]
        self.assertEqual(
            select_conversation_mode("Prečo?", analysis=snapshot, history=history),
            ANALYSIS_QA,
        )

    def test_explicit_other_ticker_does_not_borrow_nvda_snapshot(self) -> None:
        snapshot = self.snapshot()
        for question in (
            "Aká je predikcia pre AAPL?",
            "Aké sú riziká pri AAPL?",
            "Prečo je AAPL bullish?",
            "AAPL forecast?",
            "What is the forecast for AAPL?",
        ):
            with self.subTest(question=question):
                self.assertEqual(
                    explicit_snapshot_ticker_mismatch(question, snapshot), "AAPL",
                )
                result = snapshot_mismatch_result(question, snapshot)
                self.assertEqual(result.answer_source, "snapshot_ticker_mismatch")
                self.assertEqual(result.mode, GENERAL_CHAT)
                self.assertIn("AAPL", result.text)
                self.assertIn("NVDA", result.text)
                self.assertNotIn("68", result.text)

    def test_matching_ticker_and_metric_acronyms_are_not_rejected(self) -> None:
        snapshot = self.snapshot()
        for question in (
            "Aká je predikcia pre NVDA?",
            "Prečo je NVDA bullish?",
            "Čo znamená MAE?",
            "Vysvetli RSI pre NVDA.",
            "Aké riziko má ETF?",
            "Ako sa máš?",
        ):
            with self.subTest(question=question):
                self.assertIsNone(
                    explicit_snapshot_ticker_mismatch(question, snapshot)
                )

    def test_both_local_backends_fail_closed_before_model_generation(self) -> None:
        snapshot = self.snapshot()
        native = OurAIChatRuntime.__new__(OurAIChatRuntime)
        native.knowledge_enabled = False
        native.historical_reports_dir = None
        native.forward_journal = None
        ollama = LocalOllamaChatRuntime(knowledge_enabled=False)
        ollama._generate = Mock(side_effect=AssertionError("model should not run"))
        for runtime in (native, ollama):
            with self.subTest(runtime=type(runtime).__name__):
                result = runtime.respond("Aká je predikcia pre AAPL?", analysis=snapshot)
                self.assertEqual(result.answer_source, "snapshot_ticker_mismatch")
                self.assertNotIn("68", result.text)
        ollama._generate.assert_not_called()

    def test_verified_forward_reader_keeps_priority_over_ticker_guard(self) -> None:
        snapshot = self.snapshot()
        native = OurAIChatRuntime.__new__(OurAIChatRuntime)
        native.knowledge_enabled = False
        native.historical_reports_dir = None
        native.forward_journal = "verified-local-journal"
        ollama = LocalOllamaChatRuntime(
            knowledge_enabled=False, forward_journal="verified-local-journal"
        )
        for runtime, module in (
            (native, "ourai_language.runtime"),
            (ollama, "ourai_language.ollama_runtime"),
        ):
            with self.subTest(runtime=type(runtime).__name__):
                with patch(module + ".maybe_forward_answer", return_value="Verified AAPL record"):
                    result = runtime.respond(
                        "Aká je posledná predikcia pre AAPL?", analysis=snapshot
                    )
                self.assertEqual(result.text, "Verified AAPL record")
                self.assertEqual(result.answer_source, "verified_forward_journal")

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
