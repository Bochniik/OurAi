from __future__ import annotations

import unittest

from ourai_language.general_answers import deterministic_social_answer
from ourai_language.prompting import GENERAL_CHAT, select_conversation_mode
from ourai_language.contracts import AnalysisSnapshot


class GeneralAnswersTests(unittest.TestCase):
    def test_exact_slovak_smalltalk_has_natural_reply(self) -> None:
        for question in ("Ako sa máš?", "ako sa máš ?", "Ako sa mas", "AKO SA TI DARÍ?"):
            answer = deterministic_social_answer(question)
            self.assertEqual(answer, "Som pripravený pomôcť. Čo by si chcel vedieť?")
            self.assertNotIn("Hvala", answer)

    def test_greetings_and_thanks_do_not_hijack_market_data(self) -> None:
        self.assertEqual(deterministic_social_answer("ahoj"), "Ahoj! S čím ti môžem pomôcť?")
        self.assertEqual(deterministic_social_answer("ďakujem"), "Rado sa stalo. S čím ti ešte môžem pomôcť?")
        analysis = AnalysisSnapshot(
            ticker="NVDA", horizon_minutes=15, p_up=0.68, p_down=0.32,
            expected_move_pct=0.41, uncertainty="medium", decision="LONG_BIAS",
        )
        self.assertEqual(select_conversation_mode("Ako sa máš?", analysis=analysis), GENERAL_CHAT)

    def test_open_ended_questions_keep_the_model_path(self) -> None:
        for question in ("Ako funguje VWAP?", "Aké je hlavné mesto Francúzska?", "Ahoj, vysvetli mi beta."):
            self.assertIsNone(deterministic_social_answer(question))

    def test_english_social_fallback(self) -> None:
        self.assertEqual(deterministic_social_answer("How are you?"), "I'm ready to help. What would you like to know?")


if __name__ == "__main__":
    unittest.main()
