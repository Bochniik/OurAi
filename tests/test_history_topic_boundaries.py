"""Topic boundary regressions for a verified historical answer.

These checks exercise deterministic routing without a local Qwen installation.
"""
from __future__ import annotations

import unittest

from ourai_language.conversation_followups import historical_followup_question


HISTORY = (
    {"role": "user", "content": "Ako dopadla historická validácia v3?"},
    {
        "role": "assistant",
        "source": "historical_v3_reports",
        "content": "Overené výsledky.",
        "evidence_question": "Ako dopadla historická validácia v3?",
    },
)


class HistoryTopicBoundaryTests(unittest.TestCase):
    def test_new_topics_do_not_open_historical_report(self):
        for question in (
            "O koľko je drahší notebook oproti tabletu?",
            "A o koľko sa líši cena benzínu?",
            "A oproti minulému roku?",
            "V porovnaní s priemernou mzdou?",
            "A v porovnaní s cenou domu?",
        ):
            with self.subTest(question=question):
                self.assertIsNone(historical_followup_question(question, HISTORY))

    def test_referential_analytical_comparisons_remain_valid(self):
        for question in (
            "O koľko je to lepšie?",
            "A o koľko sa líši MAE modelu v3?",
            "A oproti lineárnemu modelu?",
            "A oproti nulovému baseline?",
            "V porovnaní s lineárnym modelom?",
        ):
            with self.subTest(question=question):
                self.assertIsNotNone(historical_followup_question(question, HISTORY))

    def test_untagged_history_is_never_evidence(self):
        unverified = list(HISTORY)
        unverified[-1] = {**unverified[-1], "source": "model"}
        self.assertIsNone(historical_followup_question("O koľko je to lepšie?", unverified))


if __name__ == "__main__":
    unittest.main()
