from __future__ import annotations

import unittest
from types import SimpleNamespace

from scripts.benchmark_ourai_language import CASES, evaluate


class _FakeRuntime:
    def __init__(self):
        self.calls = []

    def respond(self, prompt, *, history):
        self.calls.append((prompt, history))
        return SimpleNamespace(text="Odpoveď na zadanú otázku.")


class LanguageBenchmarkTests(unittest.TestCase):
    def test_case_list_covers_the_rejected_fundamental_answer(self):
        self.assertEqual(len(CASES), 8)
        self.assertIn("fundamentálnou a technickou", CASES[0][1])
        self.assertEqual(len({case_id for case_id, _ in CASES}), len(CASES))

    def test_manual_benchmark_stores_raw_responses_without_grades(self):
        runtime = _FakeRuntime()
        cases = CASES[:2]
        rows = evaluate(runtime, cases)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["case_id"], "finance_fundamental_technical")
        self.assertEqual(rows[0]["response"], "Odpoveď na zadanú otázku.")
        self.assertGreaterEqual(rows[0]["latency_seconds"], 0.0)
        self.assertEqual(rows[0]["error"], None)
        self.assertEqual(runtime.calls[0], (cases[0][1], []))


if __name__ == "__main__":
    unittest.main()
