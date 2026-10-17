from __future__ import annotations

import json
import unittest
from unittest.mock import patch
from urllib.error import URLError

from ourai_language.contracts import AnalysisSnapshot
from ourai_language.ollama_runtime import (
    DEFAULT_OLLAMA_MODEL, OLLAMA_ENDPOINT, LocalOllamaChatRuntime,
)


class _Response:
    def __init__(self, payload: dict):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


class LocalOllamaTests(unittest.TestCase):
    def snapshot(self):
        return AnalysisSnapshot(
            ticker="NVDA",
            company_name="NVIDIA Corporation",
            horizon_minutes=15,
            p_up=0.68,
            p_down=0.32,
            expected_move_pct=0.41,
            uncertainty="medium",
            decision="LONG_BIAS",
            reasons_for=("positive momentum",),
        )

    def test_default_is_local_quantized_model(self):
        runtime = LocalOllamaChatRuntime()
        self.assertEqual(runtime.model_id, "qwen3:1.7b")
        self.assertEqual(DEFAULT_OLLAMA_MODEL, "qwen3:1.7b")

    def test_refuses_cloud_tags_or_remote_urls(self):
        for tag in ("qwen3:cloud", "https://ollama.com/model", "model name", ""):
            with self.assertRaises(ValueError):
                LocalOllamaChatRuntime(model_id=tag)

    def test_local_api_payload_and_general_answer(self):
        with patch("ourai_language.ollama_runtime.urlopen") as urlopen:
            urlopen.return_value = _Response({
                "done": True, "done_reason": "stop",
                "message": {"role": "assistant", "content": "Fundamentálna skúma podnik; technická ceny."},
            })
            result = LocalOllamaChatRuntime().respond(
                "Vysvetli mi rozdiel medzi fundamentálnou a technickou analýzou."
            )
        self.assertIn("technická ceny", result.text)
        args, kwargs = urlopen.call_args
        request = args[0]
        self.assertEqual(request.full_url, OLLAMA_ENDPOINT)
        self.assertEqual(request.get_method(), "POST")
        payload = json.loads(request.data.decode("utf-8"))
        self.assertEqual(payload["model"], "qwen3:1.7b")
        self.assertFalse(payload["stream"])
        self.assertFalse(payload["think"])
        self.assertEqual(payload["options"]["temperature"], 0.0)
        self.assertEqual(payload["options"]["num_ctx"], 3072)
        self.assertEqual(payload["messages"][-1]["role"], "user")
        self.assertNotIn("CANONICAL_ANALYSIS_JSON", payload["messages"][0]["content"])
        self.assertEqual(kwargs["timeout"], 180)

    def test_known_analysis_and_social_questions_work_without_api(self):
        runtime = LocalOllamaChatRuntime()
        with patch("ourai_language.ollama_runtime.urlopen") as urlopen:
            result = runtime.respond("Prečo si pri NVDA bullish?", analysis=self.snapshot())
            social = runtime.respond("Ako sa máš?", analysis=self.snapshot())
        urlopen.assert_not_called()
        self.assertIn("68,0 %", result.text)
        self.assertIn("Som pripravený pomôcť.", social.text)

    def test_open_analysis_reply_remains_number_grounded(self):
        runtime = LocalOllamaChatRuntime()
        with patch("ourai_language.ollama_runtime.urlopen") as urlopen:
            urlopen.return_value = _Response({
                "done": True, "message": {"role": "assistant", "content": "Úspešnosť: 91 %."}
            })
            result = runtime.respond(
                "Vysvetli earnings yield v kontexte tejto analýzy NVDA.",
                analysis=self.snapshot(),
            )
        self.assertTrue(result.used_grounding_fallback)
        self.assertEqual(result.unsupported_percentages, (91.0,))
        self.assertIn("68.0%", result.text)

    def test_timeout_explains_slow_generation_not_server_failure(self):
        runtime = LocalOllamaChatRuntime(timeout_seconds=4)
        with patch("ourai_language.ollama_runtime.urlopen", side_effect=TimeoutError("timed out")):
            with self.assertRaisesRegex(RuntimeError, "timed out after 4 seconds"):
                runtime.respond("Čo je likvidita akcie?")

    def test_wrapped_url_error_timeout_is_diagnosed(self):
        runtime = LocalOllamaChatRuntime(timeout_seconds=4)
        with patch("ourai_language.ollama_runtime.urlopen", side_effect=URLError(TimeoutError("timed out"))):
            with self.assertRaisesRegex(RuntimeError, "timed out after 4 seconds"):
                runtime.respond("Čo je likvidita akcie?")

    def test_missing_server_reports_actionable_message(self):
        runtime = LocalOllamaChatRuntime()
        with patch("ourai_language.ollama_runtime.urlopen", side_effect=URLError("refused")):
            with self.assertRaisesRegex(RuntimeError, "Install and start Ollama"):
                runtime.respond("Čo je likvidita akcie?")

    def test_truncation_is_marked(self):
        runtime = LocalOllamaChatRuntime()
        with patch("ourai_language.ollama_runtime.urlopen") as urlopen:
            urlopen.return_value = _Response({
                "done": True, "done_reason": "length",
                "message": {"content": "Príliš krátke vysvetlenie"},
            })
            answer = runtime.respond("Čo je likvidita akcie?").text
        self.assertIn("skrátená", answer)


if __name__ == "__main__":
    unittest.main()
