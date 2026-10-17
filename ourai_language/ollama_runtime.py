"""Local Ollama chat backend for OurAI, sharing the existing grounding contract.

Uses the loopback-only Ollama HTTP endpoint. Does not call hosted Ollama
services, authenticate, download models automatically, or execute trades.
"""
from __future__ import annotations

import json
from typing import Iterable, Mapping
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .analysis_answers import deterministic_analysis_answer
from .contracts import AnalysisSnapshot
from .general_answers import deterministic_social_answer
from .grounding import find_unsupported_percentages, grounded_fallback
from .prompting import (
    ANALYSIS_QA, GENERAL_CHAT, build_messages, select_conversation_mode,
)
from .runtime import ChatResult


OLLAMA_ENDPOINT = "http://127.0.0.1:11434/api/chat"
DEFAULT_OLLAMA_MODEL = "qwen3:1.7b"


class LocalOllamaChatRuntime:
    """Deterministic, snapshot-grounded wrapper around a *local* Ollama model."""

    def __init__(
        self, *, model_id: str = DEFAULT_OLLAMA_MODEL,
        max_new_tokens: int = 384, timeout_seconds: int = 180,
    ) -> None:
        if not model_id or model_id.strip() != model_id:
            raise ValueError("a locally installed Ollama model name is required")
        if model_id.endswith(":cloud") or "://" in model_id or any(
            char.isspace() for char in model_id
        ):
            raise ValueError("only local Ollama model tags are supported")
        if max_new_tokens <= 0 or timeout_seconds <= 0:
            raise ValueError("positive max tokens and timeout are required")
        self.model_id = model_id
        self.max_new_tokens = int(max_new_tokens)
        self.timeout_seconds = int(timeout_seconds)
        self.device = "ollama-local"

    def _generate(self, messages: list[dict[str, str]]) -> str:
        payload = {
            "model": self.model_id,
            "messages": messages,
            "stream": False,
            "think": False,
            "options": {
                "temperature": 0.0,
                "num_ctx": 3072,
                "num_predict": self.max_new_tokens,
            },
        }
        request = Request(
            OLLAMA_ENDPOINT,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                data = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            if exc.code == 404:
                raise RuntimeError(
                    f"Ollama has no local model '{self.model_id}'. "
                    f"Run: ollama pull {self.model_id}"
                ) from exc
            raise RuntimeError(f"Local Ollama request failed (HTTP {exc.code}).") from exc
        except (URLError, TimeoutError, ConnectionError) as exc:
            raise RuntimeError(
                "Cannot reach the local Ollama server at 127.0.0.1:11434. "
                "Install and start Ollama for Windows first."
            ) from exc
        except (UnicodeError, ValueError) as exc:
            raise RuntimeError("Ollama returned invalid JSON.") from exc

        if not isinstance(data, dict) or not isinstance(data.get("message"), dict):
            raise RuntimeError("Ollama returned an unexpected response structure.")
        content = data["message"].get("content")
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("Ollama returned an empty text answer.")
        text = content.strip()
        if data.get("done_reason") == "length":
            text += "\n[Odpoveď bola skrátená limitom dĺžky.]"
        return text

    def respond(
        self, user_message: str, *, analysis: AnalysisSnapshot | None = None,
        history: Iterable[Mapping[str, str]] | None = None,
    ) -> ChatResult:
        history_rows = list(history or ())
        mode = select_conversation_mode(
            user_message, analysis=analysis, history=history_rows,
        )
        if analysis is not None and mode == ANALYSIS_QA:
            answer = deterministic_analysis_answer(user_message, analysis)
            if answer is not None:
                return ChatResult(text=answer, used_grounding_fallback=False, mode=mode)
        if mode == GENERAL_CHAT:
            social = deterministic_social_answer(user_message)
            if social is not None:
                return ChatResult(text=social, used_grounding_fallback=False, mode=mode)

        messages = build_messages(
            user_message, analysis=analysis, history=history_rows, mode=mode,
        )
        reply = self._generate(messages)
        if analysis is not None and mode == ANALYSIS_QA:
            unsupported = find_unsupported_percentages(reply, analysis)
            if unsupported:
                return ChatResult(
                    text=grounded_fallback(analysis),
                    used_grounding_fallback=True,
                    unsupported_percentages=unsupported,
                    mode=mode,
                )
        return ChatResult(text=reply, used_grounding_fallback=False, mode=mode)
