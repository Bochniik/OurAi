"""Local Ollama chat backend for OurAI, sharing the existing grounding contract.

Uses the loopback-only Ollama HTTP endpoint. Does not call hosted Ollama
services, authenticate, download models automatically, or execute trades.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Mapping
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .analysis_answers import deterministic_analysis_answer
from .contracts import AnalysisSnapshot
from .general_answers import deterministic_social_answer
from .financial_knowledge import lookup_financial_knowledge
from .forward_answers import maybe_forward_answer
from .historical_answers import maybe_historical_answer
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
        num_ctx: int = 3072, concise: bool = False,
        knowledge_enabled: bool = True,
        forward_journal: str | Path | None = None,
        historical_reports_dir: str | Path | None = None,
    ) -> None:
        if not model_id or model_id.strip() != model_id:
            raise ValueError("a locally installed Ollama model name is required")
        if model_id.endswith(":cloud") or "://" in model_id or any(
            char.isspace() for char in model_id
        ):
            raise ValueError("only local Ollama model tags are supported")
        if max_new_tokens <= 0 or timeout_seconds <= 0:
            raise ValueError("positive max tokens and timeout are required")
        if not isinstance(num_ctx, int) or isinstance(num_ctx, bool) or not 512 <= num_ctx <= 32768:
            raise ValueError("num_ctx must be an integer between 512 and 32768")
        self.model_id = model_id
        self.max_new_tokens = int(max_new_tokens)
        self.timeout_seconds = int(timeout_seconds)
        self.num_ctx = num_ctx
        self.concise = bool(concise)
        self.knowledge_enabled = bool(knowledge_enabled)
        self.forward_journal = forward_journal
        self.historical_reports_dir = historical_reports_dir
        self.last_response_metrics: dict[str, int | str] | None = None
        self.device = "ollama-local"

    def _generate(self, messages: list[dict[str, str]]) -> str:
        payload = {
            "model": self.model_id,
            "messages": messages,
            "stream": False,
            "think": False,
            "options": {
                "temperature": 0.0,
                "num_ctx": self.num_ctx,
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
        except TimeoutError as exc:
            raise RuntimeError(
                f"Local Ollama request timed out after {self.timeout_seconds} seconds. "
                "The server may be running but the model is loading or generating "
                "too slowly. Run 'ollama ps' to check CPU/GPU use, reduce "
                "'--max-new-tokens', or raise '--timeout-seconds'."
            ) from exc
        except URLError as exc:
            if isinstance(exc.reason, TimeoutError):
                raise RuntimeError(
                    f"Local Ollama request timed out after {self.timeout_seconds} seconds. "
                    "The server may be running but the model is loading or generating "
                    "too slowly. Run 'ollama ps', reduce '--max-new-tokens', "
                    "or raise '--timeout-seconds'."
                ) from exc
            raise RuntimeError(
                "Cannot reach the local Ollama server at 127.0.0.1:11434. "
                "Install and start Ollama for Windows first."
            ) from exc
        except ConnectionError as exc:
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
        self.last_response_metrics = {
            key: data[key] for key in (
                "load_duration", "prompt_eval_count", "prompt_eval_duration",
                "eval_count", "eval_duration", "total_duration", "done_reason",
            )
            if key in data and (
                (key == "done_reason" and isinstance(data[key], str))
                or (key != "done_reason" and type(data[key]) is int and data[key] >= 0)
            )
        }
        text = content.strip()
        if data.get("done_reason") == "length":
            text += "\n[Odpoveď bola skrátená limitom dĺžky.]"
        return text

    def respond(
        self, user_message: str, *, analysis: AnalysisSnapshot | None = None,
        history: Iterable[Mapping[str, str]] | None = None,
    ) -> ChatResult:
        self.last_response_metrics = None
        history_rows = list(history or ())
        mode = select_conversation_mode(
            user_message, analysis=analysis, history=history_rows,
        )
        if self.knowledge_enabled:
            curated = lookup_financial_knowledge(user_message, analysis=analysis)
            if curated is not None:
                return ChatResult(
                    text=curated.text, used_grounding_fallback=False,
                    mode=mode, answer_source="curated_knowledge",
                )

        if self.historical_reports_dir is not None:
            try:
                answer = maybe_historical_answer(user_message, self.historical_reports_dir)
            except (ValueError, OSError, KeyError, TypeError) as exc:
                raise RuntimeError(
                    "Historický report v3 je neplatný alebo nedostupný; "
                    "odmietam odhadovať neoverené metriky."
                ) from exc
            if answer is not None:
                return ChatResult(
                    text=answer, used_grounding_fallback=False,
                    mode=ANALYSIS_QA, answer_source="historical_v3_reports",
                )

        if self.forward_journal is not None:
            try:
                answer = maybe_forward_answer(user_message, self.forward_journal)
            except (ValueError, OSError, KeyError, TypeError) as exc:
                raise RuntimeError(
                    "Forward denník je neplatný alebo dočasne nedostupný; "
                    "odmietam odhadovať nevykázané výsledky."
                ) from exc
            if answer is not None:
                return ChatResult(
                    text=answer, used_grounding_fallback=False,
                    mode=ANALYSIS_QA, answer_source="verified_forward_journal",
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
        if self.concise:
            messages[0] = {
                **messages[0],
                "content": messages[0]["content"]
                + "\nStyle: Answer directly in three to five SHORT sentences by default. "
                "No greetings, headings, bullet lists or long introductions unless "
                "explicitly requested. Keep financial explanations accurate, state "
                "uncertainty, and do not invent prices or forecast probabilities.",
            }
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
