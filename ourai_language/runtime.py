from __future__ import annotations

import re
from pathlib import Path
from dataclasses import dataclass
from typing import Iterable, Mapping

from .analysis_answers import deterministic_analysis_answer
from .contracts import AnalysisSnapshot
from .general_answers import deterministic_social_answer
from .financial_knowledge import lookup_financial_knowledge
from .forward_answers import maybe_forward_answer
from .historical_answers import maybe_historical_answer
from .grounding import find_unsupported_percentages, grounded_fallback
from .prompting import ANALYSIS_QA, GENERAL_CHAT, build_messages, select_conversation_mode


DEFAULT_MODEL_ID = "Qwen/Qwen3-0.6B"
_THINK_RE = re.compile(r"<think>.*?</think>\s*", flags=re.DOTALL | re.IGNORECASE)


@dataclass(frozen=True)
class ChatResult:
    text: str
    used_grounding_fallback: bool
    unsupported_percentages: tuple[float, ...] = ()
    mode: str = "GENERAL_CHAT"
    answer_source: str = "model"


class OurAIChatRuntime:
    """Local conversational runtime for Orion/OurAI.

    Heavy ML dependencies are imported lazily so the trading-only test suite and CI
    do not need to install a language-model stack. The default 0.6B model is small
    enough for local inference while still starting from a real multilingual instruct
    model rather than the historical scratch-trained OurAI prototype.

    Common snapshot questions are answered by a deterministic grounded explainer
    before generation. Open-ended analysis and ordinary conversation still fall
    through to the local language model.
    """

    def __init__(
        self,
        *,
        model_id: str = DEFAULT_MODEL_ID,
        device: str = "auto",
        max_new_tokens: int = 256,
        knowledge_enabled: bool = True,
        forward_journal: str | Path | None = None,
        historical_reports_dir: str | Path | None = None,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "OurAI language runtime dependencies are missing. "
                "Install requirements-language.txt after installing the CUDA-enabled "
                "PyTorch build appropriate for this machine."
            ) from exc

        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")

        resolved_device = device
        if resolved_device == "auto":
            resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
        if resolved_device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
        if resolved_device not in {"cuda", "cpu"}:
            raise ValueError("device must be auto, cuda, or cpu")

        dtype = torch.float16 if resolved_device == "cuda" else torch.float32
        self.torch = torch
        self.device = resolved_device
        self.model_id = str(model_id)
        self.max_new_tokens = int(max_new_tokens)
        self.knowledge_enabled = bool(knowledge_enabled)
        self.forward_journal = forward_journal
        self.historical_reports_dir = historical_reports_dir
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            dtype=dtype,
            low_cpu_mem_usage=True,
        )
        self.model.to(self.device)
        self.model.eval()

        # Qwen ships sampling defaults in generation_config. We intentionally use
        # deterministic greedy decoding for grounded analysis, so clear incompatible
        # sampling-only flags instead of letting transformers warn on every turn.
        self.model.generation_config.do_sample = False
        self.model.generation_config.temperature = None
        self.model.generation_config.top_p = None
        self.model.generation_config.top_k = None

    def _render_chat(self, messages: list[dict[str, str]]) -> str:
        kwargs = {
            "tokenize": False,
            "add_generation_prompt": True,
        }
        try:
            return self.tokenizer.apply_chat_template(
                messages,
                enable_thinking=False,
                **kwargs,
            )
        except (TypeError, ValueError):
            return self.tokenizer.apply_chat_template(messages, **kwargs)

    def respond(
        self,
        user_message: str,
        *,
        analysis: AnalysisSnapshot | None = None,
        history: Iterable[Mapping[str, str]] | None = None,
    ) -> ChatResult:
        history_rows = list(history or ())
        mode = select_conversation_mode(
            user_message,
            analysis=analysis,
            history=history_rows,
        )

        if getattr(self, "knowledge_enabled", True):
            curated = lookup_financial_knowledge(user_message, analysis=analysis)
            if curated is not None:
                return ChatResult(
                    text=curated.text, used_grounding_fallback=False,
                    mode=mode, answer_source="curated_knowledge",
                )

        if getattr(self, "historical_reports_dir", None) is not None:
            try:
                historical = maybe_historical_answer(user_message, self.historical_reports_dir)
            except (ValueError, OSError, KeyError, TypeError) as exc:
                raise RuntimeError(
                    "Historický report v3 je neplatný alebo nedostupný; "
                    "odmietam odhadovať neoverené metriky."
                ) from exc
            if historical is not None:
                return ChatResult(
                    text=historical, used_grounding_fallback=False,
                    mode=ANALYSIS_QA, answer_source="historical_v3_reports",
                )

        if getattr(self, "forward_journal", None) is not None:
            try:
                forward = maybe_forward_answer(user_message, self.forward_journal)
            except (ValueError, OSError, KeyError, TypeError) as exc:
                raise RuntimeError(
                    "Forward denník je neplatný alebo dočasne nedostupný; "
                    "odmietam odhadovať nevykázané výsledky."
                ) from exc
            if forward is not None:
                return ChatResult(
                    text=forward, used_grounding_fallback=False,
                    mode=ANALYSIS_QA, answer_source="verified_forward_journal",
                )

        # Core Orion analysis questions should be maximally reliable. If the answer
        # can be constructed directly from the canonical snapshot, do that instead of
        # asking a tiny language model to paraphrase facts it may distort.
        if analysis is not None and mode == ANALYSIS_QA:
            deterministic = deterministic_analysis_answer(user_message, analysis)
            if deterministic is not None:
                return ChatResult(
                    text=deterministic,
                    used_grounding_fallback=False,
                    mode=mode,
                )

        # Exact social phrases should not rely on a tiny multilingual model
        # that sometimes responds in the wrong language. Open-ended general
        # conversation still uses the local model unchanged.
        if mode == GENERAL_CHAT:
            social = deterministic_social_answer(user_message)
            if social is not None:
                return ChatResult(
                    text=social,
                    used_grounding_fallback=False,
                    mode=mode,
                )

        messages = build_messages(
            user_message,
            analysis=analysis,
            history=history_rows,
            mode=mode,
        )
        prompt = self._render_chat(messages)
        encoded = self.tokenizer(prompt, return_tensors="pt")
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        prompt_length = int(encoded["input_ids"].shape[1])

        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id

        with self.torch.inference_mode():
            generated = self.model.generate(
                **encoded,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                repetition_penalty=1.05,
                pad_token_id=pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        new_tokens = generated[0, prompt_length:]
        reply = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        reply = _THINK_RE.sub("", reply).strip()
        if not reply:
            reply = "The language model returned an empty response."

        if analysis is not None and mode == ANALYSIS_QA:
            unsupported = find_unsupported_percentages(reply, analysis)
            if unsupported:
                return ChatResult(
                    text=grounded_fallback(analysis),
                    used_grounding_fallback=True,
                    unsupported_percentages=unsupported,
                    mode=mode,
                )

        return ChatResult(
            text=reply,
            used_grounding_fallback=False,
            mode=mode,
        )
