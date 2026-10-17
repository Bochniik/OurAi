from __future__ import annotations

import json
from typing import Iterable, Mapping

from .contracts import AnalysisSnapshot


SYSTEM_PROMPT = """You are OurAI, the conversational analyst inside OrionTrader.

Your job is to explain market-analysis outputs clearly and answer the user's questions.
You are not allowed to invent forecast numbers, prices, probabilities, expected moves,
confidence values, news, or market-state facts. When a canonical analysis snapshot is
provided, treat it as the only source of truth for forecast-specific facts.

Rules:
- Reply in the language used by the user unless they ask for another language.
- Clearly distinguish model output from interpretation.
- Never describe a weak or uncertain signal as certain.
- If data needed for an answer is absent, say that it is unavailable.
- Never promise profit, certainty, or guaranteed market direction.
- LONG_BIAS and SHORT_BIAS are analytical biases, not guaranteed outcomes.
- NO_TRADE means the evidence is insufficient for a directional action.
- Keep explanations concise unless the user asks for detail.
"""


def _normalized_history(
    history: Iterable[Mapping[str, str]] | None,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for item in history or ():
        role = str(item.get("role", "")).strip().lower()
        content = str(item.get("content", "")).strip()
        if role not in {"user", "assistant"}:
            raise ValueError("history roles must be user or assistant")
        if not content:
            continue
        rows.append({"role": role, "content": content})
    return rows


def build_messages(
    user_message: str,
    *,
    analysis: AnalysisSnapshot | None = None,
    history: Iterable[Mapping[str, str]] | None = None,
) -> list[dict[str, str]]:
    """Build a model-agnostic chat message list.

    The structured forecast is injected into the system message, not rewritten into a
    natural-language pseudo-analysis before generation. That gives downstream
    runtimes one canonical numeric source and makes grounding checks deterministic.
    """

    message = str(user_message or "").strip()
    if not message:
        raise ValueError("user_message is required")

    system = SYSTEM_PROMPT
    if analysis is not None:
        canonical = json.dumps(
            analysis.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        system += (
            "\nCANONICAL_ANALYSIS_JSON:\n"
            + canonical
            + "\nUse only these values for forecast-specific numerical claims."
        )

    messages = [{"role": "system", "content": system}]
    messages.extend(_normalized_history(history))
    messages.append({"role": "user", "content": message})
    return messages
