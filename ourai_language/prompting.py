from __future__ import annotations

import json
import re
from typing import Iterable, Mapping

from .contracts import AnalysisSnapshot


GENERAL_CHAT = "GENERAL_CHAT"
ANALYSIS_QA = "ANALYSIS_QA"

GENERAL_SYSTEM_PROMPT = """You are OurAI, the conversational assistant inside OrionTrader.

Answer the user's actual question naturally. Do not force an unrelated stock analysis into
ordinary conversation. Reply in the language used by the user unless they ask for another
language. If the user writes Slovak, use natural grammatical Slovak and avoid unnecessary
mixing with Czech or English. Standard finance terms may remain in English when useful.

Do not claim that you have live market data, a current forecast, news, or account information
unless that information is explicitly supplied in the conversation. Never promise profit,
certainty, or guaranteed market direction.
"""

ANALYSIS_SYSTEM_PROMPT = """You are OurAI, the conversational market analyst inside OrionTrader.

Your task is to explain the supplied canonical market-analysis snapshot and answer questions
about it. The snapshot is the only source of truth for forecast-specific facts.

Rules:
- Reply in the language used by the user unless they ask for another language.
- If the user writes Slovak, use natural grammatical Slovak and avoid Czech word forms.
- Never invent forecast numbers, prices, probabilities, expected moves, confidence values,
  company identity, news, catalysts, or market-state facts.
- Use the exact ticker and, when supplied, the exact company_name from the snapshot.
- If company_name is not supplied, refer to the instrument by ticker only. Never infer a name.
- Clearly distinguish model output from your interpretation of that output.
- Never describe a weak or uncertain signal as certain.
- Do not repeat the same reason or risk merely to make an answer longer.
- If data needed for an answer is absent, say that it is unavailable.
- LONG_BIAS and SHORT_BIAS are analytical biases, not guaranteed outcomes or trade orders.
- NO_TRADE means the evidence is insufficient for a directional action.
- Keep explanations concise unless the user asks for detail.
"""

_SOCIAL_MARKERS = (
    "ako sa máš",
    "ako sa mas",
    "how are you",
    "ahoj",
    "čau",
    "cau",
    "hello",
    "hi ",
    "dobrý deň",
    "dobry den",
    "ďakujem",
    "dakujem",
    "thanks",
    "thank you",
    "povedz vtip",
    "tell me a joke",
)

_ANALYSIS_MARKERS = (
    "analýz",
    "analyz",
    "predik",
    "forecast",
    "bullish",
    "bearish",
    "long_bias",
    "short_bias",
    "no_trade",
    "p(up",
    "p(down",
    "pravdepodob",
    "confidence",
    "uncertainty",
    "istot",
    "neistot",
    "rizik",
    "expected move",
    "očakávan",
    "ocakavan",
    "momentum",
    "relative strength",
    "relatívn",
    "relativn",
    "volume",
    "objem",
    "volatil",
    "katalyz",
    "news",
    "správ",
    "sprav",
    "signál",
    "signal",
    "trend",
    "akci",
    "stock",
    "ticker",
    "cena",
    "price",
    "qqq",
    "spy",
)


def _normalized_history(
    history: Iterable[Mapping[str, str]] | None,
    *,
    mode: str | None = None,
    analysis_ticker: str | None = None,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for item in history or ():
        role = str(item.get("role", "")).strip().lower()
        content = str(item.get("content", "")).strip()
        item_mode = str(item.get("mode", "")).strip().upper()
        if role not in {"user", "assistant"}:
            raise ValueError("history roles must be user or assistant")
        if not content:
            continue
        if mode is not None and item_mode and item_mode != mode:
            continue
        # When a recorded analysis turn is tagged with the snapshot it used,
        # never leak its answer into a different instrument's active snapshot.
        # Untagged legacy history remains backward compatible.
        if mode == ANALYSIS_QA and analysis_ticker is not None:
            tagged_ticker = str(item.get("analysis_ticker", "")).strip().upper()
            if tagged_ticker and tagged_ticker != analysis_ticker.upper():
                continue
        rows.append({"role": role, "content": content})
    return rows


def _is_social_message(message: str) -> bool:
    text = message.casefold().strip()
    return any(marker in text for marker in _SOCIAL_MARKERS)


def _has_analysis_signal(message: str, analysis: AnalysisSnapshot | None) -> bool:
    text = message.casefold()
    if analysis is not None:
        ticker = analysis.ticker.casefold()
        if re.search(rf"(?<!\w){re.escape(ticker)}(?!\w)", text):
            return True
        if analysis.company_name and analysis.company_name.casefold() in text:
            return True
    return any(marker in text for marker in _ANALYSIS_MARKERS)


def explicit_snapshot_ticker_mismatch(
    message: str, analysis: AnalysisSnapshot | None,
) -> str | None:
    """Detect a *clearly named* other ticker in a snapshot-specific question.

    This intentionally does not guess tickers from ordinary capitalized words,
    company names or free text. Unrecognized wording stays on the existing
    route; the guard prevents a known mismatch, not universal entity linking.
    """
    if analysis is None or not _has_analysis_signal(message, analysis):
        return None
    text = str(message or "")
    patterns = (
        r"(?<!\w)(?i:pre|pri|o|for|of|on|ticker|symbol|stock|akciu|akcie|akcii)\s+\$?([A-Z]{2,5})(?!\w)",
        r"(?<!\w)(?i:je|is)\s+([A-Z]{2,5})\s+(?i:bullish|bearish)\b",
        r"(?<!\w)([A-Z]{2,5})\s+(?i:bullish|bearish|forecast|predikcia|analýza|analyza)\b",
    )
    # Common metrics and market acronyms are not evidence of a stock identity.
    non_tickers = {"MAE", "RSI", "USD", "EUR", "UP", "DOWN", "ETF", "NYSE", "API", "SIP", "IEX"}
    for pattern in patterns:
        for match in re.finditer(pattern, text):
            ticker = match.group(1)
            if ticker != analysis.ticker and ticker not in non_tickers:
                return ticker
    return None


def _is_short_referential_followup(message: str) -> bool:
    """Only elliptical turns inherit the preceding snapshot analysis mode.

    Length alone is not evidence of continuity: 'A čo Python?' is a new
    subject, while 'Prečo?' or 'A čo to znamená?' refers to the prior answer.
    """
    text = message.casefold().strip(" \t\n?!.,")
    return text in {
        "prečo", "preco", "a prečo", "a preco",
        "a ako", "ako to", "a ako to",
        "vysvetli to", "vysvetli to jednoducho",
        "a čo to znamená", "a co to znamena",
        "čo to znamená", "co to znamena",
        "a čo ďalej", "a co dalej",
        "a potom", "čo tým myslíš", "co tym myslis",
        "a čo z toho vyplýva", "a co z toho vyplyva",
        "čo z toho vyplýva", "co z toho vyplyva",
        "vieš mi to vysvetliť jednoduchšie", "vies mi to vysvetlit jednoduchsie",
    }


def select_conversation_mode(
    user_message: str,
    *,
    analysis: AnalysisSnapshot | None = None,
    history: Iterable[Mapping[str, str]] | None = None,
) -> str:
    """Deterministically separate ordinary chat from snapshot-specific analysis Q&A.

    A small local language model should not be responsible for deciding whether the
    current Orion analysis is relevant. Explicit social/general turns are kept away
    from the snapshot. Very short follow-ups may inherit an immediately preceding
    analysis question, unless they are themselves clearly social.
    """

    message = str(user_message or "").strip()
    if not message:
        raise ValueError("user_message is required")
    if analysis is None:
        return GENERAL_CHAT
    # A greeting or thanks can precede a substantive request. The initial
    # courtesy phrase alone must not suppress snapshot-specific questions.
    courtesy_prefix = re.match(
        r"(?i)^(?:ahoj|čau|cau|hello|dobrý deň|dobry den|ďakujem|dakujem|thanks|thank you)"
        r"(?:[,!:. ]+(?:ale|a)\\b)?[,!:. ]+",
        message,
    )
    effective_message = message[courtesy_prefix.end():] if courtesy_prefix else message
    if _is_social_message(message) and not (
        courtesy_prefix and _has_analysis_signal(effective_message, analysis)
    ):
        return GENERAL_CHAT
    # Never route an explicitly different ticker to the active snapshot.
    if explicit_snapshot_ticker_mismatch(message, analysis) is not None:
        return GENERAL_CHAT
    if _has_analysis_signal(message, analysis):
        return ANALYSIS_QA

    tokens = message.split()
    if len(tokens) <= 6 and _is_short_referential_followup(message):
        for item in reversed(list(history or ())):
            if str(item.get("role", "")).strip().lower() != "user":
                continue
            # When a prior turn carries a recorded routing mode, it is more
            # authoritative than broad vocabulary such as "cena" or "trend".
            # An explicitly general turn must break analytical inheritance.
            prior_mode = str(item.get("mode", "")).strip().upper()
            if prior_mode == GENERAL_CHAT:
                return GENERAL_CHAT
            if prior_mode == ANALYSIS_QA:
                # A new active ticker must not inherit an elliptical question
                # from a different, explicitly tagged analysis. Otherwise a
                # bare "Prečo?" can silently describe the wrong instrument.
                previous_ticker = str(item.get("analysis_ticker", "")).strip().upper()
                if previous_ticker and previous_ticker != analysis.ticker:
                    return GENERAL_CHAT
                return ANALYSIS_QA
            # Backward-compatible fallback for histories without mode metadata.
            previous = str(item.get("content", "")).strip()
            if _is_social_message(previous):
                return GENERAL_CHAT
            if _has_analysis_signal(previous, analysis):
                return ANALYSIS_QA
            break

    return GENERAL_CHAT


def _analysis_fact_sheet(analysis: AnalysisSnapshot) -> str:
    company_name = analysis.company_name or "NOT_SUPPLIED_DO_NOT_INFER"
    lines = [
        "CANONICAL_ANALYSIS_FACTS:",
        f"- ticker: {analysis.ticker}",
        f"- company_name: {company_name}",
        f"- horizon: {analysis.horizon_minutes} minutes",
        f"- P(UP): {analysis.p_up * 100.0:.1f}%",
        f"- P(DOWN): {analysis.p_down * 100.0:.1f}%",
        f"- uncertainty: {analysis.uncertainty}",
        f"- decision: {analysis.decision}",
    ]
    if analysis.expected_move_pct is not None:
        lines.append(f"- expected move magnitude: {analysis.expected_move_pct:.2f}%")
    if analysis.confidence is not None:
        lines.append(f"- confidence: {analysis.confidence * 100.0:.1f}%")
    for name in (
        "market_regime",
        "relative_strength",
        "volume_state",
        "news_context",
        "model_version",
        "as_of",
    ):
        value = getattr(analysis, name)
        if value is not None:
            lines.append(f"- {name}: {value}")

    if analysis.reasons_for:
        lines.append("- supporting reasons: " + "; ".join(analysis.reasons_for))

    uncertainty_phrase = f"{analysis.uncertainty} uncertainty".casefold()
    distinct_risks = [
        reason
        for reason in analysis.reasons_against
        if reason.casefold() != uncertainty_phrase
    ]
    if distinct_risks:
        lines.append("- opposing reasons / risks: " + "; ".join(distinct_risks))
    return "\n".join(lines)


def build_messages(
    user_message: str,
    *,
    analysis: AnalysisSnapshot | None = None,
    history: Iterable[Mapping[str, str]] | None = None,
    mode: str | None = None,
) -> list[dict[str, str]]:
    """Build a model-agnostic message list with deterministic intent separation."""

    message = str(user_message or "").strip()
    if not message:
        raise ValueError("user_message is required")

    selected_mode = mode or select_conversation_mode(
        message,
        analysis=analysis,
        history=history,
    )
    if selected_mode not in {GENERAL_CHAT, ANALYSIS_QA}:
        raise ValueError("mode must be GENERAL_CHAT or ANALYSIS_QA")
    if selected_mode == ANALYSIS_QA and analysis is None:
        selected_mode = GENERAL_CHAT
    # Enforce the same boundary even if a caller explicitly forces ANALYSIS_QA.
    # Runtime-specific unavailable replies still take precedence before prompting.
    if analysis is not None and explicit_snapshot_ticker_mismatch(message, analysis):
        selected_mode = GENERAL_CHAT

    if selected_mode == ANALYSIS_QA:
        canonical = json.dumps(
            analysis.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        system = (
            ANALYSIS_SYSTEM_PROMPT
            + "\n"
            + _analysis_fact_sheet(analysis)
            + "\n\nCANONICAL_ANALYSIS_JSON:\n"
            + canonical
            + "\nThe fact sheet and JSON describe the same canonical snapshot."
        )
    else:
        system = GENERAL_SYSTEM_PROMPT

    messages = [{"role": "system", "content": system}]
    messages.extend(_normalized_history(
        history, mode=selected_mode,
        analysis_ticker=analysis.ticker if selected_mode == ANALYSIS_QA and analysis is not None else None,
    ))
    messages.append({"role": "user", "content": message})
    return messages
