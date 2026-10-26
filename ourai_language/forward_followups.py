"""Conservative ticker-switch follow-ups after verified forward evidence.

Only the immediately preceding source-tagged, canonically recorded forward
answer can provide the metric. No assistant prose or generated numeric value
is reused: the resulting question is re-read from the verified journal.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Iterable, Mapping

from .forward_answers import forward_question_kind
from .verified_tool_router import (
    _mentioned_symbols, _mentions_unsupported_forward_symbol,
)


def forward_ticker_followup_question(
    message: str, history: Iterable[Mapping[str, str]] | None,
) -> str | None:
    """Switch one verified forward question to one explicitly named ticker.

    Do not infer a ticker, source or metric from model prose. Only short,
    standalone Slovak follow-ups naming a supported ticker/curated alias are
    eligible. All other turns remain ordinary conversation.
    """
    rows = list(history or ())
    if len(rows) < 2:
        return None
    previous_user, previous_assistant = rows[-2:]
    if (
        previous_user.get("role") != "user"
        or previous_assistant.get("role") != "assistant"
        or previous_assistant.get("source") != "verified_forward_journal"
    ):
        return None
    prior = previous_assistant.get("evidence_question")
    if not isinstance(prior, str) or not prior:
        return None
    prior_kind = forward_question_kind(prior)
    if prior_kind not in ("performance", "latest", "ticker"):
        return None
    raw = str(message or "").strip()
    if len(raw) > 65:
        return None
    match = re.fullmatch(
        r"(?:a\s+)?(?:[čc]o\s+(?:pre\s+)?|pre\s+)([A-Za-z]{2,30})\s*[?.!]?",
        raw, re.IGNORECASE,
    )
    if match is None or _mentions_unsupported_forward_symbol(raw):
        return None
    symbols = _mentioned_symbols(match.group(1))
    if len(symbols) != 1:
        return None
    ticker = symbols[0]
    if prior_kind == "performance":
        return f"Aká je presnosť predikcií {ticker} vo forward denníku?"
    return f"Aká je posledná predikcia {ticker} vo forward denníku?"


def forward_latest_followup_question(
    message: str, history: Iterable[Mapping[str, str]] | None,
) -> str | None:
    """Reuse one verified forward ticker for an explicit latest-record follow-up.

    This is deliberately not a general pronoun resolver. Only the immediately
    preceding source-tagged forward answer may provide the ticker; a new
    canonical question always re-reads the existing verified journal.
    """
    raw = str(message or "").strip()
    if len(raw) > 80:
        return None
    folded = unicodedata.normalize("NFKD", raw.casefold())
    plain = "".join(char for char in folded if not unicodedata.combining(char))
    if re.fullmatch(
        r"(?:a\s+)?(?:aka\s+(?:je|bola)\s+|co\s+)?posledna\s+predikcia\s*[?.!]?",
        plain,
    ) is None:
        return None

    rows = list(history or ())
    if len(rows) < 2:
        return None
    previous_user, previous_assistant = rows[-2:]
    if (
        previous_user.get("role") != "user"
        or previous_assistant.get("role") != "assistant"
        or previous_assistant.get("source") != "verified_forward_journal"
    ):
        return None
    prior = previous_assistant.get("evidence_question")
    if not isinstance(prior, str) or forward_question_kind(prior) is None:
        return None
    if _mentions_unsupported_forward_symbol(prior):
        return None
    symbols = _mentioned_symbols(prior)
    if len(symbols) != 1:
        return None
    # A malformed or mismatched prior user request must not silently grant
    # the follow-up a different stock than the one explicitly requested.
    prior_user_text = previous_user.get("content")
    if not isinstance(prior_user_text, str) or _mentions_unsupported_forward_symbol(prior_user_text):
        return None
    user_symbols = _mentioned_symbols(prior_user_text)
    if len(user_symbols) > 1 or (user_symbols and user_symbols != symbols):
        return None
    return f"Aká je posledná predikcia {symbols[0]} vo forward denníku?"
