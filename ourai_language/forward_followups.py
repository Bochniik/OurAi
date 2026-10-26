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


def _mentions_retrospective_source(question: str) -> bool:
    """Never inherit a forward ticker from retrospective or lockbox context.

    The assistant source tag alone is not sufficient if the preceding user
    request or stored evidence question names a conflicting evidence source.
    """
    folded = unicodedata.normalize("NFKD", question.casefold())
    plain = "".join(char for char in folded if not unicodedata.combining(char))
    return re.search(
        r"\b(?:lockbox\w*|historick\w*|backtest\w*|retrospekt\w*|validac\w*|validation\w*)\b",
        plain,
    ) is not None


def _tickerless_turn_matches_evidence(user_text: str, evidence: str) -> bool:
    """Allow ticker inheritance only from known bounded, tickerless follow-ups.

    A source-tagged answer to an unrelated or overall user question cannot
    manufacture a per-symbol conversational context. This does not authorize
    a new reader: callers still validate the source and re-read the journal.
    """
    if len(user_text) > 100:
        return False
    folded = unicodedata.normalize("NFKD", user_text.casefold())
    plain = "".join(char for char in folded if not unicodedata.combining(char))
    plain = plain.strip(" .?!")
    kind = forward_question_kind(evidence)
    folded_evidence = unicodedata.normalize("NFKD", evidence.casefold())
    evidence_plain = "".join(
        char for char in folded_evidence if not unicodedata.combining(char)
    )
    latest = kind in ("ticker", "latest") and re.search(
        r"\b(?:posledn\w*|najnov\w*)\b", evidence_plain,
    ) is not None
    if latest and (
        re.fullmatch(
            r"(?:a\s+)?(?:aka\s+(?:je|bola)\s+|co\s+)?"
            r"(?:jej\s+)?posledna\s+predikcia",
            plain,
        )
        or plain in (
            "co tato predikcia znamena",
            "co tato predikcia vlastne znamena",
            "a co tato predikcia znamena",
            "a co tato predikcia vlastne znamena",
            "vysvetli mi tuto predikciu",
            "a vysvetli mi tuto predikciu",
            "co znamena ta posledna predikcia",
            "a co znamena ta posledna predikcia",
        )
    ):
        return True
    if kind == "performance" and (
        re.fullmatch(
            r"(?:a\s+)?(?:(?:aka\s+(?:je|bola)\s+)"
            r"(?:jej\s+|ich\s+)?presnost(?:\s+predikcii)?"
            r"|(?:co\s+)(?:jej\s+|ich\s+)?presnost"
            r"(?:\s+predikcii)?|presnost\s+predikcii)",
            plain,
        )
        or plain in (
            "co tieto vysledky znamenaju",
            "a co tieto vysledky znamenaju",
            "co znamenaju tieto vysledky",
            "a co znamenaju tieto vysledky",
            "vysvetli mi tieto vysledky",
            "a vysvetli mi tieto vysledky",
        )
    ):
        return True
    return False



def _named_turn_has_forward_intent(user_text: str) -> bool:
    """Do not inherit a ticker from a named but unrelated previous request.

    A verified-looking evidence tag cannot turn a stock price, news, trade,
    greeting or educational question into forward evidence. Preserve the
    already-supported short ticker-switch turns (which omit metric wording).
    """
    if forward_question_kind(user_text) is not None:
        return True
    return re.fullmatch(
        r"(?:a\s+)?(?:[čc]o\s+(?:pre\s+)?|pre\s+)"
        r"[A-Za-z]{2,30}\s*[?.!]?",
        user_text.strip(),
        re.IGNORECASE,
    ) is not None


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
    # The source tag alone is insufficient: a malformed or mismatched
    # evidence_question must not authorize a switch to a different ticker.
    # An overall verified report (no ticker in either turn) remains eligible.
    if _mentions_unsupported_forward_symbol(prior) or _mentions_retrospective_source(prior):
        return None
    prior_symbols = _mentioned_symbols(prior)
    prior_user_text = previous_user.get("content")
    if (
        len(prior_symbols) > 1
        or not isinstance(prior_user_text, str)
        or _mentions_unsupported_forward_symbol(prior_user_text)
        or _mentions_retrospective_source(prior_user_text)
    ):
        return None
    user_symbols = _mentioned_symbols(prior_user_text)
    if len(user_symbols) > 1 or (user_symbols and user_symbols != prior_symbols):
        return None
    if prior_symbols and not user_symbols and not _tickerless_turn_matches_evidence(
        prior_user_text, prior,
    ):
        return None
    if user_symbols and not _named_turn_has_forward_intent(prior_user_text):
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


def _verified_forward_ticker(
    history: Iterable[Mapping[str, str]] | None,
) -> str | None:
    """Trust only the immediately preceding, source-tagged forward answer."""
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
    if _mentions_unsupported_forward_symbol(prior) or _mentions_retrospective_source(prior):
        return None
    symbols = _mentioned_symbols(prior)
    if len(symbols) != 1:
        return None
    # Never grant a different stock than the original user explicitly named.
    prior_user_text = previous_user.get("content")
    if (
        not isinstance(prior_user_text, str)
        or _mentions_unsupported_forward_symbol(prior_user_text)
        or _mentions_retrospective_source(prior_user_text)
    ):
        return None
    user_symbols = _mentioned_symbols(prior_user_text)
    if len(user_symbols) > 1 or (user_symbols and user_symbols != symbols):
        return None
    if not user_symbols and not _tickerless_turn_matches_evidence(
        prior_user_text, prior,
    ):
        return None
    if user_symbols and not _named_turn_has_forward_intent(prior_user_text):
        return None
    return symbols[0]


def forward_performance_followup_question(
    message: str, history: Iterable[Mapping[str, str]] | None,
) -> str | None:
    """Resolve a narrowly phrased accuracy follow-up to verified forward MAE.

    This requests aggregate per-ticker forward magnitude errors, never
    accuracy of a single pending prediction or directional hit rate.
    """
    raw = str(message or "").strip()
    if len(raw) > 80:
        return None
    folded = unicodedata.normalize("NFKD", raw.casefold())
    plain = "".join(char for char in folded if not unicodedata.combining(char))
    if re.fullmatch(
        r"(?:a\s+)?(?:(?:aka\s+(?:je|bola)\s+)(?:jej\s+|ich\s+)?presnost"
        r"(?:\s+predikcii)?|(?:co\s+)(?:jej\s+|ich\s+)?presnost"
        r"(?:\s+predikcii)?|presnost\s+predikcii)\s*[?.!]?",
        plain,
    ) is None:
        return None
    ticker = _verified_forward_ticker(history)
    if ticker is None:
        return None
    return f"Aká je presnosť predikcií {ticker} vo forward denníku?"


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
        r"(?:a\s+)?(?:aka\s+(?:je|bola)\s+|co\s+)?(?:jej\s+)?posledna\s+predikcia\s*[?.!]?",
        plain,
    ) is None:
        return None

    ticker = _verified_forward_ticker(history)
    if ticker is None:
        return None
    return f"Aká je posledná predikcia {ticker} vo forward denníku?"

def forward_results_explanation_followup_question(
    message: str, history: Iterable[Mapping[str, str]] | None,
) -> str | None:
    """Re-read single-ticker forward performance for a bounded explanation.

    The explanation is deterministic, never causal or a directional-accuracy
    claim. No numbers are inherited from the prior assistant response.
    """
    raw = str(message or "").strip()
    if len(raw) > 90:
        return None
    folded = unicodedata.normalize("NFKD", raw.casefold())
    plain = "".join(char for char in folded if not unicodedata.combining(char))
    plain = plain.strip(" .?!")
    if plain not in (
        "co tieto vysledky znamenaju",
        "a co tieto vysledky znamenaju",
        "co znamenaju tieto vysledky",
        "a co znamenaju tieto vysledky",
        "vysvetli mi tieto vysledky",
        "a vysvetli mi tieto vysledky",
    ):
        return None
    rows = list(history or ())
    ticker = _verified_forward_ticker(rows)
    if ticker is None:
        return None
    # A latest-record estimate is not an observed error metric.
    if forward_question_kind(rows[-1]["evidence_question"]) != "performance":
        return None
    return f"Aká je presnosť predikcií {ticker} vo forward denníku?"



def forward_latest_explanation_followup_question(
    message: str, history: Iterable[Mapping[str, str]] | None,
) -> str | None:
    """Re-read a verified single-ticker latest record for a bounded explanation.

    A performance report, historical result, model-generated reply or generic
    ticker discussion cannot establish that a latest prediction was shown.
    """
    raw = str(message or "").strip()
    if len(raw) > 100:
        return None
    folded = unicodedata.normalize("NFKD", raw.casefold())
    plain = "".join(char for char in folded if not unicodedata.combining(char))
    plain = plain.strip(" .?!")
    if plain not in (
        "co tato predikcia znamena",
        "co tato predikcia vlastne znamena",
        "a co tato predikcia znamena",
        "a co tato predikcia vlastne znamena",
        "vysvetli mi tuto predikciu",
        "a vysvetli mi tuto predikciu",
        "co znamena ta posledna predikcia",
        "a co znamena ta posledna predikcia",
    ):
        return None
    rows = list(history or ())
    ticker = _verified_forward_ticker(rows)
    if ticker is None:
        return None
    prior = rows[-1]["evidence_question"]
    if forward_question_kind(prior) not in ("ticker", "latest"):
        return None
    folded_prior = unicodedata.normalize("NFKD", prior.casefold())
    plain_prior = "".join(
        char for char in folded_prior if not unicodedata.combining(char)
    )
    if re.search(r"\b(?:posledn\w*|najnov\w*)\b", plain_prior) is None:
        return None
    return f"Aká je posledná predikcia {ticker} vo forward denníku?"
