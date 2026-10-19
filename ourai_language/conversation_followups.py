"""Safe contextual follow-up routing for verified retrospective OurAI results.

Only a recent, explicitly tagged answer from the historical report reader can
authorize historical follow-ups. We never turn a prior generated model message
into numerical evidence or infer a new source from conversational prose.
"""
from __future__ import annotations

import unicodedata
from typing import Iterable, Mapping

from .historical_answers import historical_question_kind


def _norm(value: str) -> str:
    folded = unicodedata.normalize("NFKD", value.casefold())
    return "".join(character for character in folded if not unicodedata.combining(character))


def _context(history: Iterable[Mapping[str, str]] | None) -> str | None:
    rows = list(history or ())
    if len(rows) < 2:
        return None
    last_user, last_assistant = rows[-2:]
    if (
        last_user.get("role") != "user"
        or last_assistant.get("role") != "assistant"
        or last_assistant.get("source") != "historical_v3_reports"
    ):
        return None
    question = str(last_user.get("content", ""))
    if historical_question_kind(question) is None:
        return None
    return question


def historical_followup_question(
    message: str,
    history: Iterable[Mapping[str, str]] | None,
) -> str | None:
    """Rewrite conservative short follow-ups, preserving source and split.

    Ambiguous long or unrelated turns stay with the normal language model.
    The returned question must still go through the report reader and all
    of its provenance and numeric checks before any result is displayed.
    """
    prior = _context(history)
    if prior is None:
        return None
    raw = str(message or "").strip()
    text = _norm(raw).strip(" .?!„“\"")
    if not text or len(raw) > 110 or len(text.split()) > 13:
        return None
    if any(term in text for term in (
        "forward", "posledn", "najnov", "aktualn", "journal", "dennik",
        "teraz", "zajtra", "dnes", "live", "obchod", "nakup", "predaj",
    )):
        return None

    split = "lockbox teste" if any(term in text for term in (
        "lockbox", "testovac", "testovaci",
    )) else "validácii" if "validac" in text else None
    prior_kind = historical_question_kind(prior)
    if split is None:
        if prior_kind in ("validation", "compare_validation"):
            split = "validácii"
        elif prior_kind in ("test", "compare_test"):
            split = "lockbox teste"

    if any(term in text for term in ("nulov", "null")):
        target = "nulovým baseline"
    elif any(term in text for term in ("linearn", "linear")):
        target = "lineárnym modelom"
    elif "nulov" in _norm(prior) or "null" in _norm(prior):
        target = "nulovým baseline"
    elif "linearn" in _norm(prior) or "linear" in _norm(prior):
        target = "lineárnym modelom"
    else:
        target = "lineárnym a nulovým baseline"

    why = text.startswith(("a preco", "preco", "a z akeho dovodu", "z akeho dovodu"))
    if why:
        return "Prečo sa líšia historické výsledky modelu v3?"
    simple = text.startswith((
        "vysvetli", "a vysvetli", "jednoducho", "zhrn",
        "a co z toho", "co z toho", "co to znamena pre model",
    ))
    if simple:
        if split == "lockbox teste":
            return "Zhrň historické výsledky lockbox testu v3 jednoducho."
        if split == "validácii":
            return "Zhrň výsledky historickej validácie v3 jednoducho."
        return "Zhrň historické výsledky modelu v3 jednoducho."
    if text.startswith(("a pri ktorej", "a pri ktorych", "a ktora akcia", "a ktore akcie")):
        return "Pri ktorých akciách robí historický model v3 najväčšie chyby?"

    comparative = prior_kind is not None and prior_kind.startswith("compare")
    comparable_prompt = any(term in text for term in (
        "o kolko", "oproti", "v porovnani", "porovnaj", "kolko leps",
        "kolko hors", "aky rozdiel", "rozdiel oproti", "prekon",
    ))
    if comparative or comparable_prompt or (
        text.startswith(("a co ", "a oproti ", "a ten ", "a s ")) and
        any(term in text for term in ("linearn", "linear", "nulov", "null", "baseline"))
    ):
        scope = ("na historickom lockbox teste" if split == "lockbox teste" else "v historickej validácii") if split else "v historickej validácii aj na historickom lockbox teste"
        return f"Porovnaj výsledky modelu v3 {scope} s {target}."

    change_split = (
        any(term in text for term in ("lockbox", "validac", "testovac", "testovaci"))
        and text.startswith(("a ", "co ", "ako ", "a na", "a co"))
    )
    if change_split:
        scope = "na historickom lockbox teste" if split == "lockbox teste" else "v historickej validácii"
        return f"Aká bola MAE modelu v3 {scope}?"
    return None
