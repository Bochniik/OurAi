"""Constrained semantic classification for read-only v3 retrospective reports.

A language model may propose a *tool category*, never a financial number or a
free-form tool argument. This module validates the proposal and maps it to a
fixed question accepted by the deterministic report reader. General questions,
forward predictions, and current/live data are explicitly excluded.
"""
from __future__ import annotations

import json
import re
import unicodedata


SYSTEM_PROMPT = """Classify the user's question for a strictly READ-ONLY historical
15-minute absolute-return-magnitude performance report reader. Reply ONLY a
JSON object, never prose. Schema:
{"intent":"none|overview|validation|test|compare|symbols|causes|simple",
 "split":"both|validation|test", "baseline":"both|linear|null"}.
Choose "none" for arbitrary/general/educational questions, current stock
prices, live/latest/forward predictions, trading, or when unsure. Use
"overview" for historical model accuracy; "test" for previously unseen held-out
lockbox results; "validation" for historical validation; "compare" for
comparisons of the retrospective v3 model and baselines, including natural
phrasing without the word 'compare'; "symbols" for which stocks historically
have the largest/smallest errors; "causes" when asked *why* measured historical
performance differs; "simple" when asked to summarize historical performance.
For compare, set split and baseline from the question when given; otherwise
"both". Do not produce numbers, user-supplied text, explanations or file paths.
Examples:
'How did you do on unseen historical data?' => test
'Ako sa ti darilo na dátach, ktoré si predtým nevidel?' => test
'Explain what a stock is' => none
'How well do you compare with a linear model?' => compare, linear, both
'What is the latest NVDA prediction?' => none
'Why did v3 outperform the baseline historically?' => causes
"""

_ALLOWED = {"none", "overview", "validation", "test", "compare", "symbols", "causes", "simple"}
_SPLITS = {"both", "validation", "test"}
_BASELINES = {"both", "linear", "null"}
_CURRENT = (
    "forward", "journal", "dennik", "latest", "recent", "current", "live",
    "teraz", "dnes", "dnesn", "aktualn", "posledn", "najnov", "real-time",
    "shadow", "paper trading", "paper-trading",
    "real time", "prave teraz", "zajtra", "v realnom case",
)


def _plain(text: str) -> str:
    folded = unicodedata.normalize("NFKD", text.casefold())
    return "".join(c for c in folded if not unicodedata.combining(c))


# An LLM classification is a suggestion, not proof that the user asked for
# retrospective model performance. Require explicit linguistic evidence in the
# *user's* question before permitting a historical report to be opened.
_HISTORICAL_CUES = (
    "histor", "retrospekt", "backtest", "lockbox", "validac", "held-out",
    "unseen", "neviden", "nevidel", "out-of-sample", "testovac",
)
_MODEL_CUES = (
    "ourai", "model", "v3", "baseline", "predik", "forecast",
    "tvoj", "tvoje", "tvojich", "nasich", "our ", "your ",
    "sa ti", "si na tom", "did you",
)
_PERFORMANCE_CUES = (
    "vysled", "result", "performance", "presn", "accuracy", "mae",
    "chyb", "error", "baseline", "prekon", "outperform", "porovn",
    "compare", "leps", "better", "hors", "worse", "obstal", "daril",
    "did you do", "uspesn", "vyhodnot", "validac", "lockbox",
    "testovac", "struggle", "reliable", "spolahliv",
)
_DEFINITION_STARTS = (
    "co je ", "co znamena ", "what is ", "define ", "vysvetli pojem ",
)


def _is_retrospective_performance_request(text: str) -> bool:
    historical = any(cue in text for cue in _HISTORICAL_CUES)
    model = any(cue in text for cue in _MODEL_CUES)
    performance = any(cue in text for cue in _PERFORMANCE_CUES)
    if text.startswith(_DEFINITION_STARTS) and not (
        historical and model and performance
    ):
        return False
    return performance and (historical or model)


def _mentions_current_or_forward(text: str) -> bool:
    # Match at the start of a word, not inside an unrelated word (e.g.
    # "deliver" contains "live"). Prefix matching handles Slovak inflections.
    return any(re.search(r"(?<!\w)" + re.escape(token), text) for token in _CURRENT)


def explicit_heldout_performance_question(original_question: str) -> str | None:
    """Fail-closed fallback for unmistakable *self-performance on unseen data* intent.

    A small classifier can return `none` for colloquial Slovak even when the
    user is asking OurAI about its held-out results. Never infer unseen
    performance from a generated explanation: this only selects the local
    verified lockbox report reader, which still checks all report provenance.
    """
    text = _plain(str(original_question or "").strip())
    if not text or _mentions_current_or_forward(text):
        return None
    if text.startswith(_DEFINITION_STARTS):
        return None
    if not re.search(r"(?<!\w)(?:dat\w*|dataset\w*|vzork\w*|samples?\b)", text):
        return None
    if not any(term in text for term in (
        "nevidel", "neviden", "unseen", "held-out", "out-of-sample",
    )):
        return None
    if not re.search(r"(?<!\w)(?:ti|si|tvoj\w*|ourai|v3|modelu?\b|you|your)\b", text):
        return None
    if not any(term in text for term in (
        "daril", "obstal", "presn", "uspes", "vysled",
        "perform", "accur", "error", "chyb",
    )):
        return None
    return "Aký bol historický lockbox test v3?"


def canonical_historical_question(
    original_question: str,
    model_response: str,
) -> str | None:
    """Accept a validated classification, never a model-authored source claim.

    A temporal guard prevents an accidental historical answer to a question
    about recent/forward/live activity, even when the model misclassifies it.
    """
    user_text = _plain(str(original_question or "").strip())
    if (
        not user_text
        or _mentions_current_or_forward(user_text)
        or not _is_retrospective_performance_request(user_text)
    ):
        return None
    if not isinstance(model_response, str) or len(model_response) > 500:
        return None
    try:
        payload = json.loads(model_response)
    except (ValueError, TypeError):
        return None
    if not isinstance(payload, dict) or set(payload) != {"intent", "split", "baseline"}:
        return None
    intent = payload["intent"]
    split = payload["split"]
    baseline = payload["baseline"]
    if (
        not isinstance(intent, str) or intent not in _ALLOWED
        or not isinstance(split, str) or split not in _SPLITS
        or not isinstance(baseline, str) or baseline not in _BASELINES
    ):
        return None
    if intent == "none":
        return None
    if intent == "overview":
        return "Ako presný bol historicky model v3?"
    if intent == "validation":
        return "Ako dopadla historická validácia v3?"
    if intent == "test":
        return "Aký bol historický lockbox test v3?"
    if intent == "symbols":
        return "Pri ktorých akciách robí historický model v3 najväčšie chyby?"
    if intent == "causes":
        return "Prečo sa líšia historické výsledky modelu v3?"
    if intent == "simple":
        if split == "validation":
            return "Zhrň výsledky historickej validácie v3 jednoducho."
        if split == "test":
            return "Zhrň historické výsledky lockbox testu v3 jednoducho."
        return "Zhrň historické výsledky modelu v3 jednoducho."

    # Every possible model classification maps to an internally authored query.
    # The source reader alone computes metrics and chooses the correct split.
    baseline_label = {
        "linear": "lineárnym modelom",
        "null": "nulovým baseline",
        "both": "lineárnym aj nulovým baseline",
    }[baseline]
    scope = {
        "both": "",
        "validation": "v historickej validácii ",
        "test": "na historickom lockbox teste ",
    }[split]
    return f"Porovnaj model v3 {scope}s {baseline_label}.".replace("  ", " ")
