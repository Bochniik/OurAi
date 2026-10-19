"""Constrained semantic classification for read-only v3 retrospective reports.

A language model may propose a *tool category*, never a financial number or a
free-form tool argument. This module validates the proposal and maps it to a
fixed question accepted by the deterministic report reader. General questions,
forward predictions, and current/live data are explicitly excluded.
"""
from __future__ import annotations

import json
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
    "real time", "prave teraz", "zajtra", "v realnom case",
)


def _plain(text: str) -> str:
    folded = unicodedata.normalize("NFKD", text.casefold())
    return "".join(c for c in folded if not unicodedata.combining(c))


def canonical_historical_question(
    original_question: str,
    model_response: str,
) -> str | None:
    """Accept a validated classification, never a model-authored source claim.

    A temporal guard prevents an accidental historical answer to a question
    about recent/forward/live activity, even when the model misclassifies it.
    """
    user_text = _plain(str(original_question or "").strip())
    if not user_text or any(token in user_text for token in _CURRENT):
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
