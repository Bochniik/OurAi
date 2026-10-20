"""Constrained read-only tool choice across historical reports and forward journal.

Qwen can only suggest a tool/action enum. Source selection is independently
anchored to the user's own words and routed through existing verified readers.
No model-authored paths, tickers, numbers, timestamps or arbitrary tool calls.
"""
from __future__ import annotations

import json
import re
import unicodedata

from .semantic_router import canonical_historical_question
from trading.forward_analytics import FROZEN_SYMBOLS


SYSTEM_PROMPT = """You route a question to one of two READ-ONLY verified OurAI tools.
Return ONLY JSON with exactly four string keys:
{"tool":"none|historical_v3|forward_v2",
 "action":"none|overview|validation|test|compare|symbols|causes|simple|summary|performance|latest",
 "split":"both|validation|test","baseline":"both|linear|null"}.
Use historical_v3 ONLY for OurAI retrospective validation, lockbox and baseline
results. Use forward_v2 ONLY for actual recorded/pending/resolved magnitude
predictions and shadow/paper forward observations. Historical backtests are
NOT live/forward evidence. Choose none for generic educational questions,
stock prices, news, trading orders, hypothetical situations or uncertainty.
Historical actions: overview,validation,test,compare,symbols,causes,simple.
Forward actions: summary,performance,latest.
Set split=both and baseline=both for forward_v2 and none.
Never output file paths, specific symbols, financial numbers or explanations.
Example: "Ako sa ti darí v shadow režime?" => forward_v2,performance,both,both.
Example: "Porovnaj historickú MAE s lineárnym modelom" =>
historical_v3,compare,both,linear.
Example: "Čo je paper trading?" => none,none,both,both.
"""


_HISTORICAL_ACTIONS = frozenset((
    "overview", "validation", "test", "compare", "symbols", "causes", "simple",
))
_FORWARD_ACTIONS = frozenset(("summary", "performance", "latest"))
_SPLITS = frozenset(("both", "validation", "test"))
_BASELINES = frozenset(("both", "linear", "null"))
_FORWARD_ANCHORS = (
    "shadow", "paper trading", "paper-trading", "forward", "zive sledovanie",
    "ziveho sledovania", "real-time", "v realnom case", "live-forward",
)
_FORWARD_SUBJECTS = (
    "odhad", "predik", "model", "ourai", "zaznam", "dennik",
    "vysled", "chyb", "mae", "uspes", "presn", "daril", "stav",
    "pocet", "kolko", "zatial", "dotera", "report", "hodnoten",
    "dari", "mas ", "have ",
)
_CURRENT_TRADE = (
    "kupi", "kup ", "predaj", "nakup", "kupit", "predat", "zobchoduj",
    "order", "obchodny prikaz", "garantuj", "zaruc",
)
_DEFINITIONS = ("co je ", "co znamena ", "what is ", "define ", "vysvetli pojem ")


def _plain(text: str) -> str:
    folded = unicodedata.normalize("NFKD", text.casefold())
    return "".join(character for character in folded if not unicodedata.combining(character))


def _user_forward_evidence(question: str) -> bool:
    text = _plain(question).strip(" .?!„“\"")
    if not text or len(text) > 500 or text.startswith(_DEFINITIONS):
        return False
    if not any(anchor in text for anchor in _FORWARD_ANCHORS):
        return False
    if not any(marker in text for marker in _FORWARD_SUBJECTS):
        return False
    if any(marker in text for marker in _CURRENT_TRADE):
        return False
    # A question crossing two incompatible evaluation sources needs an
    # explicit clarification; never silently relabel retrospective evidence.
    if any(marker in text for marker in (
        "lockbox", "validac", "backtest", "retrospekt", "historick",
    )):
        return False
    return True


def _mentioned_symbol(question: str) -> str | None:
    return next(
        (
            symbol for symbol in FROZEN_SYMBOLS
            if re.search(r"(?<!\w)" + re.escape(symbol) + r"(?!\w)", question, re.I)
        ),
        None,
    )


def select_verified_tool(
    user_question: str,
    model_response: str,
    *,
    historical_available: bool,
    forward_available: bool,
) -> tuple[str, str] | None:
    """Return (source, *internally authored* query), or no safe tool choice.

    The called report reader remains responsible for validation of every
    numerical statement and for the missing/corrupt input fail-closed path.
    """
    if not isinstance(model_response, str) or len(model_response) > 500:
        return None
    try:
        payload = json.loads(model_response)
    except (ValueError, TypeError):
        return None
    if not isinstance(payload, dict) or set(payload) != {
        "tool", "action", "split", "baseline",
    }:
        return None
    tool, action, split, baseline = (
        payload["tool"], payload["action"], payload["split"], payload["baseline"],
    )
    if not all(isinstance(v, str) for v in (tool, action, split, baseline)):
        return None
    if split not in _SPLITS or baseline not in _BASELINES:
        return None
    if tool == "historical_v3":
        if not historical_available or action not in _HISTORICAL_ACTIONS:
            return None
        original_schema = json.dumps({
            "intent": action, "split": split, "baseline": baseline,
        })
        canonical = canonical_historical_question(user_question, original_schema)
        return ("historical_v3", canonical) if canonical is not None else None
    if tool == "forward_v2":
        if not forward_available or action not in _FORWARD_ACTIONS:
            return None
        if split != "both" or baseline != "both" or not _user_forward_evidence(user_question):
            return None
        ticker = _mentioned_symbol(user_question)
        if action == "performance":
            question = "Aká je presnosť predikcií "
            question += f"{ticker} " if ticker else ""
            return "forward_v2", question + "vo forward denníku?"
        if action == "latest":
            question = "Aká je posledná predikcia "
            question += f"{ticker} " if ticker else ""
            return "forward_v2", question + "vo forward denníku?"
        question = "Zhrň predikcie "
        question += f"{ticker} " if ticker else ""
        return "forward_v2", question + "vo forward denníku."
    return None
