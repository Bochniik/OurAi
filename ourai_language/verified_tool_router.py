"""Constrained read-only tool choice across historical reports and forward journal.

Qwen can only suggest a tool/action enum. Source selection is independently
anchored to the user's own words and routed through existing verified readers.
No model-authored paths, tickers, numbers, timestamps or arbitrary tool calls.
"""
from __future__ import annotations

import json
import re
import unicodedata

from .forward_answers import is_forward_educational_request
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
Example: "Môžeš mi vysvetliť, ako funguje shadow model?" => none,none,both,both.
Example: "Prosím, vysvetli mi, čo znamená MAE vo forward denníku." => none,none,both,both.
Example: "Vysvetli mi výsledky vo forward denníku." => forward_v2,performance,both,both.
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
def _plain(text: str) -> str:
    folded = unicodedata.normalize("NFKD", text.casefold())
    return "".join(character for character in folded if not unicodedata.combining(character))


def _user_forward_evidence(question: str) -> bool:
    text = _plain(question).strip(" .?!„“\"")
    if not text or len(text) > 500 or is_forward_educational_request(question):
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


# Curated, unambiguous Slovak company forms for frozen-universe tickers.
# These are user-word aliases, never model-proposed ticker arguments.
_VERIFIED_NVDA_ALIASES = frozenset(("NVIDIA", "NVIDIE", "NVIDII", "NVIDIU", "NVIDIOU"))
_NVDA_ALIAS_PATTERN = r"(?<!\w)(?:nvidia|nvidie|nvidii|nvidiu|nvidiou)(?!\w)"
_VERIFIED_TSLA_ALIASES = frozenset(("TESLA", "TESLY", "TESLE", "TESLU", "TESLOU"))
_TSLA_ALIAS_PATTERN = r"(?<!\w)(?:tesla|tesly|tesle|teslu|teslou)(?!\w)"
_VERIFIED_MSFT_ALIASES = frozenset(("MICROSOFT", "MICROSOFTU", "MICROSOFTE", "MICROSOFTOM"))
_MSFT_ALIAS_PATTERN = r"(?<!\w)(?:microsoft|microsoftu|microsofte|microsoftom)(?!\w)"


def _mentioned_symbols(question: str) -> tuple[str, ...]:
    """Read exact frozen tickers and bounded verified company-name aliases."""
    mentioned = {
        symbol for symbol in FROZEN_SYMBOLS
        if re.search(r"(?<!\w)" + re.escape(symbol) + r"(?!\w)", question, re.I)
    }
    if re.search(_NVDA_ALIAS_PATTERN, question, re.I):
        mentioned.add("NVDA")
    if re.search(_TSLA_ALIAS_PATTERN, question, re.I):
        mentioned.add("TSLA")
    if re.search(_MSFT_ALIAS_PATTERN, question, re.I):
        mentioned.add("MSFT")
    return tuple(symbol for symbol in FROZEN_SYMBOLS if symbol in mentioned)


# These are metric/source abbreviations, not stock tickers. Only consider
# unsupported symbols in explicit ticker syntax; never treat every acronym
# in a financial question as a stock name.
_NON_TICKER_ABBREVIATIONS = frozenset((
    "MAE", "RMSE", "MSE", "BPS", "USD", "EUR", "UTC", "RSI",
    "SIP", "IEX", "NYSE", "ETF", "API", "JSON", "UP", "DOWN",
))


def _mentions_unsupported_forward_symbol(question: str) -> bool:
    """Fail closed on explicit unsupported ticker syntax in opt-in routing.

    This intentionally does not guess that arbitrary unmarked words or
    lowercase prose are stock symbols. Explicit ticker/stock cues accept
    mixed-case candidates including longer company names; the ordinary Slovak preposition "pre"
    remains uppercase-only to avoid interpreting everyday words as tickers.
    Cashtags and uppercase symbol pairs remain supported.
    """
    candidates: list[str] = []
    for pattern in (
        r"(?<!\w)\$([A-Za-z]{2,5})(?!\w)",
        r"(?<!\w)(?i:pre)\s+([A-Z]{2,5})(?!\w)",
        r"(?<!\w)(?i:ticker|symbol|akci\w*|stock)\s+([A-Za-z]{2,30})(?!\w)",
    ):
        candidates.extend(match.group(1) for match in re.finditer(pattern, question))
    # Scan overlapping pairs too: a non-overlapping match can skip the
    # third ticker in "NVDA, NVDA, IBM" and silently narrow the query.
    pair = (
        r"(?=(?<!\w)([A-Z]{2,5})\s*"
        r"(?i:vs\.?|versus|oproti|a|aj|/|,)\s*"
        r"([A-Z]{2,5})(?!\w))"
    )
    for match in re.finditer(pair, question):
        candidates.extend(match.groups())
    # Mixed/lowercase unsupported tickers in explicit stock comparisons
    # must not be dropped when the other side is a known frozen ticker.
    # Require whole-word tokens: "shadow" must not be split at its "a".
    mixed_pair = (
        r"(?=(?<!\w)([A-Za-z]{2,5})(?!\w)\s*"
        r"(?i:vs\.?|versus|oproti|a|aj|/|,)\s*"
        r"(?<!\w)([A-Za-z]{2,5})(?!\w))"
    )
    for match in re.finditer(mixed_pair, question):
        left, right = match.groups()
        # "NVDA, len NVDA" repeats one ticker; "len" is the Slovak
        # qualifier "only" here, not a second unsupported stock.
        # Do not exempt "NVDA, len ibm" or an unrelated follow-up.
        if right.casefold() == "len" and left.upper() in FROZEN_SYMBOLS:
            if re.match(
                r"\s+" + re.escape(left) + r"(?!\w)",
                question[match.end(2):], re.I,
            ):
                continue
        if any(token.upper() in FROZEN_SYMBOLS for token in (left, right)):
            candidates.extend((left, right))
    # A recognized company alias must not silently narrow a comparison
    # with another (possibly unsupported) company or ticker. Inspect both
    # orders and overlapping pairs in longer lists, including repeated aliases.
    join = r"\s*(?i:vs\.?|versus|oproti|a|aj|/|,)\s*"
    name = r"([A-Za-z]{2,30})(?!\w)"
    for alias_pattern in (_NVDA_ALIAS_PATTERN, _TSLA_ALIAS_PATTERN, _MSFT_ALIAS_PATTERN):
        for pattern in (
            r"(?=" + alias_pattern + join + name + r")",
            r"(?=(?<!\w)([A-Za-z]{2,30})" + join + alias_pattern + r")",
        ):
            candidates.extend(match.group(1) for match in re.finditer(
                pattern, question, re.I,
            ))
    return any(
        candidate.upper() not in FROZEN_SYMBOLS
        and candidate.upper() not in _NON_TICKER_ABBREVIATIONS
        and candidate.upper() not in _VERIFIED_NVDA_ALIASES
        and candidate.upper() not in _VERIFIED_TSLA_ALIASES
        and candidate.upper() not in _VERIFIED_MSFT_ALIASES
        for candidate in candidates
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
        symbols = _mentioned_symbols(user_question)
        # A single-ticker journal query cannot faithfully answer a question
        # naming several stocks. Never silently pick the first ticker.
        if len(symbols) > 1 or _mentions_unsupported_forward_symbol(user_question):
            return None
        ticker = symbols[0] if symbols else None
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
