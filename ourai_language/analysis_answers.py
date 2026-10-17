"""Grounded, language-aware explanations of canonical Orion forecast snapshots.

The deterministic path deliberately covers common forecast questions only. It
paraphrases known structured phrases without inventing additional market facts.
"""
from __future__ import annotations

import re

from .contracts import AnalysisSnapshot


_SK_REASON_TERMS = {
    "positive short-term momentum": "pozitívne krátkodobé momentum",
    "positive momentum": "pozitívne momentum",
    "relative strength versus qqq": "relatívna sila voči QQQ",
    "relative strength": "relatívna sila",
    "positive vs qqq": "pozitívna relatívna sila voči QQQ",
    "elevated volume": "zvýšený objem obchodovania",
    "bullish": "rastové trhové podmienky",
    "bearish": "klesajúce trhové podmienky",
    "medium uncertainty": "stredná neistota modelu",
    "low uncertainty": "nízka neistota modelu",
    "high uncertainty": "vysoká neistota modelu",
    "no strong news catalyst supplied": "v dodaných údajoch nie je potvrdený výrazný spravodajský katalyzátor",
    "no strong catalyst supplied": "v dodaných údajoch nie je potvrdený výrazný katalyzátor",
    "no catalyst": "v dodaných údajoch nie je potvrdený katalyzátor",
}

_SK_UNCERTAINTY = {
    "low": "nízka",
    "medium": "stredná",
    "high": "vysoká",
    "unavailable": "nezadaná",
}


def _is_slovak(text: str) -> bool:
    lowered = text.casefold()
    markers = (
        "prečo", "preco", "aké", "ake", "rizik", "robil", "urobil", "zhrň",
        "zhrn", "pravdepodob", "čo", "co", "analýz", "analyz", "predik",
        "názor", "nazor", "zmenilo", "jednoducho",
    )
    return any(marker in lowered for marker in markers)


def _subject(snapshot: AnalysisSnapshot) -> str:
    if snapshot.company_name:
        return f"{snapshot.ticker} ({snapshot.company_name})"
    return snapshot.ticker


def _decision_phrase_sk(decision: str) -> str:
    return {
        "LONG_BIAS": "rastovú preferenciu (LONG_BIAS)",
        "SHORT_BIAS": "preferenciu poklesu (SHORT_BIAS)",
        "NO_TRADE": "NO_TRADE, teda bez dostatočného obchodného signálu",
        "UNAVAILABLE": "nedostupné rozhodnutie",
    }.get(decision, decision)


def _decision_phrase_en(decision: str) -> str:
    return {
        "LONG_BIAS": "a bullish bias (LONG_BIAS)",
        "SHORT_BIAS": "a bearish bias (SHORT_BIAS)",
        "NO_TRADE": "NO_TRADE because the signal is not strong enough",
        "UNAVAILABLE": "an unavailable decision",
    }.get(decision, decision)


def _distinct(items: tuple[str, ...]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        normalized = " ".join(item.casefold().strip().split())
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        result.append(item.strip())
    return result


def _render_sk_fact(text: str) -> str:
    """Translate only known snapshot descriptors; quote unfamiliar raw fields."""
    raw = str(text).strip()
    key = " ".join(raw.casefold().split())
    mapped = _SK_REASON_TERMS.get(key)
    if mapped is not None:
        return mapped
    # Unknown free-form source is not 'translated' by inventing its meaning.
    return f"„{raw}“"


def _is_uncertainty_fact(text: str) -> bool:
    lowered = text.casefold()
    return "uncertainty" in lowered or "neistot" in lowered


def _is_missing_catalyst_fact(text: str) -> bool:
    lowered = text.casefold().strip()
    return (
        ("catalyst" in lowered and ("no " in lowered or "not supplied" in lowered))
        or "neuvádza katalyzátor" in lowered
    )


def _fmt_sk(value: float, precision: int = 1) -> str:
    return f"{value:.{precision}f}".replace(".", ",")


def _uncertainty_sk(snapshot: AnalysisSnapshot) -> str:
    if snapshot.uncertainty == "unavailable":
        return "Miera neistoty nebola dodaná."
    return f"Neistota modelu je {_SK_UNCERTAINTY[snapshot.uncertainty]}."


def _support_sk(snapshot: AnalysisSnapshot) -> str:
    reasons = _distinct(snapshot.reasons_for)
    if not reasons:
        reasons = [
            value for value in (snapshot.market_regime, snapshot.relative_strength, snapshot.volume_state)
            if value
        ]
    if not reasons:
        return "snapshot neuvádza konkrétne podporné faktory"
    rendered = _distinct(tuple(_render_sk_fact(item) for item in reasons))
    return ", ".join(rendered)


def _support_en(snapshot: AnalysisSnapshot) -> str:
    reasons = _distinct(snapshot.reasons_for)
    if not reasons:
        reasons = [
            value for value in (snapshot.market_regime, snapshot.relative_strength, snapshot.volume_state)
            if value
        ]
    return ", ".join(reasons) if reasons else "no supporting reasons were supplied"


def _risk_items_sk(snapshot: AnalysisSnapshot) -> list[str]:
    parts: list[str] = []
    saw_uncertainty = False
    saw_catalyst = False
    for item in _distinct(snapshot.reasons_against):
        if _is_uncertainty_fact(item):
            # The normalized canonical uncertainty is the source of truth.
            saw_uncertainty = True
            continue
        if _is_missing_catalyst_fact(item):
            if not saw_catalyst:
                parts.append("v dodaných údajoch nie je potvrdený výrazný katalyzátor")
                saw_catalyst = True
            continue
        parts.append(_render_sk_fact(item))
    if snapshot.news_context and _is_missing_catalyst_fact(snapshot.news_context) and not saw_catalyst:
        parts.append("v dodaných údajoch nie je potvrdený výrazný katalyzátor")
    if snapshot.uncertainty != "unavailable":
        parts.append(f"{_SK_UNCERTAINTY[snapshot.uncertainty]} neistota modelu")
    elif saw_uncertainty:
        parts.append("miera neistoty nie je dostupná")
    return _distinct(tuple(parts))


def _risks_sk(snapshot: AnalysisSnapshot) -> str:
    items = _risk_items_sk(snapshot)
    return ", ".join(items) if items else "snapshot neuvádza konkrétne rizikové faktory"


def _risks_en(snapshot: AnalysisSnapshot) -> str:
    items = _distinct(snapshot.reasons_against)
    if snapshot.uncertainty != "unavailable" and not any(
        _is_uncertainty_fact(item) for item in items
    ):
        items.append(f"{snapshot.uncertainty} model uncertainty")
    if snapshot.news_context and _is_missing_catalyst_fact(snapshot.news_context):
        if not any(_is_missing_catalyst_fact(item) for item in items):
            items.append("no strong confirmed catalyst is supplied")
    return ", ".join(items) if items else "the snapshot does not provide a specific risk list"


def deterministic_analysis_answer(user_message: str, snapshot: AnalysisSnapshot) -> str | None:
    """Return common-analysis answers from supplied facts; other questions use the LLM."""
    text = str(user_message or "").strip()
    if not text:
        return None
    lowered = text.casefold()
    slovak = _is_slovak(text)
    subject = _subject(snapshot)

    risk_intent = any(marker in lowered for marker in ("rizik", "risk", "nevýhod", "nevyhod"))
    action_intent = any(
        marker in lowered
        for marker in ("čo by si", "co by si", "čo by som", "co by som", "robil", "urobil", "what would you do")
    )
    probability_intent = any(
        marker in lowered
        for marker in ("pravdepodob", "p(up", "p(down", "probability", "šanca", "sanca")
    )
    change_intent = any(
        marker in lowered
        for marker in ("zmenilo", "zmeniť názor", "zmenit nazor", "change your view", "change the view")
    )
    why_intent = any(marker in lowered for marker in ("prečo", "preco", "why", "bullish", "bearish"))
    summary_intent = any(
        marker in lowered
        for marker in ("zhrň", "zhrn", "sumar", "summary", "jednoducho", "analýzu", "analyzu")
    )

    if risk_intent:
        if slovak:
            return (
                f"Pri {subject} snapshot uvádza tieto riziká a obmedzenia: {_risks_sk(snapshot)}. "
                f"Rozhodnutie {_decision_phrase_sk(snapshot.decision)} nie je zárukou budúceho vývoja ceny."
            )
        return (
            f"The main risks for {subject} are: {_risks_en(snapshot)}. "
            f"The decision {_decision_phrase_en(snapshot.decision)} is not a guaranteed outcome."
        )

    if action_intent:
        if slovak:
            return (
                f"Pri {subject} snapshot naznačuje {_decision_phrase_sk(snapshot.decision)}. "
                f"P(UP) je {_fmt_sk(snapshot.p_up * 100)} % a P(DOWN) {_fmt_sk(snapshot.p_down * 100)} %. "
                "Nie je to automatický príkaz na obchod; pred rozhodnutím treba zohľadniť "
                "aktuálne údaje, náklady a riziko."
            )
        return (
            f"Based only on this snapshot, {subject} shows {_decision_phrase_en(snapshot.decision)}. "
            f"P(UP) is {snapshot.p_up * 100:.1f}% and P(DOWN) is {snapshot.p_down * 100:.1f}%. "
            "That is not an automatic trade instruction; new data, costs and risk need to be considered."
        )

    if probability_intent:
        if slovak:
            move = (
                f" Odhadovaná veľkosť pohybu je {_fmt_sk(snapshot.expected_move_pct, 2)} %; "
                "nejde o garantovaný výnos ani o určenie smeru."
                if snapshot.expected_move_pct is not None else ""
            )
            return (
                f"Pre {subject} na horizonte {snapshot.horizon_minutes} minút je P(UP) "
                f"{_fmt_sk(snapshot.p_up * 100)} % a P(DOWN) {_fmt_sk(snapshot.p_down * 100)} %.{move}"
            )
        move = (
            f" Expected magnitude is {snapshot.expected_move_pct:.2f}%, not a guaranteed return."
            if snapshot.expected_move_pct is not None else ""
        )
        return (
            f"For {subject} over {snapshot.horizon_minutes} minutes, P(UP) is "
            f"{snapshot.p_up * 100:.1f}% and P(DOWN) is {snapshot.p_down * 100:.1f}%.{move}"
        )

    if change_intent:
        if slovak:
            return (
                f"Snapshot neobsahuje konkrétny prah, pri ktorom by sa zmenilo hodnotenie {subject}. "
                "Rozhodnutie môže zmeniť až nový výpočet nad aktuálnymi údajmi. "
                f"Súčasné podporné faktory: {_support_sk(snapshot)}."
            )
        return (
            f"This snapshot does not contain an exact threshold for changing the view on {subject}. "
            "A new forecast using fresh inputs is required. "
            f"Current supporting factors: {_support_en(snapshot)}."
        )

    if why_intent:
        if slovak:
            return (
                f"Pri {subject} snapshot uvádza {_decision_phrase_sk(snapshot.decision)}. "
                f"Podporné faktory: {_support_sk(snapshot)}. "
                f"Na horizonte {snapshot.horizon_minutes} minút model odhaduje P(UP) "
                f"{_fmt_sk(snapshot.p_up * 100)} % a P(DOWN) {_fmt_sk(snapshot.p_down * 100)} %. "
                f"{_uncertainty_sk(snapshot)}"
            )
        return (
            f"For {subject}, the snapshot shows {_decision_phrase_en(snapshot.decision)} "
            f"based on {_support_en(snapshot)}. Over {snapshot.horizon_minutes} minutes, P(UP) is "
            f"{snapshot.p_up * 100:.1f}% versus P(DOWN) {snapshot.p_down * 100:.1f}%. "
            f"Uncertainty is {snapshot.uncertainty}."
        )

    if summary_intent:
        if slovak:
            move = (
                f" Odhadovaná veľkosť pohybu: {_fmt_sk(snapshot.expected_move_pct, 2)} %."
                if snapshot.expected_move_pct is not None else ""
            )
            return (
                f"{subject}: {_decision_phrase_sk(snapshot.decision)}. "
                f"P(UP) {_fmt_sk(snapshot.p_up * 100)} %, P(DOWN) {_fmt_sk(snapshot.p_down * 100)} %."
                f"{move} Podporné faktory: {_support_sk(snapshot)}. "
                f"Riziká a obmedzenia: {_risks_sk(snapshot)}."
            )
        move = (
            f", expected magnitude {snapshot.expected_move_pct:.2f}%"
            if snapshot.expected_move_pct is not None else ""
        )
        return (
            f"{subject}: {_decision_phrase_en(snapshot.decision)}. "
            f"P(UP) {snapshot.p_up * 100:.1f}%, P(DOWN) {snapshot.p_down * 100:.1f}%{move}. "
            f"Supporting factors: {_support_en(snapshot)}. Risks: {_risks_en(snapshot)}."
        )

    return None
