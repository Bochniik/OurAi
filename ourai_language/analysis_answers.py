from __future__ import annotations

from .contracts import AnalysisSnapshot


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
        "LONG_BIAS": "rastový sklon (LONG_BIAS)",
        "SHORT_BIAS": "klesajúci sklon (SHORT_BIAS)",
        "NO_TRADE": "NO_TRADE – signál nie je dostatočne silný",
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
        key = item.casefold().strip()
        if not key or key in seen:
            continue
        seen.add(key)
        result.append(item.strip())
    return result


def _support_sk(snapshot: AnalysisSnapshot) -> str:
    reasons = _distinct(snapshot.reasons_for)
    if reasons:
        return ", ".join(reasons)
    pieces = [
        value
        for value in (snapshot.market_regime, snapshot.relative_strength, snapshot.volume_state)
        if value
    ]
    return ", ".join(pieces) if pieces else "podporné dôvody neboli v snapshote dodané"


def _support_en(snapshot: AnalysisSnapshot) -> str:
    return _support_sk(snapshot)


def _risks_sk(snapshot: AnalysisSnapshot) -> str:
    risks = _distinct(snapshot.reasons_against)
    parts: list[str] = []
    if risks:
        parts.extend(risks)
    if snapshot.uncertainty != "unavailable" and not any(
        snapshot.uncertainty.casefold() in risk.casefold() for risk in parts
    ):
        parts.append(f"{snapshot.uncertainty} neistota modelu")
    if snapshot.news_context:
        lowered = snapshot.news_context.casefold()
        if "no strong catalyst" in lowered or "bez" in lowered or "no catalyst" in lowered:
            parts.append("chýba výrazný potvrdený katalyzátor")
    return ", ".join(_distinct(tuple(parts))) if parts else "snapshot neobsahuje konkrétny zoznam rizík"


def _risks_en(snapshot: AnalysisSnapshot) -> str:
    risks = _distinct(snapshot.reasons_against)
    parts = list(risks)
    if snapshot.uncertainty != "unavailable" and not any(
        snapshot.uncertainty.casefold() in risk.casefold() for risk in parts
    ):
        parts.append(f"{snapshot.uncertainty} model uncertainty")
    if snapshot.news_context and "no strong catalyst" in snapshot.news_context.casefold():
        parts.append("no strong confirmed catalyst is supplied")
    return ", ".join(_distinct(tuple(parts))) if parts else "the snapshot does not provide a specific risk list"


def deterministic_analysis_answer(
    user_message: str,
    snapshot: AnalysisSnapshot,
) -> str | None:
    """Return a grounded answer for common Orion analysis questions.

    This deliberately handles only intents that can be answered exactly from the
    canonical snapshot. Unknown or open-ended questions fall through to the LLM.
    """

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
    why_intent = any(
        marker in lowered
        for marker in ("prečo", "preco", "why", "bullish", "bearish")
    )
    summary_intent = any(
        marker in lowered
        for marker in ("zhrň", "zhrn", "sumar", "summary", "jednoducho", "analýzu", "analyzu")
    )

    if risk_intent:
        if slovak:
            return (
                f"Najväčšie riziká pri {subject} sú: {_risks_sk(snapshot)}. "
                f"Model má {snapshot.uncertainty} neistotu, takže {_decision_phrase_sk(snapshot.decision)} "
                "nie je istota výsledku."
            )
        return (
            f"The main risks for {subject} are: {_risks_en(snapshot)}. "
            f"Model uncertainty is {snapshot.uncertainty}, so {_decision_phrase_en(snapshot.decision)} is not a certainty."
        )

    if action_intent:
        if slovak:
            return (
                f"Na základe samotného snapshotu by som {subject} čítal ako {_decision_phrase_sk(snapshot.decision)}. "
                f"P(UP) je {snapshot.p_up * 100:.1f} % a P(DOWN) {snapshot.p_down * 100:.1f} %. "
                "Neberiem to ako automatický príkaz na obchod; pri zmene dát sa musí znovu prepočítať analýza."
            )
        return (
            f"Based only on this snapshot, I would interpret {subject} as {_decision_phrase_en(snapshot.decision)}. "
            f"P(UP) is {snapshot.p_up * 100:.1f}% and P(DOWN) is {snapshot.p_down * 100:.1f}%. "
            "That is not an automatic trade instruction; the analysis should be recomputed when the inputs change."
        )

    if probability_intent:
        move = (
            f" Očakávaná veľkosť pohybu je {snapshot.expected_move_pct:.2f} %."
            if slovak and snapshot.expected_move_pct is not None
            else f" Expected move magnitude is {snapshot.expected_move_pct:.2f}%."
            if snapshot.expected_move_pct is not None
            else ""
        )
        if slovak:
            return (
                f"Pre {subject} na horizonte {snapshot.horizon_minutes} minút je P(UP) "
                f"{snapshot.p_up * 100:.1f} % a P(DOWN) {snapshot.p_down * 100:.1f} %.{move}"
            )
        return (
            f"For {subject} over {snapshot.horizon_minutes} minutes, P(UP) is "
            f"{snapshot.p_up * 100:.1f}% and P(DOWN) is {snapshot.p_down * 100:.1f}%.{move}"
        )

    if change_intent:
        if slovak:
            return (
                f"Konkrétny prah, ktorý by zmenil názor na {subject}, tento snapshot neobsahuje. "
                "Názor sa má zmeniť až po novom prepočte forecast enginu z aktuálnych dát; "
                f"súčasný signál podporujú najmä: {_support_sk(snapshot)}."
            )
        return (
            f"This snapshot does not contain an exact threshold that would change the view on {subject}. "
            "The view should change only after the forecast engine recomputes the signal from new data; "
            f"the current signal is mainly supported by: {_support_en(snapshot)}."
        )

    if why_intent:
        if slovak:
            return (
                f"OurAI má pri {subject} {_decision_phrase_sk(snapshot.decision)}, pretože snapshot uvádza najmä: "
                f"{_support_sk(snapshot)}. Na horizonte {snapshot.horizon_minutes} minút je P(UP) "
                f"{snapshot.p_up * 100:.1f} % oproti P(DOWN) {snapshot.p_down * 100:.1f} %. "
                f"Neistota je {snapshot.uncertainty}."
            )
        return (
            f"OurAI has {_decision_phrase_en(snapshot.decision)} on {subject} mainly because of: "
            f"{_support_en(snapshot)}. Over {snapshot.horizon_minutes} minutes, P(UP) is "
            f"{snapshot.p_up * 100:.1f}% versus P(DOWN) {snapshot.p_down * 100:.1f}%. "
            f"Uncertainty is {snapshot.uncertainty}."
        )

    if summary_intent:
        expected = (
            f", očakávaný pohyb približne {snapshot.expected_move_pct:.2f} %"
            if slovak and snapshot.expected_move_pct is not None
            else f", expected move about {snapshot.expected_move_pct:.2f}%"
            if snapshot.expected_move_pct is not None
            else ""
        )
        if slovak:
            return (
                f"{subject}: {_decision_phrase_sk(snapshot.decision)}. P(UP) {snapshot.p_up * 100:.1f} %, "
                f"P(DOWN) {snapshot.p_down * 100:.1f} %{expected}. Hlavné podporné faktory: "
                f"{_support_sk(snapshot)}. Riziká: {_risks_sk(snapshot)}."
            )
        return (
            f"{subject}: {_decision_phrase_en(snapshot.decision)}. P(UP) {snapshot.p_up * 100:.1f}%, "
            f"P(DOWN) {snapshot.p_down * 100:.1f}%{expected}. Main supporting factors: "
            f"{_support_en(snapshot)}. Risks: {_risks_en(snapshot)}."
        )

    return None
