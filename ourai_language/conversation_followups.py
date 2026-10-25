"""Safe contextual follow-up routing for verified retrospective OurAI results.

Only a recent, explicitly tagged answer from the historical report reader can
authorize historical follow-ups. We never turn a prior generated model message
into numerical evidence or infer a new source from conversational prose.
"""
from __future__ import annotations

import unicodedata
from typing import Iterable, Mapping

from .historical_answers import (
    corrected_baseline_followup_target, historical_question_kind,
    historical_source_correction_target, is_bare_baseline_followup,
    bare_day_wins_followup_target, historical_day_wins_count_context,
    is_bare_day_wins_share_followup,
    bare_day_wins_baseline_share_target, historical_day_wins_share_context,
    bare_day_wins_split_share_target,
)


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
    question = str(last_assistant.get("evidence_question") or last_user.get("content", ""))
    if historical_question_kind(question) is None:
        return None
    return question


def historical_followup_question(
    message: str,
    history: Iterable[Mapping[str, str]] | None,
) -> str | None:
    """Rewrite conservative short follow-ups, preserving source and split.

    Ambiguous long or unrelated turns stay with the normal language model.
    A narrowly scoped 'a teraz to porovnaj s X' changes only the baseline,
    never the verified historical source or its validation/lockbox split.
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
    # A short "A pri nulovom?" changes only the baseline of an immediately
    # preceding verified day-win *percentage*. Never infer the metric from
    # assistant prose, a generic MAE answer or a day-win count.
    share_baseline = bare_day_wins_baseline_share_target(raw)
    if share_baseline is not None:
        share_context = historical_day_wins_share_context(prior)
        if share_context is None or share_context[1] == share_baseline:
            return None
        split = share_context[0]
        scope = (
            "v historickej validácii" if split == "validation"
            else "na historickom lockbox teste"
        )
        baseline = (
            "lineárny baseline" if share_baseline == "linear"
            else "nulový baseline"
        )
        return f"Koľko percent trhových dní model v3 prekonal {baseline} {scope}?"

    # An elliptical split change after a verified day-win percentage keeps
    # the metric and baseline, but re-reads the *other* historical report.
    # Neither generated answer prose nor a prior MAE/count supplies evidence.
    share_split = bare_day_wins_split_share_target(raw)
    if share_split is not None:
        share_context = historical_day_wins_share_context(prior)
        if share_context is not None:
            if share_context[0] == share_split:
                return None
            baseline = (
                "lineárny baseline" if share_context[1] == "linear"
                else "nulový baseline"
            )
            scope = (
                "v historickej validácii" if share_split == "validation"
                else "na historickom lockbox teste"
            )
            return f"Koľko percent trhových dní model v3 prekonal {baseline} {scope}?"
        # Do not silently convert a day-win *count* into an MAE comparison.
        # Other verified MAE/summary split follow-ups retain existing routing.
        if historical_day_wins_count_context(prior) is not None:
            return None

    # "A koľko je to percent?" refers to the *immediately preceding*
    # verified day-win count, not to generic MAE or prediction hit rate.
    if is_bare_day_wins_share_followup(raw):
        context = historical_day_wins_count_context(prior)
        if context is None:
            return None
        source_split, source_baseline = context
        scope = (
            "v historickej validácii" if source_split == "validation"
            else "na historickom lockbox teste"
        )
        baseline = (
            "lineárny baseline" if source_baseline == "linear"
            else "nulový baseline"
        )
        return f"Koľko percent trhových dní model v3 prekonal {baseline} {scope}?"

    # "A teraz to porovnaj s nulovým" refers to the previous *verified*
    # retrospective report, not current market data. Accept only this
    # tightly scoped discourse transition; other uses of "teraz" remain
    # excluded along with forward/live/trading requests.
    day_wins_baseline = bare_day_wins_followup_target(raw)
    discourse_baseline_comparison = (
        is_bare_baseline_followup(raw) or day_wins_baseline is not None
    )
    if any(term in text for term in (
        "forward", "posledn", "najnov", "aktualn", "journal", "dennik",
        "zajtra", "dnes", "live", "obchod", "nakup", "predaj",
    )) or ("teraz" in text and not discourse_baseline_comparison):
        return None

    # A correction may change the requested retrospective source only when
    # the immediately preceding assistant answer came from the verified
    # historical reader. Never inherit the old lockbox/validation split.
    corrected = historical_source_correction_target(raw)
    if corrected == "validation":
        return "Aké boli výsledky historickej validácie v3?"
    if corrected == "test":
        return "Aký bol historický lockbox test v3?"
    if corrected == "ambiguous":
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

    # A metric switch is allowed only for a verified, single historical split.
    # A prior two-report overview cannot silently pick validation or lockbox.
    if day_wins_baseline is not None:
        if split not in ("validácii", "lockbox teste"):
            return None
        scope = (
            "v historickej validácii" if split == "validácii"
            else "na historickom lockbox teste"
        )
        baseline = "lineárny baseline" if day_wins_baseline == "linear" else "nulový baseline"
        return f"Koľko trhových dní model v3 prekonal {baseline} {scope}?"

    # For an exclusive correction, the positive target wins even though the
    # excluded baseline is also present in the user's sentence.
    baseline_correction = corrected_baseline_followup_target(raw)
    if baseline_correction == "null":
        target = "nulovým baseline"
    elif baseline_correction == "linear":
        target = "lineárnym modelom"
    elif any(term in text for term in ("nulov", "null")):
        target = "nulovým baseline"
    elif any(term in text for term in ("linearn", "linear")):
        target = "lineárnym modelom"
    elif "nulov" in _norm(prior) or "null" in _norm(prior):
        target = "nulovým baseline"
    elif "linearn" in _norm(prior) or "linear" in _norm(prior):
        target = "lineárnym modelom"
    else:
        target = "lineárnym a nulovým baseline"

    # A verified earlier answer does not make unrelated new questions about
    # Python, markets or financial concepts into historical follow-ups.
    # Only genuinely elliptical "why" prompts may inherit the report.
    why = text in (
        "a preco", "preco", "a z akeho dovodu", "z akeho dovodu",
        "a preco to tak je", "preco to tak je",
        "preco su tie vysledky rozdielne", "a preco su tie vysledky rozdielne",
    )
    if why:
        return "Prečo sa líšia historické výsledky modelu v3?"
    simple = text.startswith((
        "vysvetli", "a vysvetli", "jednoducho", "zhrn",
        "a co z toho", "co z toho", "co to znamena pre model",
    ))
    if simple:
        # Preserve referential summaries, not new standalone education requests.
        referential = (
            text in (
                "jednoducho", "zhrn", "vysvetli", "a vysvetli",
                "vysvetli to", "a vysvetli to", "zhrn to",
            )
            or text.startswith((
                "vysvetli to jednoducho", "a vysvetli to jednoducho",
                "zhrn vysledky", "vysvetli tie vysledky",
                "a co z toho", "co z toho", "co to znamena pre model",
            ))
        )
        if not referential:
            return None
        if split == "lockbox teste":
            return "Zhrň historické výsledky lockbox testu v3 jednoducho."
        if split == "validácii":
            return "Zhrň výsledky historickej validácie v3 jednoducho."
        return "Zhrň historické výsledky modelu v3 jednoducho."
    if text.startswith(("a pri ktorej", "a pri ktorych", "a ktora akcia", "a ktore akcie")):
        # "A ktorá akcia má najvyššiu dividendu?" changes the subject.
        # Only a short elliptical reference or an explicit error/performance
        # question may inherit the previous historical report.
        elliptical = text in (
            "a pri ktorej", "a pri ktorych",
            "a pri ktorej akcii", "a pri ktorych akciach",
            "a ktora akcia", "a ktore akcie",
        )
        historical_error = any(cue in text for cue in (
            "chyb", "mae", "odchyl", "nepresn", "presn", "vysledk",
        ))
        if elliptical or historical_error:
            return "Pri ktorých akciách robí historický model v3 najväčšie chyby?"
        return None

    # A fresh comparison of unrelated subjects is not a request to compare
    # historical model evidence. Keep only an explicit result/baseline target
    # or a short reference back to the immediately preceding report.
    if text.startswith(("porovnaj ", "a porovnaj ")):
        referential_comparison = text.startswith((
            "porovnaj tie vysledky", "a porovnaj tie vysledky",
        ))
        referential_comparison = referential_comparison or text in (
            "porovnaj to", "a porovnaj to", "porovnaj ich", "a porovnaj ich",
        )
        grounded_comparison = any(term in text for term in (
            "vysledk", "baseline", "nulov", "linearn", "linear",
            "lockbox", "validac", "model v3", "mae",
        ))
        if not (referential_comparison or grounded_comparison):
            return None

    # Comparison-like vocabulary can also introduce an independent question.
    # Do not turn prices or a comparison between unrelated subjects into
    # retrospective model metrics merely because history contained a report.
    if text.startswith(("o kolko stoji ", "a o kolko stoji ", "kolko stoji ")):
        return None
    if text.startswith(("aky rozdiel medzi ", "a aky rozdiel medzi ")):
        if not any(term in text for term in (
            "vysled", "model", "baseline", "mae", "lockbox",
            "validac", "histor", "nulov", "linearn",
        )):
            return None

    # Independent price/delta requests should not inherit historical evidence.
    # Require an explicit metric, prior-result reference or comparison baseline.
    if text.startswith(("o kolko ", "a o kolko ")):
        if not any(marker in text for marker in (
            "to ", "toho ", "tie vysledky", "model", "baseline", "mae",
            "chyb", "presn", "leps", "hors", "nulov", "linearn",
            "lockbox", "validac",
        )):
            return None

    # An explicit new comparison target must refer to known model evidence.
    # "A oproti minulému roku?" is not a request for historical v3 metrics.
    if text.startswith(("a oproti ", "oproti ", "a v porovnani s ", "v porovnani s ")):
        if not any(cue in text for cue in (
            "linearn", "linear", "nulov", "null", "baseline", "model",
            "v3", "mae", "validac", "lockbox", "vysledk",
            "tomu", "tymto", "predchadzajucim vysledk",
        )):
            return None

    comparative = prior_kind is not None and prior_kind.startswith("compare")
    comparable_prompt = any(term in text for term in (
        "o kolko", "oproti", "v porovnani", "porovnaj", "kolko leps",
        "kolko hors", "aky rozdiel", "rozdiel oproti", "prekon",
    ))
    baseline_followup = (
        text.startswith(("a co ", "a oproti ", "a ten ", "a s ", "a k "))
        and any(term in text for term in (
            "linearn", "linear", "nulov", "null", "baseline",
        ))
    )
    change_split = (
        any(term in text for term in ("lockbox", "validac", "testovac", "testovaci"))
        and text.startswith(("a ", "co ", "ako "))
    )
    # Source provenance alone is not permission to hijack an arbitrary new
    # question: only a clear referential cue can inherit the old comparison.
    if not (comparable_prompt or baseline_followup or change_split):
        return None
    if comparative or comparable_prompt or baseline_followup:
        scope = (
            "na historickom lockbox teste" if split == "lockbox teste"
            else "v historickej validácii"
        ) if split else ""
        return f"Porovnaj výsledky modelu v3 {scope} s {target}.".replace("  ", " ")
    if change_split:
        scope = (
            "na historickom lockbox teste" if split == "lockbox teste"
            else "v historickej validácii"
        )
        return f"Aká bola MAE modelu v3 {scope}?"
    return None
