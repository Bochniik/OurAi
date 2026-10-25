"""Read-only, deterministic explanations of v3 retrospective magnitude reports.

Never equate historical validation/lockbox performance with forward performance.
The report files are local inputs, not produced or modified by this module.
"""
from __future__ import annotations

import json
import math
import re
import unicodedata
from pathlib import Path
from typing import Any

PROTOCOL = "ourai-v3-nonlinear-research-preregistration-v1"
REPORT_FILES = {
    "validation": "nonlinear_magnitude_validation_v1.json",
    "test": "nonlinear_magnitude_test_lockbox_v1.json",
}


def _normalize(message: str) -> str:
    folded = unicodedata.normalize("NFKD", message.casefold())
    return "".join(c for c in folded if not unicodedata.combining(c))


def is_historical_educational_request(message: str) -> bool:
    """Recognize explicit definitions without opening retrospective reports.

    Require a definition clause, not just the word "vysvetli": explaining
    observed validation results is a legitimate verified-reader request.
    """
    text = _normalize(str(message or "")).strip().strip('.?!„“"')
    text = re.sub(r"^prosim[, ]+", "", text)
    definitions = (
        "co znamena ", "co je ", "ako funguje ", "ako vlastne funguje ",
        "na co sluzi ", "aky je vyznam ", "aky je rozdiel medzi ",
        "vysvetli pojem ", "define ", "what does ", "how does ",
        "explain what ", "what is the meaning of ",
    )
    if text.startswith(definitions):
        return True
    # An English "what is the MAE of model v3" can ask for a measured
    # value; generic "what is historical MAE" remains a definition.
    if text.startswith("what is "):
        return not bool(re.search(
            r"\b(?:mae|accuracy|error rate)\b.*\b(?:of|for)\b"
            r".*\b(?:ourai|model|v3)\b",
            text,
        ))
    preface = re.match(
        r"^(?:(?:mozes|vies|dokazes)(?: mi)?(?: prosim)? )?"
        r"(?:vysvetli|vysvetlit)(?: mi)?[, ]+",
        text,
    )
    if preface and text[preface.end():].startswith(definitions):
        return True
    if text.startswith("can you explain "):
        subject = text[len("can you explain "):]
        return subject.startswith((
            "what does ", "what is ", "how does ", "the meaning of ",
        )) or bool(re.match(r"^what .+ means(?: |$)", subject))
    return False


def historical_source_correction_target(message: str) -> str | None:
    """Parse a *bare* source correction, not a standalone evidence request.

    Returns validation/test for an unambiguous short correction, "ambiguous"
    for other correction phrasing, or None for ordinary requests. A caller
    must require a recent verified historical turn before using the target.
    """
    text = _normalize(str(message or "")).strip().strip('.?!„“"')
    prefix = r"(?:nie\s*,?\s*)?(?:myslel(?:a)? som|mal(?:a)? som na mysli)\b"
    if not re.match(r"^" + prefix, text):
        return None
    match = re.fullmatch(
        prefix + r"\s+(?:tu\s+)?(?:historicku\s+)?"
        r"(validaciu|validacne vysledky|lockbox(?: test)?|test lockbox)",
        text,
    )
    if match is None:
        return "ambiguous"
    return "validation" if match.group(1).startswith("validac") else "test"


def corrected_baseline_followup_target(message: str) -> str | None:
    """Parse a bounded 'only X, not Y' correction, without using history.

    This is not a standalone report request. Callers may resolve it only
    after an immediately preceding verified historical answer. Reject a
    contradictory exclusion or any extra clause rather than guessing.
    """
    text = _normalize(str(message or "")).strip().strip('.?!„“"')
    match = re.fullmatch(
        r"(?:nie,\s*)?(?:a\s+)?teraz\s+to\s+porovnaj\s+"
        r"(?:iba|len)\s+(?:s|so)\s+"
        r"(nulov\w*|null|linearn\w*|linear)"
        r"(?:\s+(?:baseline|modelom))?\s*,\s*"
        r"(?:a\s+)?nie\s+(?:s|so)\s+"
        r"(nulov\w*|null|linearn\w*|linear)"
        r"(?:\s+(?:baseline|modelom))?",
        text,
    )
    if match is None:
        return None
    requested, excluded = (
        "linear" if word.startswith(("linearn", "linear")) else "null"
        for word in match.groups()
    )
    return requested if requested != excluded else None


def is_bare_baseline_followup(message: str) -> bool:
    """Only a verified historical conversation may resolve this elliptical request.

    A standalone "A teraz to porovnaj s nulovým" has no known report split;
    it must not open validation and lockbox reports by default.
    """
    if corrected_baseline_followup_target(message) is not None:
        return True
    text = _normalize(str(message or "")).strip().strip('.?!„“"')
    return bool(re.fullmatch(
        r"(?:a )?teraz (?:to )?porovnaj (?:to )?(?:s|so) "
        r"(?:nulov\w*|null|linearn\w*|linear)"
        r"(?: (?:baseline|modelom))?",
        text,
    ))


def bare_day_wins_followup_target(message: str) -> str | None:
    """Recognize only bounded elliptical requests to change MAE into day wins.

    The target is NOT permission to open a report. A caller must first check
    the immediately preceding verified historical source and split.
    """
    text = _normalize(str(message or "")).strip().strip('.?!„“"')
    match = re.fullmatch(
        r"(?:dobre,\s*ale\s+teraz\s+mi\s+namiesto\s+mae\s+ukaz,\s*|a\s+)"
        r"kolko\s+(?:trhovych\s+)?dni\s+model\s+(?:v3\s+)?prekonal\s+"
        r"(linearn\w*|linear|nulov\w*|null)"
        r"(?:\s+(?:baseline|model))?",
        text,
    )
    if match is None:
        return None
    return "linear" if match.group(1).startswith(("linearn", "linear")) else "null"


def is_bare_day_wins_share_followup(message: str) -> bool:
    """A referential percentage question is not a standalone report request."""
    text = _normalize(str(message or "")).strip().strip('.?!„“"')
    return bool(re.fullmatch(
        r"(?:a )?(?:kolko je to percent|kolko percent to je|"
        r"ake percento to je|kolko je to v percentach)",
        text,
    ))


def bare_day_wins_baseline_share_target(message: str) -> str | None:
    """Parse only a short baseline change after a verified day-win *share*.

    A bare "A pri nulovom?" has no independent report/split/metric scope.
    It must never authorize a reader without a verified previous turn.
    """
    text = _normalize(str(message or "")).strip().strip('.?!„“"')
    match = re.fullmatch(
        r"(?:a\\s+)?pri\\s+(nulov\\w*|null|linearn\\w*|linear)"
        r"(?:\\s+(?:baseline|modeli))?",
        text,
    )
    if match is None:
        return None
    return "linear" if match.group(1).startswith(("linearn", "linear")) else "null"


def historical_day_wins_share_context(message: str) -> tuple[str, str] | None:
    """Identify a single split and baseline in an explicit day-win share query.

    The caller separately verifies the source of the immediately preceding
    answer. A day-win count, MAE comparison or two-split overview is not
    sufficient to carry the percentage metric into a new baseline.
    """
    text = _normalize(str(message or "")).strip().strip('.?!„“"')
    match = re.fullmatch(
        r"(?:kolko percent trhovych dni|ake percento dni) model v3 prekonal "
        r"(linearn\\w*|linear|nulov\\w*|null) baseline "
        r"(v historickej validacii|na historickom lockbox teste)",
        text,
    )
    if match is None:
        return None
    baseline = "linear" if match.group(1).startswith(("linearn", "linear")) else "null"
    split = "validation" if match.group(2).startswith("v historickej") else "test"
    return split, baseline


def historical_day_wins_count_context(message: str) -> tuple[str, str] | None:
    """Recover a split and baseline only from an explicit day-win count query.

    Callers must independently require an immediately preceding, verified
    historical answer. Neither generic MAE nor two-report summaries qualify.
    """
    text = _normalize(str(message or "")).strip().strip('.?!„“"')
    match = re.fullmatch(
        r"kolko (?:trhovych )?dni model (?:v3 )?prekonal "
        r"(linearn\w*|linear|nulov\w*|null) (?:baseline|model) "
        r"(v historickej validacii|na historickom lockbox teste)",
        text,
    )
    if match is None:
        return None
    baseline = "linear" if match.group(1).startswith(("linearn", "linear")) else "null"
    split = "validation" if match.group(2).startswith("v historickej") else "test"
    return split, baseline


def historical_question_kind(message: str) -> str | None:
    """Classify only model-performance requests, never finance definitions.

    Latest/forward predictions belong to the verified forward journal, not
    retrospective files. Explicit historical requests may contain 'last' in
    another context, so historical markers take precedence there.
    """
    text = _normalize(str(message or "").strip())
    if (not text or is_historical_educational_request(message)
            or is_bare_baseline_followup(message)
            or bare_day_wins_followup_target(message) is not None
            or is_bare_day_wins_share_followup(message)
            or bare_day_wins_baseline_share_target(message) is not None):
        return None
    # A bare "Nie, myslel som validáciu" needs a verified preceding source.
    # The standalone deterministic reader must never assume that context.
    if historical_source_correction_target(message) is not None:
        return None
    if explicit_comparison_baseline(message) == "ambiguous":
        return None
    historic = any(term in text for term in (
        "historick", "validac", "lockbox", "retrospekt", "backtest",
        "testovac", "testovaci", "testovych", "historical", "validation",
        "held-out",
    ))
    compare_action = any(term in text for term in (
        "porovnaj", "porovnanie", "porovnat", "porovnava", "prekon",
        "lepsi", "lepsie", "horsie", "rozdiel", "compare", "better than",
        "worse than", "beat the",
    ))
    compare_subject = any(term in text for term in (
        "linearn", "linear", "nulov", "null", "baseline", "model v3",
        "v3 model", "svoje vysledk", "tvoje vysledk",
    ))
    comparison = compare_action and compare_subject
    performance = any(term in text for term in (
        "ako presn", "presnost", "presne boli", "mae", "najvacsie chyb",
        "najvacsich chyb", "najhorsie", "chybovost", "chyby model",
        "model robi chyb", "performance", "accurac", "errors",
        "porovnaj model", "porovnanie model", "baseline",
    ))
    # An educational definition must not be misrepresented as the model's
    # measured performance just because it happens to mention MAE/baseline.
    if text.startswith(("co je ", "co znamena ", "what is ", "define ")) and not historic:
        if not any(term in text for term in ("tvoj", "moj", "nas ", "ourai", "v3")):
            return None
    if not (historic or performance or comparison):
        return None
    if not historic and any(term in text for term in (
        "posledn", "najnov", "aktualn", "latest", "forward", "dennik", "journal",
        "shadow", "paper trading", "paper-trading", "live",
    )):
        return None
    if any(term in text for term in ("pri ktorych", "ktora akci", "ktore akci", "per-symbol", "podla akci", "by stock", "which stock", "najvacsie chyb", "najvacsich chyb")):
        return "symbols"
    if "forward" in text or "live" in text:
        return "boundary"
    if historic and text.startswith(("preco", "a preco", "z akeho dovodu")):
        return "causes"
    if historic and any(term in text for term in (
        "jednoducho", "jednoduchsie", "zhrn", "co z toho vyplyva",
    )):
        return "simple"
    if comparison:
        if "lockbox" in text or "testovac" in text or "held-out" in text or "testovaci" in text:
            return "compare_test"
        if "validac" in text or "validation" in text:
            return "compare_validation"
        return "compare"
    if "lockbox" in text or "testovac" in text or "held-out" in text or "testovaci" in text:
        return "test"
    if "validac" in text or "validation" in text:
        return "validation"
    if "porovnaj" in text or "porovnanie" in text or "compare" in text:
        return "compare"
    return "overview"


def _nonnegative(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"invalid historical metric: {label}")
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"invalid historical metric: {label}")
    return number


def _count(value: Any, label: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"invalid historical count: {label}")
    return value


def _read_report(directory: Path, split: str) -> dict[str, Any] | None:
    path = directory / REPORT_FILES[split]
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("historical report is not an object")
    if data.get("protocol_name") != PROTOCOL or data.get("horizon_minutes") != 15:
        raise ValueError("historical report protocol/horizon mismatch")
    if not isinstance(data.get("dataset_id"), str) or not data["dataset_id"]:
        raise ValueError("historical report missing dataset identity")
    if not isinstance(data.get("protocol_sha256"), str) or not re.fullmatch(
        r"[0-9a-f]{64}", data["protocol_sha256"]
    ):
        raise ValueError("historical report missing protocol hash")
    if split == "validation":
        if data.get("test_lockbox_accessed") is not False or data.get("original_validation_used") is not True:
            raise ValueError("historical validation provenance mismatch")
    else:
        if data.get("test_lockbox_accessed") is not True or data.get("post_test_tuning_allowed") is not False:
            raise ValueError("historical test provenance mismatch")
    if type(data.get(split + "_supported")) is not bool:
        raise ValueError("historical report missing gate outcome")
    results = data.get(split)
    if not isinstance(results, dict) or not isinstance(results.get("overall"), dict):
        raise ValueError("historical report missing overall metrics")
    overall = results["overall"]
    count = _count(overall.get("evaluated_count"), "evaluated_count")
    if count == 0:
        raise ValueError("historical report contains zero evaluations")
    metrics = {
        field: _nonnegative(overall.get(field), field)
        for field in ("nonlinear_mae_bps", "linear_mae_bps", "null_mae_bps")
    }
    days = _count(results.get("market_date_count"), "market_date_count")
    if not days:
        raise ValueError("historical report contains zero market days")
    wins = {}
    for field in ("market_date_wins_vs_linear", "market_date_wins_vs_null"):
        wins[field] = _count(results.get(field), field)
        if wins[field] > days:
            raise ValueError("historical wins exceed observed market days")
    return {
        "dataset_id": data["dataset_id"],
        "protocol_sha256": data["protocol_sha256"],
        "supported": data[split + "_supported"],
        "count": count,
        "days": days,
        **metrics,
        **wins,
    }


def _fmt(value: float) -> str:
    return f"{value:.2f}".replace(".", ",")


def _summary(split: str, report: dict[str, Any]) -> str:
    name = "Historická validácia" if split == "validation" else "Historický test lockbox"
    accepted = "splnil" if report["supported"] else "nesplnil"
    return (
        f"{name} v3 (15 min, absolútna veľkosť pohybu): "
        f"{report['count']} vyhodnotených vzoriek, {report['days']} trhových dní. "
        f"MAE champion {_fmt(report['nonlinear_mae_bps'])} bps; "
        f"lineárny baseline {_fmt(report['linear_mae_bps'])} bps; "
        f"nulový baseline {_fmt(report['null_mae_bps'])} bps. "
        f"Víťazstvá po dňoch: {report['market_date_wins_vs_linear']}/{report['days']} "
        f"voči lineárnemu, {report['market_date_wins_vs_null']}/{report['days']} "
        f"voči nulovému baseline. Model {accepted} príslušné historické kritériá."
    )


def _baseline_comparison(split: str, report: dict[str, Any], baseline: str) -> str:
    """Describe a baseline difference, never converting it to hit rate."""
    model_mae = report["nonlinear_mae_bps"]
    baseline_mae = report[f"{baseline}_mae_bps"]
    baseline_label = "lineárny" if baseline == "linear" else "nulový"
    split_label = "Validácia" if split == "validation" else "Lockbox"
    difference = baseline_mae - model_mae
    if abs(difference) < 1e-12:
        verdict = "MAE je rovnaká."
    else:
        change = "nižšia" if difference > 0 else "vyššia"
        relative = (
            f" ({_fmt(abs(difference) / baseline_mae * 100)} %)"
            if baseline_mae > 0 else ""
        )
        verdict = (
            f"MAE OurAI je {change} o {_fmt(abs(difference))} bps{relative}."
        )
    return (
        f"{split_label}: OurAI v3 {_fmt(model_mae)} bps, "
        f"{baseline_label} baseline {_fmt(baseline_mae)} bps. {verdict}"
    )


def _day_wins_comparison(
    split: str, report: dict[str, Any], baseline: str, *, include_share: bool = False,
) -> str:
    """Describe observed daily MAE wins; their share is not directional accuracy."""
    label = "lineárnemu" if baseline == "linear" else "nulovému"
    scope = "V historickej validácii" if split == "validation" else "V historickom lockbox teste"
    wins = report[f"market_date_wins_vs_{baseline}"]
    result = (
        f"{scope} dosiahol OurAI v3 nižšiu dennú MAE oproti {label} baseline "
        f"v {wins} z {report['days']} trhových dní."
    )
    if include_share:
        # The validated report guarantees days > 0 and 0 <= wins <= days.
        result += (
            f" To predstavuje {_fmt(100 * wins / report['days'])} % "
            "sledovaných trhových dní, nie presnosť jednotlivých predikcií."
        )
    return result


def _day_wins_metric_request(message: str) -> tuple[bool, bool]:
    """Identify explicit daily-win count/share questions, not generic accuracy.

    The existing deterministic router must first classify the question as
    retrospective comparison; this helper never opens a report on its own.
    """
    text = _normalize(message)
    if not re.search(r"\bprekonal\b", text):
        return False, False
    share = bool(re.search(
        r"\b(?:ake percento|kolko percent)\s+(?:trhovych\s+)?dni\b", text
    ))
    count = bool(re.search(r"\bkolko\s+(?:trhovych\s+)?dni\b", text))
    return count or share, share


def explicit_comparison_baseline(message: str) -> str | None:
    """Honor a clearly negated baseline contrast; reject unresolved exclusions.

    Return linear/null/both for explicit targets, None when unspecified, and
    ambiguous when a negative instruction has no supported positive target.
    Never infer an excluded baseline as requested just because it is named.
    """
    text = _normalize(str(message or ""))
    contrast = re.search(
        r"(?<!\w)(?:nie\s+(?:s|so|oproti)|neporovnavaj\b.{0,70}?\b(?:s|so|oproti))"
        r"\s+(linearn\w*|linear\w*|nulov\w*|null)\b"
        r".{0,70}?\b(?:ale|radsej)\s+(?:s|so|oproti)\s+"
        r"(linearn\w*|linear\w*|nulov\w*|null)\b",
        text,
    )
    if contrast:
        def label(word: str) -> str:
            return "linear" if word.startswith(("linearn", "linear")) else "null"

        excluded, requested = map(label, contrast.groups())
        if excluded != requested:
            return requested
    # Without a clear "not X, but Y" contrast, never silently include a
    # baseline that the user explicitly told us not to compare against.
    if re.search(r"(?<!\w)(?:neporovnavaj\b|nie\s+(?:s|so|oproti)\s+(?:linearn|linear|nulov|null))", text):
        return "ambiguous"
    linear = bool(re.search(r"(?<!\w)(?:linearn|linear)\w*", text))
    null = bool(re.search(r"(?<!\w)(?:nulov\w*|null\b)", text))
    if linear and null:
        return "both"
    if linear:
        return "linear"
    if null:
        return "null"
    return None


def _comparison_targets(message: str) -> tuple[str, ...]:
    selected = explicit_comparison_baseline(message)
    if selected in {"linear", "null"}:
        return (selected,)
    return ("linear", "null")


def maybe_historical_answer(message: str, reports_dir: str | Path | None) -> str | None:
    """Return a sourced answer, or None for non-historical intent / disabled input.

    Errors on supplied but corrupt reports must propagate, not become generated
    plausible-sounding statistics. No per-symbol results can be inferred from
    aggregated overall or per-market-date results.
    """
    if reports_dir is None:
        return None
    kind = historical_question_kind(message)
    if kind is None:
        return None
    directory = Path(reports_dir)
    splits = ("test",) if kind in {"test", "compare_test"} else (
        ("validation",) if kind in {"validation", "compare_validation"}
        else ("validation", "test")
    )
    reports = {split: _read_report(directory, split) for split in splits}
    available = {key: value for key, value in reports.items() if value is not None}
    if len(available) == 2:
        a, b = available.values()
        if a["dataset_id"] != b["dataset_id"] or a["protocol_sha256"] != b["protocol_sha256"]:
            raise ValueError("historical validation/test provenance mismatch")
    if not available:
        return (
            "Požadovaný historický report v3 v zadanom adresári nie je dostupný. "
            "Bez lokálneho reportu nemôžem uviesť metriky."
        )
    if kind == "causes":
        return (
            "Historické reporty v3 potvrdzujú nameranú MAE, ale neurčujú "
            "príčinu rozdielov medzi modelmi. Bez samostatnej analýzy "
            "chýb podľa volatility, likvidity a ďalších faktorov neviem "
            "vysvetliť, prečo bol model presnejší alebo menej presný. "
            "Nižšia historická chyba nezaručuje lepšie budúce predikcie."
        )
    if kind == "simple":
        parts = []
        for split, report in available.items():
            name = "historickej validácii" if split == "validation" else "historickom lockbox teste"
            model_mae = report["nonlinear_mae_bps"]
            linear_mae = report["linear_mae_bps"]
            if model_mae < linear_mae:
                verdict = "nižšia"
            elif model_mae > linear_mae:
                verdict = "vyššia"
            else:
                verdict = "rovnaká"
            parts.append(
                f"Na {name} dosiahol OurAI v3 MAE "
                f"{_fmt(model_mae)} bps; lineárny model {_fmt(linear_mae)} bps "
                f"(chyba OurAI bola {verdict})."
            )
        return (
            "Jednoducho: " + " ".join(parts) +
            " Hovoríme len o 15-minútovej absolútnej veľkosti pohybu, "
            "nie o jeho smere, budúcej úspešnosti alebo zisku."
        )
    if kind == "symbols":
        return (
            "Dostupné historické reporty v3 obsahujú agregované a denné MAE, "
            "nie spoľahlivé hodnotenie po jednotlivých akciách. "
            "Preto neviem určiť, pri ktorej akcii robí model najväčšie chyby. "
            "Na poradie podľa tickerov potrebujem samostatnú per-symbol diagnostiku."
        )
    if kind == "boundary":
        return (
            "Historické reporty v3 nie sú forward výsledky. "
            "Na hodnotenie predikcií zaznamenaných v reálnom čase je potrebný "
            "samostatný overený forward denník."
        )
    if kind.startswith("compare"):
        targets = _comparison_targets(message)
        # A named count/share of day-level MAE wins comes only from the
        # validated daily-win fields, never from direction hit rates.
        day_wins, share = _day_wins_metric_request(message)
        details = " ".join(
            (_day_wins_comparison(split, report, baseline, include_share=share)
             if day_wins else _baseline_comparison(split, report, baseline))
            for split, report in available.items() for baseline in targets
        )
    else:
        details = " ".join(_summary(split, report) for split, report in available.items())
    if len(available) != len(splits):
        missing = ", ".join(split for split in splits if split not in available)
        details += f" Report pre {missing} nie je v zadanom adresári dostupný."
    return (
        details + " MAE je priemerná absolútna chyba odhadu veľkosti pohybu "
        "(1 bps = 0,01 percentuálneho bodu); nižšia je lepšia. "
        "Toto nie je presnosť smeru UP/DOWN ani dôkaz ziskovosti či "
        "úspechu v nezávislom forward teste."
    )
