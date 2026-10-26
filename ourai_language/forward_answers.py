"""Deterministic Slovak answers from a verified local v3 forward-only journal.

This path deliberately bypasses a generative model: a narrow read-only bridge
must not hallucinate a direction, probability, profit or live price.
"""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path

from trading.forward_analytics import FROZEN_SYMBOLS, read_forward_report


def _normalize(text: str) -> str:
    folded = unicodedata.normalize("NFKD", text.casefold())
    return "".join(c for c in folded if not unicodedata.combining(c))


def _ticker_in(text: str) -> str | None:
    return next(
        (symbol for symbol in FROZEN_SYMBOLS
         if re.search(rf"(?<!\w){re.escape(symbol)}(?!\w)", text, re.IGNORECASE)),
        None,
    )


_DEFINITION_CLAUSES = (
    "co je ", "co znamena ", "ako funguje ", "ako vlastne funguje ",
    "na co sluzi ", "aky je rozdiel medzi ", "aky je vyznam ",
    "what is ", "what does ", "how does ", "what is the difference between ",
)


def is_forward_educational_request(message: str) -> bool:
    """Do not mistake explanations of concepts for observed forward evidence.

    Keep performance requests such as "vysvetli mi výsledky forward denníka"
    eligible: only an explicit definition clause is educational.
    """
    text = _normalize(str(message or "")).strip().strip('.?!„“"')
    text = re.sub(r"^prosim[, ]+", "", text)
    # Preserve the existing conservative education boundary for broad
    # "please explain" prompts; only explicit observed-result phrasing
    # such as "vysvetli mi výsledky" opts into the reader.
    if text.startswith(("vysvetli prosim ", "vysvetli prosim, ", "can you explain ")):
        return True
    if text.startswith(_DEFINITION_CLAUSES + (
        "define ", "vysvetli pojem ", "explain what ",
    )):
        return True
    preface = re.match(
        r"^(?:(?:mozes|vies|dokazes)(?: mi)?(?: prosim)? )?"
        r"(?:vysvetli|vysvetlit)(?: mi)?[, ]+",
        text,
    )
    return bool(
        preface and text[preface.end():].startswith(_DEFINITION_CLAUSES)
    )



def _colloquial_error_magnitude_request(text: str) -> bool:
    """Recognize bounded Slovak questions about observed error size.

    These expressions alone do not authorize a forward source: the caller
    requires an explicit journal/shadow cue, rather than assuming that
    unspecified or retrospective errors came from forward observations.
    """
    return any(re.search(pattern, text) for pattern in (
        r"\b(?:ako velmi|nakolko)\b.{0,120}\b(?:myl\w*|odchyl\w*)\b",
        r"\bako daleko\b.{0,120}\b(?:skutocnost\w*|realit\w*)\b",
        r"\bo kolko\b.{0,120}\b(?:lisil\w*|odchyl\w*)\b",
        r"\b(?:aka|ake|aku)\b.{0,120}\bodchylk\w*\b",
    ))


def _directional_accuracy_request(text: str) -> bool:
    """Detect explicit direction hit-rate requests, not magnitude MAE.

    A request for directional correctness cannot be answered by a v3 model
    that predicts only absolute movement magnitude. Avoid interpreting the
    word "presnost" as authorization to display magnitude MAE instead.
    """
    direction = re.search(
        r"\b(?:smerov\w*|smer\w*|rast\s*/\s*pokles|"
        r"rast\s+alebo\s+pokles|up\s*/\s*down)\b",
        text,
    )
    hit_rate = re.search(
        r"\b(?:presn\w*|uspesn\w*|spravn\w*|trafil\w*|"
        r"kolkokrat|ako\s+casto)\b",
        text,
    )
    return bool(direction and hit_rate)



def _ambiguous_accuracy_request(text: str) -> bool:
    """Recognize colloquial correctness questions without guessing a metric.

    A hit rate could refer to UP/DOWN direction or to a threshold on
    magnitude error. Neither can be inferred from the word "spravne".
    """
    return any(re.search(pattern, text) for pattern in (
        r"\\bako\\s+casto\\b.{0,120}\\b(?:trafil\\w*|spravn\\w*|uspesn\\w*)\\b",
        r"\\bkolkokrat\\b.{0,120}\\b(?:trafil\\w*|spravn\\w*|uspesn\\w*)\\b",
        r"\\b(?:aka|ake)\\s+(?:(?:je|bola|boli|su)\\s+)?uspesnost\\b",
        r"\\bako\\s+uspesn\\w*\\b.{0,120}\\b(?:predikc\\w*|odhad\\w*|forecast\\w*)\\b",
    ))


def forward_question_kind(message: str) -> str | None:
    """Require explicit forward/record/forecast intent; not everyday questions."""
    text = _normalize(str(message or "").strip())
    if not text or is_forward_educational_request(message):
        return None
    # An observed error magnitude cannot explain why a model erred.
    if re.match(r"^(?:preco|z akeho dovodu|co sposobilo)\b", text) and re.search(
        r"\b(?:myl\w*|odchyl\w*|chyb\w*)\b", text,
    ):
        return None
    # Never substitute magnitude MAE for a requested direction hit rate.
    # Without an explicit forward source, do not assume forward evidence.
    if _directional_accuracy_request(text):
        if any(word in text for word in (
            "historick", "lockbox", "validac", "backtest", "retrospekt",
        )):
            return None
        if any(word in text for word in (
            "forward", "shadow", "paper trading", "paper-trading",
            "dennik", "journal", "zaznam",
        )):
            return "direction_unavailable"
        return None
    # "How often were you right?" is not a request for magnitude MAE.
    # Do not select forward evidence without a named source, and do not
    # reinterpret retrospective success rates as forward observations.
    if _ambiguous_accuracy_request(text) and not any(word in text for word in (
        "mae", "bps", "odchyl", "chyb", "magnitude", "velkost", "absolutn",
    )):
        if any(word in text for word in (
            "historick", "lockbox", "validac", "backtest", "retrospekt",
        )):
            return None
        if any(word in text for word in (
            "forward", "shadow", "paper trading", "paper-trading",
            "dennik", "journal", "zaznam",
        )):
            return "accuracy_clarification"
        return None
    if _colloquial_error_magnitude_request(text):
        if any(word in text for word in (
            "historick", "lockbox", "validac", "backtest", "retrospekt",
        )):
            return None
        if not any(word in text for word in (
            "forward", "shadow", "paper trading", "paper-trading",
            "dennik", "journal", "zaznam",
        )):
            return None
        return "performance"
    ticker = _ticker_in(text)
    explicit = any(word in text for word in (
        "forward", "dennik", "journal", "zaznam", "predikc", "predpoved",
        "forecast", "tvoje odhad", "nas odhad", "vas odhad", "mae",
    ))
    tied = ticker is not None and any(word in text for word in (
        "pohyb", "ocakav", "odhad", "model", "presn", "chyb", "predik",
        "predpoved", "vysled", "magnitude",
    ))
    accuracy = any(word in text for word in (
        "ako presn", "ako presne", "najvacsie chyby", "najvacsich chyb",
        "presnost nas", "nase chyby", "najhorsie predikcie",
    ))
    if not (explicit or tied or accuracy):
        return None
    if any(word in text for word in ("chyb", "presn", "mae", "vysled", "porovnaj", "presnost")):
        return "performance"
    if ticker:
        return "ticker"
    if any(word in text for word in ("posledn", "aktualn", "najnov", "latest", "teraz")):
        return "latest"
    return "summary"


def _fmt(value: float | None, decimals: int = 2) -> str:
    return "nedostupné" if value is None else f"{value:.{decimals}f}".replace(".", ",")


def _metrics(row: dict) -> str:
    if not row.get("resolved", row.get("resolution_count", 0)):
        return "Zatiaľ nemáme žiadne vyhodnotené predikcie."
    return (
        f"MAE magnitude: champion {_fmt(row['champion_mae_bps'])} bps; "
        f"lineárny baseline {_fmt(row['linear_mae_bps'])} bps; "
        f"nulový baseline {_fmt(row['null_mae_bps'])} bps."
    )


def maybe_forward_answer(
    message: str,
    journal_path: str | Path | None,
) -> str | None:
    """Return verified facts for recognizable forward-journal questions only."""
    if journal_path is None:
        return None
    kind = forward_question_kind(message)
    if kind is None:
        return None
    # Verify first, and fail closed even when the requested ticker has no events.
    # Invalid files propagate as errors; never fall through to generative guessing.
    report = read_forward_report(journal_path)
    if not report["journal_exists"]:
        return (
            "Lokálny forward denník v2 zatiaľ neexistuje. Nemám žiadne "
            "skutočné forward predikcie ani vyhodnotené výsledky. "
            "Model v3 odhaduje len veľkosť 15-minútového pohybu, nie smer."
        )
    if not report["prediction_count"]:
        return (
            "Forward denník v2 existuje a jeho hashový reťazec je platný, "
            "ale zatiaľ obsahuje 0 predikcií. Presnosť preto nie je možné hodnotiť."
        )
    if kind == "direction_unavailable":
        return (
            "Smerovú úspešnosť forward predikcií neviem z tohto denníka "
            "vyhodnotiť: model v3 odhaduje absolútnu veľkosť 15-minútového "
            "pohybu, nie smer rastu alebo poklesu. MAE meria chybu veľkosti "
            "pohybu, nie podiel správnych smerových predpovedí. "
            "Neuvádzam preto žiadne percento smerovej úspešnosti."
        )
    if kind == "accuracy_clarification":
        return (
            "Otázka, ako často boli predikcie správne, je nejednoznačná. "
            "Myslíš úspešnosť odhadu smeru rastu/poklesu, alebo chybu "
            "odhadu absolútnej veľkosti 15-minútového pohybu (MAE)? "
            "Model v3 predikuje iba veľkosť pohybu; MAE nie je percento "
            "správnych smerových predpovedí. Bez upresnenia neuvádzam "
            "percento úspešnosti."
        )
    ticker = _ticker_in(message)
    if ticker:
        row = report["symbols"][ticker]
        if not row["predictions"]:
            return f"V overenom forward denníku zatiaľ nemám žiadnu predikciu pre {ticker}."
        last = report["latest_by_symbol"][ticker]
        if kind == "performance":
            return (
                f"{ticker}: {row['predictions']} predikcií, "
                f"{row['resolved']} vyhodnotených, {row['pending']} čakajúcich "
                f"(z nich {row['overdue_pending']} po termíne). "
                + _metrics(row)
                + " Ide o veľkosť pohybu, nie smer; historické testovanie "
                  "nepredstavuje forward potvrdenie úspešnosti."
            )
        return (
            f"Najnovšia zaznamenaná predikcia pre {ticker}: "
            f"{_fmt(last['expected_magnitude_bps'])} bps "
            f"({_fmt(last['expected_magnitude_bps'] / 100)} %) absolútnej "
            f"15-minútovej zmeny. Vytvorená: {last['predicted_at']} (UTC); "
            f"termín výsledku: {last['target_due_at']} (UTC). "
            f"Stav: {'vyhodnotená' if last['resolved'] else 'čaká na vyhodnotenie'}. "
            "Nejde o odhad smeru UP/DOWN ani obchodné odporúčanie; "
            "záznam nemusí predstavovať aktuálnu predikciu."
        )

    total = report["prediction_count"]
    resolved = report["resolution_count"]
    pending = report["pending_count"]
    if kind == "performance":
        per_symbol = sorted(
            (
                (sym, row["champion_mae_bps"], row["resolved"])
                for sym, row in report["symbols"].items() if row["resolved"]
            ),
            key=lambda entry: (-entry[1], entry[0]),
        )
        worst = (
            f" Najväčšia pozorovaná MAE: {per_symbol[0][0]} "
            f"{_fmt(per_symbol[0][1])} bps ({per_symbol[0][2]} vyhodnotených)."
            if per_symbol else ""
        )
        return (
            f"Forward v2: {total} predikcií, {resolved} vyhodnotených, "
            f"{pending} čakajúcich (z nich {report['overdue_pending_count']} "
            "po termíne). "
            + (
                f"Celková {_metrics(report)}{worst}"
                if resolved else "Zatiaľ nie sú žiadne vyhodnotené výsledky."
            )
            + " Sú to opisné metriky 15-minútovej absolútnej zmeny; "
              "neznamenajú ziskovosť ani potvrdené splnenie forward protokolu."
        )
    latest = report["latest_prediction"]
    base = (
        f"Overený forward denník v2: {total} predikcií, {resolved} vyhodnotených, "
        f"{pending} čakajúcich (z nich {report['overdue_pending_count']} po termíne). "
    )
    if latest and kind == "latest":
        return base + (
            f"Posledný záznam: {latest['symbol']} {_fmt(latest['expected_magnitude_bps'])} bps "
            f"({_fmt(latest['expected_magnitude_bps']/100)} %) očakávanej "
            f"absolútnej 15-minútovej zmeny, čas {latest['predicted_at']} UTC. "
            "Je to historicky najnovší zápis v denníku, nie záruka čerstvých "
            "live dát ani predikcia smeru."
        )
    return (
        base + _metrics(report) + " Model v3 odhaduje veľkosť pohybu, "
        "nie rast/pokles; ide len o overený miestny denník."
    )
