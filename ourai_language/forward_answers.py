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


def forward_question_kind(message: str) -> str | None:
    """Route only user-grounded requests for OurAI's recorded predictions.

    A word such as "forward", "MAE" or "prediction" is not evidence that
    the user wants *our* forward journal. In particular, ordinary questions
    about how an indicator, metric or forward testing works must never be
    replaced with journal counts, even when a journal is configured.
    """
    text = _normalize(str(message or "").strip()).strip(" .?!„“\"")
    if not text:
        return None

    # Definitions and conceptual comparisons belong to educational answers.
    # This gate is intentionally before every other keyword-based route.
    if text.startswith((
        "co je ", "co su ", "co znamena ", "co oznacuje ",
        "what is ", "what are ", "what does ", "define ",
        "vysvetli pojem ", "vysvetli mi pojem ", "explain the term ",
        "ako funguje ", "ako funguju ", "how does ", "how do ",
        "aky je rozdiel medzi ", "aka je rozdiel medzi ",
        "what is the difference between ", "vysvetli rozdiel medzi ",
        "ako sa pocita ", "ako vypocitat ", "how to calculate ",
    )):
        return None

    ticker = _ticker_in(text)
    journal = any(word in text for word in (
        "forward", "dennik", "journal", "zaznam", "shadow",
    ))
    owner = bool(re.search(
        r"(?<!\\w)(?:tvoj\\w*|moj\\w*|nas\\w*|vas\\w*|"
        r"ourai|your|my|our|robis|sa ti|si na tom)(?!\\w)", text
    ))
    forecast = any(word in text for word in (
        "predikc", "predpoved", "forecast", "odhad",
    ))
    subject = forecast or any(word in text for word in (
        "pohyb", "ocakav", "model", "presn", "chyb", "vysled",
        "mae", "magnitude", "uspes", "vykonn",
    ))
    recent = any(word in text for word in (
        "posledn", "aktualn", "najnov", "latest", "recent",
    ))
    if not (journal or (owner and subject) or (ticker and subject)
            or (recent and forecast)):
        return None

    if any(word in text for word in (
        "chyb", "presn", "mae", "vysled", "porovnaj",
        "presnost", "uspes", "vykonn",
    )):
        return "performance"
    if ticker:
        return "ticker"
    if recent or any(word in text for word in ("teraz", "dnes")):
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
