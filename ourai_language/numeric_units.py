"""Deterministic conversion of explicitly requested basis-point units.

Only bare unit conversions qualify, not performance assessments or claimed
OurAI results. No journal, trading data or language model is consulted.
"""
from __future__ import annotations

from decimal import Decimal, InvalidOperation
import re
import unicodedata


def _plain(value: str) -> str:
    folded = unicodedata.normalize("NFKD", value.casefold())
    return "".join(ch for ch in folded if not unicodedata.combining(ch))


def explicit_bps_conversion_answer(question: str) -> str | None:
    """Convert bps only when the entire Slovak question explicitly asks for it."""
    if not isinstance(question, str) or len(question) > 140:
        return None
    text = " ".join(_plain(question).strip().split()).rstrip("?.! ")
    number = r"([0-9]{1,5}(?:[,.][0-9]{1,4})?)"
    templates = (
        rf"kolko (?:percent|percentualnych bodov) (?:je|predstavuje) {number} (?:bps|bazickych bodov)",
        rf"preved {number} (?:bps|bazickych bodov) na (?:percenta|percentualne body)",
    )
    match = next((m for pattern in templates if (m := re.fullmatch(pattern, text))), None)
    if match is None:
        return None
    try:
        value = Decimal(match.group(1).replace(",", "."))
    except InvalidOperation:
        return None
    # Conversion is between units of percentage points, not an observed model metric.
    converted = value / Decimal(100)
    readable = lambda item: format(item.normalize(), "f").replace(".", ",")
    return (
        f"{readable(value)} bps = {readable(converted)} percentuálneho bodu. "
        "Jeden bázický bod je 0,01 percentuálneho bodu. "
        "Tento prevod sám osebe nevypovedá o úspešnosti predikcií ani o zisku."
    )
