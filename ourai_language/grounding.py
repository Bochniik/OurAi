from __future__ import annotations

import re
from dataclasses import dataclass

from .contracts import AnalysisSnapshot


_PERCENT_RE = re.compile(r"(?<![\w.])(-?\d+(?:[\.,]\d+)?)\s*%")


@dataclass(frozen=True)
class GroundingViolation(ValueError):
    unsupported_percentages: tuple[float, ...]

    def __str__(self) -> str:
        values = ", ".join(f"{value:g}%" for value in self.unsupported_percentages)
        return f"reply contains unsupported forecast percentages: {values}"


def canonical_percentage_values(snapshot: AnalysisSnapshot) -> tuple[float, ...]:
    """Percent-like numerical values the model may quote from one snapshot."""

    values = [snapshot.p_up * 100.0, snapshot.p_down * 100.0]
    if snapshot.expected_move_pct is not None:
        values.append(snapshot.expected_move_pct)
    if snapshot.confidence is not None:
        values.append(snapshot.confidence * 100.0)
    return tuple(values)


def extract_percentages(text: str) -> tuple[float, ...]:
    values: list[float] = []
    for match in _PERCENT_RE.finditer(str(text)):
        values.append(float(match.group(1).replace(",", ".")))
    return tuple(values)


def find_unsupported_percentages(
    text: str,
    snapshot: AnalysisSnapshot,
    *,
    tolerance: float = 0.25,
) -> tuple[float, ...]:
    """Return percentages not grounded in the canonical forecast payload.

    The first guard intentionally targets the most dangerous hallucinations for Orion:
    probability/confidence/move percentages. Broader factual grounding will be added
    when live news and market-context sources are connected.
    """

    allowed = canonical_percentage_values(snapshot)
    unsupported: list[float] = []
    for value in extract_percentages(text):
        if not any(abs(value - expected) <= tolerance for expected in allowed):
            unsupported.append(value)
    return tuple(unsupported)


def assert_grounded_percentages(text: str, snapshot: AnalysisSnapshot) -> None:
    unsupported = find_unsupported_percentages(text, snapshot)
    if unsupported:
        raise GroundingViolation(unsupported)


def grounded_fallback(snapshot: AnalysisSnapshot) -> str:
    """Deterministic safe response used if generated forecast numbers drift."""

    fields = [
        f"{snapshot.ticker} {snapshot.horizon_minutes}m",
        f"P(UP)={snapshot.p_up * 100.0:.1f}%",
        f"P(DOWN)={snapshot.p_down * 100.0:.1f}%",
    ]
    if snapshot.expected_move_pct is not None:
        fields.append(f"expected move={snapshot.expected_move_pct:.2f}%")
    if snapshot.confidence is not None:
        fields.append(f"confidence={snapshot.confidence * 100.0:.1f}%")
    fields.extend(
        [
            f"uncertainty={snapshot.uncertainty}",
            f"decision={snapshot.decision}",
        ]
    )
    return " | ".join(fields)
