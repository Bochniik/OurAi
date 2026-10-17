"""Grounded conversational layer for OurAI.

The package deliberately keeps the language model separate from the trading forecast
engine. Forecast numbers enter through :class:`AnalysisSnapshot`; the conversational
runtime may explain them, but it must not invent replacements.
"""

from .contracts import AnalysisSnapshot
from .grounding import (
    GroundingViolation,
    canonical_percentage_values,
    find_unsupported_percentages,
    grounded_fallback,
)
from .prompting import SYSTEM_PROMPT, build_messages

__all__ = [
    "AnalysisSnapshot",
    "GroundingViolation",
    "SYSTEM_PROMPT",
    "build_messages",
    "canonical_percentage_values",
    "find_unsupported_percentages",
    "grounded_fallback",
]
