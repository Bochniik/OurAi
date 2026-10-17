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
from .prompting import (
    ANALYSIS_QA,
    GENERAL_CHAT,
    ANALYSIS_SYSTEM_PROMPT,
    GENERAL_SYSTEM_PROMPT,
    build_messages,
    select_conversation_mode,
)

__all__ = [
    "ANALYSIS_QA",
    "GENERAL_CHAT",
    "ANALYSIS_SYSTEM_PROMPT",
    "GENERAL_SYSTEM_PROMPT",
    "AnalysisSnapshot",
    "GroundingViolation",
    "build_messages",
    "canonical_percentage_values",
    "find_unsupported_percentages",
    "grounded_fallback",
    "select_conversation_mode",
]
