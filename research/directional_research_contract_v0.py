"""Standalone, unvalidated research-only directional forecast *shape* contract.

No data acquisition, model execution, forward journal access, Orion import,
network request, or trading permission. This is NOT a verified evidence reader.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from math import isclose, isfinite
import re


FROZEN_RESEARCH_SYMBOLS = frozenset(("NVDA", "AAPL", "TSLA", "JPM", "AMD", "META", "MSFT"))
CONTRACT_NAME = "ourai_directional_research_v0"
REQUIRED_FIELDS = frozenset((
    "contract", "schema_version", "mode", "validation_status",
    "evidence_status", "research_only", "trading_enabled",
    "experiment_id", "symbol", "horizon_minutes", "as_of_utc",
    "target_due_at_utc", "probability_conditioning", "p_up", "p_down",
    "expected_signed_change_pct",
))


def _utc_timestamp(value: object) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ValueError("research timestamp must be explicit UTC (Z)")
    try:
        result = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise ValueError("invalid research timestamp") from exc
    if result.utcoffset() != timedelta(0):
        raise ValueError("research timestamp must be UTC")
    return result


def _finite_number(value: object, *, probability: bool = False) -> float:
    if type(value) not in (int, float):
        raise ValueError("research number must be numeric, not bool/string")
    try:
        number = float(value)
    except (ValueError, OverflowError) as exc:
        raise ValueError("invalid research number") from exc
    if not isfinite(number) or (probability and not 0 <= number <= 1):
        raise ValueError("invalid research probability or nonfinite number")
    return number


def validate_directional_research_v0(payload: object) -> dict:
    """Validate only the shape and fail-closed safety labels, NOT accuracy.

    P(UP) and P(DOWN) are conditional on a nonzero 15m return; flat
    observations are outside this binary event. The signed expected return
    is an independent research estimate, NOT a verified or tradable signal.
    """
    if type(payload) is not dict or set(payload) != REQUIRED_FIELDS:
        raise ValueError("research contract has missing or unexpected fields")
    constants = {
        "contract": CONTRACT_NAME,
        "schema_version": 0,
        "mode": "RESEARCH_ONLY",
        "validation_status": "NOT_INDEPENDENTLY_VALIDATED",
        "evidence_status": "UNVERIFIED_MODEL_OUTPUT",
        "research_only": True,
        "trading_enabled": False,
        "horizon_minutes": 15,
        "probability_conditioning": "up_vs_down_given_nonzero_15m_return",
    }
    for key, expected in constants.items():
        if type(payload[key]) is not type(expected) or payload[key] != expected:
            raise ValueError("research safety/semantics field mismatch: " + key)
    if payload["symbol"] not in FROZEN_RESEARCH_SYMBOLS:
        raise ValueError("unsupported research symbol")
    experiment_id = payload["experiment_id"]
    if not isinstance(experiment_id, str) or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9._-]{0,119}", experiment_id
    ):
        raise ValueError("invalid research experiment id")
    start = _utc_timestamp(payload["as_of_utc"])
    due = _utc_timestamp(payload["target_due_at_utc"])
    if due - start != timedelta(minutes=15):
        raise ValueError("research target must be exactly 15 minutes after as_of")
    p_up = _finite_number(payload["p_up"], probability=True)
    p_down = _finite_number(payload["p_down"], probability=True)
    if not isclose(p_up + p_down, 1.0, rel_tol=0, abs_tol=1e-6):
        raise ValueError("conditional directional probabilities must sum to one")
    signed = _finite_number(payload["expected_signed_change_pct"])
    return {**payload, "p_up": p_up, "p_down": p_down,
            "expected_signed_change_pct": signed}
