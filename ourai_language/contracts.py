from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite
from typing import Any, Mapping


_ALLOWED_DECISIONS = {"LONG_BIAS", "SHORT_BIAS", "NO_TRADE", "UNAVAILABLE"}
_ALLOWED_UNCERTAINTY = {"low", "medium", "high", "unavailable"}


def _finite_probability(value: Any, name: str) -> float:
    numeric = float(value)
    if not isfinite(numeric) or not 0.0 <= numeric <= 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1]")
    return numeric


def _optional_finite_nonnegative(value: Any, name: str) -> float | None:
    if value is None:
        return None
    numeric = float(value)
    if not isfinite(numeric) or numeric < 0.0:
        raise ValueError(f"{name} must be finite and non-negative when supplied")
    return numeric


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _text_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        text = value.strip()
        return (text,) if text else ()
    if not isinstance(value, (list, tuple)):
        raise ValueError("reasons must be a string, list, tuple, or null")
    return tuple(text for item in value if (text := str(item).strip()))


@dataclass(frozen=True)
class AnalysisSnapshot:
    """Canonical forecast payload supplied to the conversational model.

    The language model is allowed to explain these values, not replace them. This
    object is intentionally independent from any particular forecast-model class so
    Orion can keep one stable interface while the prediction engine evolves.
    """

    ticker: str
    horizon_minutes: int
    p_up: float
    p_down: float
    expected_move_pct: float | None
    uncertainty: str
    decision: str
    confidence: float | None = None
    market_regime: str | None = None
    relative_strength: str | None = None
    volume_state: str | None = None
    news_context: str | None = None
    reasons_for: tuple[str, ...] = ()
    reasons_against: tuple[str, ...] = ()
    model_version: str | None = None
    as_of: str | None = None

    def __post_init__(self) -> None:
        ticker = str(self.ticker or "").strip().upper()
        if not ticker:
            raise ValueError("ticker is required")
        object.__setattr__(self, "ticker", ticker)

        horizon = int(self.horizon_minutes)
        if horizon <= 0:
            raise ValueError("horizon_minutes must be positive")
        object.__setattr__(self, "horizon_minutes", horizon)

        p_up = _finite_probability(self.p_up, "p_up")
        p_down = _finite_probability(self.p_down, "p_down")
        if abs((p_up + p_down) - 1.0) > 1e-6:
            raise ValueError("p_up + p_down must equal 1")
        object.__setattr__(self, "p_up", p_up)
        object.__setattr__(self, "p_down", p_down)

        object.__setattr__(
            self,
            "expected_move_pct",
            _optional_finite_nonnegative(self.expected_move_pct, "expected_move_pct"),
        )
        if self.confidence is not None:
            object.__setattr__(
                self,
                "confidence",
                _finite_probability(self.confidence, "confidence"),
            )

        uncertainty = str(self.uncertainty or "").strip().lower()
        if uncertainty not in _ALLOWED_UNCERTAINTY:
            raise ValueError(
                "uncertainty must be one of: " + ", ".join(sorted(_ALLOWED_UNCERTAINTY))
            )
        object.__setattr__(self, "uncertainty", uncertainty)

        decision = str(self.decision or "").strip().upper()
        if decision not in _ALLOWED_DECISIONS:
            raise ValueError(
                "decision must be one of: " + ", ".join(sorted(_ALLOWED_DECISIONS))
            )
        object.__setattr__(self, "decision", decision)

        for name in (
            "market_regime",
            "relative_strength",
            "volume_state",
            "news_context",
            "model_version",
            "as_of",
        ):
            object.__setattr__(self, name, _optional_text(getattr(self, name)))
        object.__setattr__(self, "reasons_for", _text_tuple(self.reasons_for))
        object.__setattr__(self, "reasons_against", _text_tuple(self.reasons_against))

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "AnalysisSnapshot":
        if not isinstance(payload, Mapping):
            raise TypeError("analysis payload must be a mapping")
        return cls(
            ticker=payload.get("ticker", ""),
            horizon_minutes=payload.get("horizon_minutes", payload.get("horizon", 0)),
            p_up=payload.get("p_up"),
            p_down=payload.get("p_down"),
            expected_move_pct=payload.get("expected_move_pct"),
            uncertainty=payload.get("uncertainty", "unavailable"),
            decision=payload.get("decision", "UNAVAILABLE"),
            confidence=payload.get("confidence"),
            market_regime=payload.get("market_regime"),
            relative_strength=payload.get("relative_strength"),
            volume_state=payload.get("volume_state"),
            news_context=payload.get("news_context"),
            reasons_for=payload.get("reasons_for"),
            reasons_against=payload.get("reasons_against"),
            model_version=payload.get("model_version"),
            as_of=payload.get("as_of"),
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["reasons_for"] = list(self.reasons_for)
        payload["reasons_against"] = list(self.reasons_against)
        return payload
