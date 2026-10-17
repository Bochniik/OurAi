"""Outcome-independent 15-minute anchor selection for v6 direction research.

The anchor uses only the observation timestamp, never the target direction or
future return. Adjacent accepted *label intervals* on the same instrument
and session have disjoint interiors, though other instruments may correlate.
"""
from __future__ import annotations

from datetime import timezone
from typing import Iterable

from trading.temporal_sequences import SequenceRef


ANCHOR_MODULUS_MINUTES = 15
ANCHOR_PHASE = 0


def anchored_training_refs(
    refs: Iterable[SequenceRef],
    *,
    modulus_minutes: int = ANCHOR_MODULUS_MINUTES,
    phase: int = ANCHOR_PHASE,
) -> list[SequenceRef]:
    if modulus_minutes <= 0 or not 0 <= phase < modulus_minutes:
        raise ValueError("invalid fixed time anchor")
    selected: list[SequenceRef] = []
    seen: set[tuple[str, object]] = set()
    for ref in refs:
        timestamp = ref.sample.timestamp
        if timestamp.tzinfo is None:
            raise ValueError("observation timestamp must have timezone")
        minute = int(timestamp.astimezone(timezone.utc).timestamp() // 60)
        if minute % modulus_minutes != phase:
            continue
        key = (ref.sample.symbol, timestamp)
        if key in seen:
            raise ValueError("duplicate symbol-timestamp anchor")
        seen.add(key)
        selected.append(ref)
    selected.sort(key=lambda ref: (ref.sample.symbol, ref.sample.timestamp))
    previous: dict[tuple[str, str, str], int] = {}
    for ref in selected:
        sample = ref.sample
        key = (sample.symbol, sample.session_date, sample.session)
        current = int(sample.timestamp.astimezone(timezone.utc).timestamp() // 60)
        last = previous.get(key)
        if last is not None and current - last < modulus_minutes:
            raise AssertionError("anchor intervals overlap within a symbol/session")
        previous[key] = current
    return selected
