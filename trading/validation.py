"""Chronological purged walk-forward validation for OurAI trading experiments.

The goal is to make it difficult to accidentally train on labels whose outcome
period overlaps a later validation window. Folds are built from complete New York
market dates, but purging is performed from each sample's actual target timestamps,
so the logic remains safe if longer-horizon or cross-session targets are added later.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Mapping, Optional, Sequence

from .dataset import TradingSample


@dataclass(frozen=True)
class WalkForwardFold:
    """One deterministic development fold.

    ``train`` is always strictly earlier than ``validation``. Samples close to the
    validation boundary may be removed by the embargo, and samples whose target
    interval reaches the validation window are removed by purging.
    """

    index: int
    train: tuple[TradingSample, ...]
    validation: tuple[TradingSample, ...]
    train_market_dates: tuple[str, ...]
    validation_market_dates: tuple[str, ...]
    validation_start: datetime
    validation_end: datetime
    purged_overlap_count: int
    embargoed_count: int

    def manifest(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "train_samples": len(self.train),
            "validation_samples": len(self.validation),
            "train_market_dates": list(self.train_market_dates),
            "validation_market_dates": list(self.validation_market_dates),
            "validation_start": self.validation_start.isoformat(),
            "validation_end": self.validation_end.isoformat(),
            "purged_overlap_count": self.purged_overlap_count,
            "embargoed_count": self.embargoed_count,
        }


def _parse_target_timestamp(value: object) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        timestamp = value
    elif isinstance(value, str):
        try:
            timestamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError(f"invalid target future_timestamp: {value!r}") from exc
    else:
        raise TypeError("target future_timestamp must be datetime, ISO string, or None")

    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ValueError("target future_timestamp must be timezone-aware")
    return timestamp.astimezone(timezone.utc)


def sample_label_end(sample: TradingSample) -> datetime:
    """Return the latest resolved target timestamp for one sample.

    If a hand-built sample has no explicit future timestamps, its label interval
    conservatively ends at the feature timestamp.
    """

    if sample.timestamp.tzinfo is None or sample.timestamp.utcoffset() is None:
        raise ValueError("sample timestamp must be timezone-aware")

    latest = sample.timestamp.astimezone(timezone.utc)
    for target in sample.targets.values():
        if not isinstance(target, Mapping):
            raise TypeError("sample targets must contain mappings")
        future = _parse_target_timestamp(target.get("future_timestamp"))
        if future is not None and future > latest:
            latest = future
    return latest


def purged_walk_forward_folds(
    samples: Sequence[TradingSample],
    *,
    min_train_market_dates: int = 20,
    validation_market_dates: int = 5,
    step_market_dates: int = 5,
    embargo_minutes: float = 60.0,
    max_train_market_dates: Optional[int] = None,
) -> list[WalkForwardFold]:
    """Build expanding or rolling chronological development folds.

    Parameters
    ----------
    samples:
        Development samples only. A final untouched test set should be reserved
        before calling this function.
    min_train_market_dates:
        Number of complete market dates required before the first validation block.
    validation_market_dates:
        Number of complete market dates in each validation block.
    step_market_dates:
        How many market dates to advance between folds.
    embargo_minutes:
        Additional gap immediately before validation. Training samples whose feature
        timestamp falls inside this gap are excluded even when their labels have
        already matured.
    max_train_market_dates:
        ``None`` creates expanding-window folds. A positive integer creates a rolling
        window containing at most this many most-recent market dates before each
        validation block.

    Purging uses the latest ``future_timestamp`` present in each training sample.
    Any training row whose label interval reaches or crosses ``validation_start`` is
    removed. This protects against overlap when longer-horizon targets are added.

    ``min_train_market_dates`` is checked before purging/embargo. Boundary controls
    are allowed to remove rows or even an entire boundary date; the fold remains
    valid as long as at least one training sample survives. The resulting actual
    train-date coverage is recorded in each fold manifest for auditability.
    """

    if min_train_market_dates < 1:
        raise ValueError("min_train_market_dates must be at least 1")
    if validation_market_dates < 1:
        raise ValueError("validation_market_dates must be at least 1")
    if step_market_dates < 1:
        raise ValueError("step_market_dates must be at least 1")
    if embargo_minutes < 0:
        raise ValueError("embargo_minutes must be non-negative")
    if max_train_market_dates is not None:
        if max_train_market_dates < min_train_market_dates:
            raise ValueError(
                "max_train_market_dates must be >= min_train_market_dates when supplied"
            )

    ordered_samples = sorted(
        samples,
        key=lambda sample: (sample.timestamp, sample.symbol, sample.session),
    )
    for sample in ordered_samples:
        if sample.timestamp.tzinfo is None or sample.timestamp.utcoffset() is None:
            raise ValueError("sample timestamp must be timezone-aware")

    market_dates = sorted({sample.session_date for sample in ordered_samples})
    required = min_train_market_dates + validation_market_dates
    if len(market_dates) < required:
        raise ValueError(
            f"at least {required} market dates are required, got {len(market_dates)}"
        )

    by_date: dict[str, list[TradingSample]] = {date: [] for date in market_dates}
    for sample in ordered_samples:
        by_date.setdefault(sample.session_date, []).append(sample)

    folds: list[WalkForwardFold] = []
    validation_start_index = min_train_market_dates
    fold_index = 0
    embargo = timedelta(minutes=float(embargo_minutes))

    while validation_start_index + validation_market_dates <= len(market_dates):
        validation_dates = market_dates[
            validation_start_index : validation_start_index + validation_market_dates
        ]
        candidate_train_dates = market_dates[:validation_start_index]
        if max_train_market_dates is not None:
            candidate_train_dates = candidate_train_dates[-max_train_market_dates:]

        validation_rows = [
            sample for date in validation_dates for sample in by_date.get(date, ())
        ]
        if not validation_rows:
            validation_start_index += step_market_dates
            continue

        validation_start = min(sample.timestamp for sample in validation_rows).astimezone(
            timezone.utc
        )
        validation_end = max(
            max(sample.timestamp.astimezone(timezone.utc), sample_label_end(sample))
            for sample in validation_rows
        )
        embargo_boundary = validation_start - embargo

        train_rows: list[TradingSample] = []
        purged_overlap_count = 0
        embargoed_count = 0
        for date in candidate_train_dates:
            for sample in by_date.get(date, ()):
                sample_time = sample.timestamp.astimezone(timezone.utc)
                if sample_label_end(sample) >= validation_start:
                    purged_overlap_count += 1
                    continue
                if embargo_minutes > 0 and sample_time >= embargo_boundary:
                    embargoed_count += 1
                    continue
                train_rows.append(sample)

        if not train_rows:
            raise ValueError(
                "purge/embargo removed all training samples; reduce the embargo or "
                "increase the amount of history"
            )

        actual_train_dates = tuple(sorted({sample.session_date for sample in train_rows}))
        latest_train_label = max(sample_label_end(sample) for sample in train_rows)
        if latest_train_label >= validation_start:
            raise AssertionError("purged fold still contains overlapping training labels")
        if max(sample.timestamp for sample in train_rows) >= validation_start:
            raise AssertionError("walk-forward training is not strictly chronological")

        folds.append(
            WalkForwardFold(
                index=fold_index,
                train=tuple(train_rows),
                validation=tuple(validation_rows),
                train_market_dates=actual_train_dates,
                validation_market_dates=tuple(validation_dates),
                validation_start=validation_start,
                validation_end=validation_end,
                purged_overlap_count=purged_overlap_count,
                embargoed_count=embargoed_count,
            )
        )
        fold_index += 1
        validation_start_index += step_market_dates

    if not folds:
        raise ValueError("configuration produced no walk-forward folds")
    return folds


def walk_forward_manifest(folds: Sequence[WalkForwardFold]) -> dict[str, Any]:
    """Return deterministic fold metadata suitable for experiment lineage."""

    return {
        "fold_count": len(folds),
        "folds": [fold.manifest() for fold in folds],
        "total_train_samples_across_folds": sum(len(fold.train) for fold in folds),
        "total_validation_samples_across_folds": sum(
            len(fold.validation) for fold in folds
        ),
        "purged_overlap_count": sum(fold.purged_overlap_count for fold in folds),
        "embargoed_count": sum(fold.embargoed_count for fold in folds),
        "policy": "strictly_chronological_purged_walk_forward",
    }
