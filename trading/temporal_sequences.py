"""Memory-efficient causal sequence assembly for v5 direction research.

The module converts target-stock samples plus context-only ETF samples into fixed
length feature sequences. It never reads targets from context instruments. Every
context join is exact on the already-available sample timestamp and every target
sequence is required to be exactly one-minute contiguous inside one symbol/session/day.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from math import isfinite
from typing import Iterable, Mapping, Sequence

import numpy as np

from .dataset import TradingSample


TARGET_FEATURES = (
    "return_1m",
    "return_5m",
    "return_15m",
    "return_30m",
    "distance_from_vwap",
    "relative_volume_20",
    "range_position_20",
    "realized_volatility_20",
    "spread_bps",
    "session_progress",
)
CONTEXT_FEATURES = (
    "return_1m",
    "return_5m",
    "return_15m",
    "return_30m",
    "distance_from_vwap",
    "relative_volume_20",
    "realized_volatility_20",
)
CONTEXT_ROLES = ("SPY", "QQQ", "SECTOR_ETF")
SECTOR_MAP = {
    "NVDA": "XLK",
    "AAPL": "XLK",
    "AMD": "XLK",
    "MSFT": "XLK",
    "META": "XLC",
    "TSLA": "XLY",
    "JPM": "XLF",
}


SampleKey = tuple[str, object]
GroupKey = tuple[str, str, str]


def sample_key(sample: TradingSample) -> SampleKey:
    return sample.symbol, sample.timestamp


def _finite_feature(sample: TradingSample | None, name: str) -> float:
    if sample is None:
        return float("nan")
    value = sample.features.get(name)
    if value is None or isinstance(value, bool):
        return float("nan")
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return numeric if isfinite(numeric) else float("nan")


@dataclass(frozen=True)
class SequenceRef:
    group_key: GroupKey
    end_index: int
    sample: TradingSample


@dataclass(frozen=True)
class SequenceScaler:
    means: np.ndarray
    scales: np.ndarray

    def __post_init__(self) -> None:
        if self.means.ndim != 1 or self.scales.ndim != 1:
            raise ValueError("sequence scaler must be one-dimensional")
        if self.means.shape != self.scales.shape:
            raise ValueError("sequence scaler mean/scale shape mismatch")
        if np.any(~np.isfinite(self.means)):
            raise ValueError("sequence scaler means must be finite")
        if np.any(~np.isfinite(self.scales)) or np.any(self.scales <= 0):
            raise ValueError("sequence scaler scales must be finite and positive")


@dataclass
class TemporalSequenceCorpus:
    sequence_length: int
    feature_names: tuple[str, ...]
    group_samples: dict[GroupKey, tuple[TradingSample, ...]]
    group_features: dict[GroupKey, np.ndarray]
    location_by_key: dict[SampleKey, tuple[GroupKey, int]]
    ref_by_key: dict[SampleKey, SequenceRef]

    @property
    def raw_feature_count(self) -> int:
        return len(self.feature_names)

    @property
    def model_feature_count(self) -> int:
        return self.raw_feature_count * 2

    def raw_row(self, key: SampleKey) -> np.ndarray:
        group_key, index = self.location_by_key[key]
        return self.group_features[group_key][index]

    def raw_window(self, ref: SequenceRef) -> np.ndarray:
        start = ref.end_index - self.sequence_length + 1
        return self.group_features[ref.group_key][start : ref.end_index + 1]

    def transform_window(self, ref: SequenceRef, scaler: SequenceScaler) -> np.ndarray:
        raw = self.raw_window(ref).astype(np.float32, copy=False)
        present = np.isfinite(raw)
        normalized = np.zeros_like(raw, dtype=np.float32)
        if np.any(present):
            means = scaler.means.astype(np.float32, copy=False)
            scales = scaler.scales.astype(np.float32, copy=False)
            normalized[present] = (
                (raw - means[np.newaxis, :]) / scales[np.newaxis, :]
            )[present]
        missing = (~present).astype(np.float32)
        return np.concatenate([normalized, missing], axis=1)


def _feature_names() -> tuple[str, ...]:
    names = [f"target__{name}" for name in TARGET_FEATURES]
    for role in CONTEXT_ROLES:
        names.extend(f"{role.lower()}__{name}" for name in CONTEXT_FEATURES)
    return tuple(names)


def build_temporal_sequence_corpus(
    target_samples: Iterable[TradingSample],
    context_samples: Iterable[TradingSample],
    *,
    sequence_length: int = 60,
    sector_map: Mapping[str, str] | None = None,
) -> TemporalSequenceCorpus:
    if sequence_length < 2:
        raise ValueError("sequence_length must be at least 2")
    sector_map = dict(SECTOR_MAP if sector_map is None else sector_map)

    targets = list(target_samples)
    contexts = list(context_samples)
    context_by_key = {sample_key(sample): sample for sample in contexts}

    grouped: dict[GroupKey, list[TradingSample]] = {}
    for sample in targets:
        grouped.setdefault((sample.symbol, sample.session_date, sample.session), []).append(sample)

    group_samples: dict[GroupKey, tuple[TradingSample, ...]] = {}
    group_features: dict[GroupKey, np.ndarray] = {}
    location_by_key: dict[SampleKey, tuple[GroupKey, int]] = {}
    ref_by_key: dict[SampleKey, SequenceRef] = {}
    names = _feature_names()

    for group_key, rows in grouped.items():
        ordered = tuple(sorted(rows, key=lambda sample: sample.timestamp))
        group_samples[group_key] = ordered
        matrix = np.full((len(ordered), len(names)), np.nan, dtype=np.float32)
        target_symbol = group_key[0]
        sector_symbol = sector_map.get(target_symbol)
        if not sector_symbol:
            raise ValueError(f"missing sector ETF mapping for {target_symbol}")

        for index, sample in enumerate(ordered):
            cursor = 0
            for name in TARGET_FEATURES:
                matrix[index, cursor] = _finite_feature(sample, name)
                cursor += 1
            for role_symbol in ("SPY", "QQQ", sector_symbol):
                context = context_by_key.get((role_symbol, sample.timestamp))
                for name in CONTEXT_FEATURES:
                    matrix[index, cursor] = _finite_feature(context, name)
                    cursor += 1
            key = sample_key(sample)
            location_by_key[key] = (group_key, index)

            if index + 1 < sequence_length:
                continue
            start = index - sequence_length + 1
            contiguous = True
            for position in range(start + 1, index + 1):
                if ordered[position].timestamp - ordered[position - 1].timestamp != timedelta(minutes=1):
                    contiguous = False
                    break
            if contiguous:
                ref_by_key[key] = SequenceRef(
                    group_key=group_key,
                    end_index=index,
                    sample=sample,
                )

        group_features[group_key] = matrix

    return TemporalSequenceCorpus(
        sequence_length=sequence_length,
        feature_names=names,
        group_samples=group_samples,
        group_features=group_features,
        location_by_key=location_by_key,
        ref_by_key=ref_by_key,
    )


def fit_sequence_scaler(
    corpus: TemporalSequenceCorpus,
    training_keys: Sequence[SampleKey],
) -> SequenceScaler:
    rows = [corpus.raw_row(key) for key in training_keys if key in corpus.location_by_key]
    if not rows:
        raise ValueError("no training feature rows available for sequence scaler")
    matrix = np.stack(rows).astype(np.float64, copy=False)
    means = np.zeros(matrix.shape[1], dtype=np.float64)
    scales = np.ones(matrix.shape[1], dtype=np.float64)
    for column in range(matrix.shape[1]):
        values = matrix[:, column]
        present = values[np.isfinite(values)]
        if present.size == 0:
            continue
        means[column] = float(np.mean(present))
        std = float(np.std(present))
        if isfinite(std) and std > 1e-12:
            scales[column] = std
    return SequenceScaler(means=means, scales=scales)
