"""Fixed GRU direction challenger for v5 train-only research.

PyTorch is imported lazily so the normal lightweight CPU CI does not need to install
GPU dependencies. The caller provides already-frozen causal sequence refs and a
train-only scaler.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Iterable, Mapping, Sequence

import numpy as np

from .direction import DirectionPrediction
from .temporal_sequences import SequenceRef, SequenceScaler, TemporalSequenceCorpus


@dataclass(frozen=True)
class GRUDirectionConfig:
    horizon_minutes: int = 15
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.1
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 5
    batch_size: int = 256
    gradient_clip_norm: float = 1.0
    seed: int = 42

    def __post_init__(self) -> None:
        if self.horizon_minutes <= 0:
            raise ValueError("horizon_minutes must be positive")
        if self.hidden_size <= 0 or self.num_layers <= 0:
            raise ValueError("GRU hidden_size and num_layers must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("invalid optimizer configuration")
        if self.epochs <= 0 or self.batch_size <= 0:
            raise ValueError("epochs and batch_size must be positive")
        if self.gradient_clip_norm <= 0:
            raise ValueError("gradient_clip_norm must be positive")


@dataclass(frozen=True)
class GRUDirectionFitSummary:
    training_sequences: int
    input_features: int
    up_rate: float
    epoch_losses: tuple[float, ...]
    device: str


@dataclass
class GRUDirectionModel:
    config: GRUDirectionConfig
    scaler: SequenceScaler
    corpus: TemporalSequenceCorpus
    torch_model: object
    device: str


def _label_for_ref(ref: SequenceRef, horizon_minutes: int) -> int | None:
    target = ref.sample.targets.get(f"{int(horizon_minutes)}m")
    if not isinstance(target, Mapping):
        return None
    direction = str(target.get("direction") or "").strip().upper()
    if direction == "UP":
        return 1
    if direction == "DOWN":
        return 0
    return None


def _torch():
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, Dataset
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is required for the v5 GRU experiment. Install/use the local GPU environment first."
        ) from exc
    return torch, nn, DataLoader, Dataset


def fit_gru_direction(
    corpus: TemporalSequenceCorpus,
    refs: Sequence[SequenceRef],
    *,
    scaler: SequenceScaler,
    config: GRUDirectionConfig | None = None,
    device: str | None = None,
    progress_prefix: str | None = None,
) -> tuple[GRUDirectionModel, GRUDirectionFitSummary]:
    config = config or GRUDirectionConfig()
    usable = [(ref, _label_for_ref(ref, config.horizon_minutes)) for ref in refs]
    usable = [(ref, label) for ref, label in usable if label is not None]
    if len(usable) < 100:
        raise ValueError(f"not enough directional GRU sequences: {len(usable)} < 100")
    labels = np.asarray([int(label) for _, label in usable], dtype=np.float32)
    if len(np.unique(labels)) < 2:
        raise ValueError("GRU direction training requires both UP and DOWN labels")

    torch, nn, DataLoader, Dataset = _torch()
    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if resolved_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")

    class _SequenceDataset(Dataset):
        def __len__(self):
            return len(usable)

        def __getitem__(self, index):
            ref, label = usable[index]
            matrix = corpus.transform_window(ref, scaler)
            return torch.from_numpy(matrix), torch.tensor(float(label), dtype=torch.float32)

    class _GRU(nn.Module):
        def __init__(self):
            super().__init__()
            self.gru = nn.GRU(
                input_size=corpus.model_feature_count,
                hidden_size=config.hidden_size,
                num_layers=config.num_layers,
                dropout=config.dropout if config.num_layers > 1 else 0.0,
                batch_first=True,
            )
            self.head = nn.Linear(config.hidden_size, 1)

        def forward(self, x):
            output, _ = self.gru(x)
            return self.head(output[:, -1, :]).squeeze(-1)

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    model = _GRU().to(resolved_device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    loss_fn = nn.BCEWithLogitsLoss()
    generator = torch.Generator()
    generator.manual_seed(config.seed)
    loader = DataLoader(
        _SequenceDataset(),
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=resolved_device == "cuda",
        generator=generator,
    )

    epoch_losses: list[float] = []
    model.train()
    for epoch_index in range(config.epochs):
        total_loss = 0.0
        total_count = 0
        for x, y in loader:
            x = x.to(resolved_device, non_blocking=True)
            y = y.to(resolved_device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
            optimizer.step()
            batch_count = int(y.shape[0])
            total_loss += float(loss.detach().cpu()) * batch_count
            total_count += batch_count
        epoch_loss = total_loss / max(1, total_count)
        if not isfinite(epoch_loss):
            raise ValueError("non-finite GRU training loss")
        epoch_losses.append(epoch_loss)
        if progress_prefix:
            print(
                f"{progress_prefix} epoch {epoch_index + 1}/{config.epochs} "
                f"loss={epoch_loss:.6f}"
            )

    wrapped = GRUDirectionModel(
        config=config,
        scaler=scaler,
        corpus=corpus,
        torch_model=model,
        device=resolved_device,
    )
    summary = GRUDirectionFitSummary(
        training_sequences=len(usable),
        input_features=corpus.model_feature_count,
        up_rate=float(np.mean(labels)),
        epoch_losses=tuple(epoch_losses),
        device=resolved_device,
    )
    return wrapped, summary


def forecast_gru_directions(
    model: GRUDirectionModel,
    refs: Iterable[SequenceRef],
    *,
    include_actual: bool = True,
) -> list[DirectionPrediction]:
    torch, _, _, _ = _torch()
    usable: list[tuple[SequenceRef, int | None]] = []
    for ref in refs:
        actual = _label_for_ref(ref, model.config.horizon_minutes)
        if include_actual and actual is None:
            continue
        usable.append((ref, actual))

    model.torch_model.eval()
    predictions: list[DirectionPrediction] = []
    batch_size = max(1, model.config.batch_size * 2)
    with torch.no_grad():
        for start in range(0, len(usable), batch_size):
            chunk = usable[start : start + batch_size]
            matrix = np.stack(
                [model.corpus.transform_window(ref, model.scaler) for ref, _ in chunk]
            )
            x = torch.from_numpy(matrix).to(model.device)
            probabilities = torch.sigmoid(model.torch_model(x)).detach().cpu().numpy()
            for (ref, actual), p_up in zip(chunk, probabilities):
                predictions.append(
                    DirectionPrediction(
                        symbol=ref.sample.symbol,
                        timestamp=ref.sample.timestamp,
                        horizon_minutes=model.config.horizon_minutes,
                        p_up=float(p_up),
                        actual_direction=actual if include_actual else None,
                    )
                )
    return predictions
