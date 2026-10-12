"""Core building blocks for the trading-focused OurAI project."""

from .context import (
    ContextJoinResult,
    ContextObservation,
    ContextSourcePolicy,
    join_point_in_time_context,
)
from .dataset import (
    DEFAULT_HORIZONS_MINUTES,
    MarketBar,
    TradingSample,
    build_causal_samples,
    chronological_session_split,
    dataset_manifest,
)
from .experiment import (
    ExperimentIdentity,
    build_experiment_identity,
    canonical_json,
    sha256_metadata,
)
from .ingestion import (
    CSVIngestionConfig,
    load_market_bars_csv,
    prepare_trading_dataset,
    sha256_file,
    write_samples_jsonl,
)
from .validation import (
    WalkForwardFold,
    purged_walk_forward_folds,
    sample_label_end,
    walk_forward_manifest,
)

__all__ = [
    "ContextJoinResult",
    "ContextObservation",
    "ContextSourcePolicy",
    "join_point_in_time_context",
    "DEFAULT_HORIZONS_MINUTES",
    "MarketBar",
    "TradingSample",
    "build_causal_samples",
    "chronological_session_split",
    "dataset_manifest",
    "ExperimentIdentity",
    "build_experiment_identity",
    "canonical_json",
    "sha256_metadata",
    "CSVIngestionConfig",
    "load_market_bars_csv",
    "prepare_trading_dataset",
    "sha256_file",
    "write_samples_jsonl",
    "WalkForwardFold",
    "purged_walk_forward_folds",
    "sample_label_end",
    "walk_forward_manifest",
]
