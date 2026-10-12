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
from .evaluation import (
    LONG,
    NO_TRADE,
    SHORT,
    DecisionPolicyConfig,
    ForecastPrediction,
    TradeDecision,
    calibration_table,
    decide_forecast,
    evaluate_forecasts,
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
from .orion_adapter import (
    OrionAdapterConfig,
    OrionAdapterResult,
    normalize_orion_rows,
    orion_row_to_market_bar,
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
    "LONG",
    "NO_TRADE",
    "SHORT",
    "DecisionPolicyConfig",
    "ForecastPrediction",
    "TradeDecision",
    "calibration_table",
    "decide_forecast",
    "evaluate_forecasts",
    "ExperimentIdentity",
    "build_experiment_identity",
    "canonical_json",
    "sha256_metadata",
    "CSVIngestionConfig",
    "load_market_bars_csv",
    "prepare_trading_dataset",
    "sha256_file",
    "write_samples_jsonl",
    "OrionAdapterConfig",
    "OrionAdapterResult",
    "normalize_orion_rows",
    "orion_row_to_market_bar",
    "WalkForwardFold",
    "purged_walk_forward_folds",
    "sample_label_end",
    "walk_forward_manifest",
]
