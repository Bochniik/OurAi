"""Core building blocks for the trading-focused OurAI project."""

from .dataset import (
    DEFAULT_HORIZONS_MINUTES,
    MarketBar,
    TradingSample,
    build_causal_samples,
    chronological_session_split,
    dataset_manifest,
)
from .ingestion import (
    CSVIngestionConfig,
    load_market_bars_csv,
    prepare_trading_dataset,
    sha256_file,
    write_samples_jsonl,
)

__all__ = [
    "DEFAULT_HORIZONS_MINUTES",
    "MarketBar",
    "TradingSample",
    "build_causal_samples",
    "chronological_session_split",
    "dataset_manifest",
    "CSVIngestionConfig",
    "load_market_bars_csv",
    "prepare_trading_dataset",
    "sha256_file",
    "write_samples_jsonl",
]
