"""Build a local OurAI training dataset from Alpaca historical 1-minute bars."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from trading.historical_market import (
    DEFAULT_SYMBOLS,
    HistoricalMarketConfig,
    fetch_and_build_alpaca_dataset,
)


def _parse_datetime(value: str) -> datetime:
    parsed = datetime.fromisoformat(str(value).strip().replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise argparse.ArgumentTypeError("datetime must include an explicit timezone or Z")
    return parsed


def _csv_values(value: str) -> tuple[str, ...]:
    values = tuple(item.strip() for item in str(value).split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("at least one comma-separated value is required")
    return values


def _integer_values(value: str) -> tuple[int, ...]:
    try:
        values = tuple(int(item.strip()) for item in str(value).split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("horizons must be comma-separated integers") from exc
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("horizons must contain positive integers")
    return values


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description=(
            "Fetch Alpaca 1-minute OHLCV, cache a canonical local source file and "
            "build leakage-safe train/validation/test JSONL splits with explicit "
            "bar-availability timestamps. Credentials are read only from local "
            "ALPACA_API_KEY_ID / ALPACA_API_SECRET_KEY environment variables."
        )
    )
    result.add_argument("--start", required=True, type=_parse_datetime)
    result.add_argument("--end", required=True, type=_parse_datetime)
    result.add_argument(
        "--symbols",
        type=_csv_values,
        default=DEFAULT_SYMBOLS,
        help="Comma-separated symbols (default: NVDA,AAPL,TSLA,JPM,AMD,META,MSFT)",
    )
    result.add_argument("--feed", choices=("iex", "sip", "boats", "otc"), default="iex")
    result.add_argument(
        "--sessions",
        type=_csv_values,
        default=("regular",),
        help="Comma-separated premarket,regular,afterhours (default: regular)",
    )
    result.add_argument("--adjustment", choices=("raw", "split", "dividend", "all"), default="raw")
    result.add_argument("--horizons", type=_integer_values, default=(5, 15, 30, 60))
    result.add_argument("--train-fraction", type=float, default=0.70)
    result.add_argument("--validation-fraction", type=float, default=0.15)
    result.add_argument("--minimum-absolute-return-bps", type=float, default=3.0)
    result.add_argument("--spread-multiplier", type=float, default=1.5)
    result.add_argument("--batch-size", type=int, default=20)
    result.add_argument("--max-pages-per-batch", type=int, default=100)
    result.add_argument("--allow-partial", action="store_true")
    result.add_argument(
        "--output-dir",
        default="data/processed/market/ourai-historical-v2",
        help="Local output directory. data/processed is gitignored by default.",
    )
    return result


def main() -> int:
    args = parser().parse_args()
    config = HistoricalMarketConfig(
        symbols=tuple(args.symbols),
        start=args.start,
        end=args.end,
        feed=args.feed,
        adjustment=args.adjustment,
        sessions=tuple(args.sessions),
        batch_size=args.batch_size,
        max_pages_per_batch=args.max_pages_per_batch,
        allow_partial=args.allow_partial,
    )
    manifest = fetch_and_build_alpaca_dataset(
        Path(args.output_dir),
        market_config=config,
        horizons_minutes=tuple(args.horizons),
        train_fraction=args.train_fraction,
        validation_fraction=args.validation_fraction,
        minimum_absolute_return_bps=args.minimum_absolute_return_bps,
        spread_multiplier=args.spread_multiplier,
    )
    summary = {
        "builder": manifest["builder"],
        "dataset_id": manifest["dataset_id"],
        "output_dir": str(Path(args.output_dir).resolve()),
        "source_rows": manifest["source_cache"]["row_count"],
        "sample_count": manifest["dataset"]["sample_count"],
        "market_date_count": manifest["dataset"]["market_date_count"],
        "symbols": manifest["dataset"]["symbols"],
        "splits": manifest["dataset"].get("splits"),
        "source_partial": manifest["source"].get("partial"),
        "bar_availability_delay_minutes": manifest["configuration"].get(
            "bar_availability_delay_minutes"
        ),
        "manifest_sha256": manifest["manifest_sha256"],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
