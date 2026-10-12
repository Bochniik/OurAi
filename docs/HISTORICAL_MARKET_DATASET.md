# Historical market dataset v1

OurAI can build a local, reproducible training dataset from Alpaca historical stock bars without committing raw market data or credentials to Git.

## Contract

The v1 builder intentionally accepts only **1-minute** bars because the current causal feature layer uses exact 1-minute lookbacks. It normalizes provider rows into `MarketBar`, filters explicit market sessions, requires exact future timestamps for 5m/15m/30m/60m targets, then creates chronological whole-date train/validation/test splits.

Default research universe:

- NVDA
- AAPL
- TSLA
- JPM
- AMD
- META
- MSFT

The default feed is `iex`. IEX is a single-exchange feed and its volume must not be described as consolidated US-market volume. `sip` can be selected when the local Alpaca entitlement permits it; the manifest records the selected feed either way.

## Credentials

Credentials are read from the **local environment only** and are never written into the dataset manifest:

```powershell
$env:ALPACA_API_KEY_ID="..."
$env:ALPACA_API_SECRET_KEY="..."
```

Do not put credentials in source files, commits, workflow YAML, command-line arguments, or generated manifests. The public repository's self-hosted runner must remain manual-only for trusted workflows.

## Build

Example regular-hours dataset:

```powershell
python scripts/build_historical_market_dataset.py `
  --start 2026-01-01T00:00:00Z `
  --end 2026-10-01T00:00:00Z `
  --symbols NVDA,AAPL,TSLA,JPM,AMD,META,MSFT `
  --feed iex `
  --sessions regular `
  --horizons 5,15,30,60 `
  --output-dir data/processed/market/regular-v1
```

The output directory contains:

```text
source_bars.csv
train.jsonl
validation.jsonl
test.jsonl
manifest.json
```

`data/processed/` is already gitignored. The manifest records provider/feed/time range, requested symbols, pagination completeness, source SHA-256, output SHA-256 hashes, feature/target configuration, chronological split dates and a deterministic dataset ID.

## Fail-closed behavior

The builder refuses to proceed when:

- timestamps are naive,
- the requested interval is not 1 minute,
- no usable bars are returned,
- duplicate symbol/timestamps are observed in strict mode,
- pagination reaches its configured ceiling and would produce partial history,
- fewer than three market dates are available for chronological train/validation/test splitting.

Partial history can be explicitly enabled for research with `--allow-partial`, but it remains marked `partial: true` in lineage and should not be used to approve a production model.

## Next step

After a real dataset is built locally, `scripts/train_trading_baseline.py` can consume the generated split files. The first serious experiment should start with the 15-minute horizon and report held-out probability calibration, Brier/log loss, expected-return error, abstention coverage and cost-aware outcomes before any comparison with OrionTrader incumbents.
