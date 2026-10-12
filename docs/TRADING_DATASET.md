# Trading dataset foundation

This document defines the first data contract for the trading-focused OurAI project.

## Objective

One training sample represents what OurAI could have known at one real market timestamp, plus one or more outcomes that become known only later.

Conceptually:

```text
(symbol, timestamp)
        │
        ├── causal features known at timestamp
        │
        └── future outcomes at exact horizons
              ├── 5m
              ├── 15m
              ├── 30m
              └── 60m
```

The causal builder lives in `trading/dataset.py`. Raw CSV ingestion and deterministic split export live in `trading/ingestion.py`.

## Current feature set

The first pass intentionally uses only timestamped bar/quote information whose causality is easy to verify:

- price
- volume
- 1m / 5m / 15m / 30m returns
- session VWAP distance
- 20-observation relative volume
- 20-bar range position
- trailing realized volatility
- quoted spread in basis points when bid/ask are present
- session progress
- time-of-day context
- feature missingness fraction

All rolling values are built from the current bar and/or earlier bars only. No future row is used in any feature.

## Raw market CSV contract

`load_market_bars_csv()` accepts a provider-neutral CSV with these required columns:

```text
symbol,timestamp,open,high,low,close,volume
```

Optional columns are:

```text
bid,ask,session
```

Rules:

- timestamps should be timezone-aware ISO-8601 values;
- naive timestamps are rejected unless an explicit `source_timezone` is configured;
- OHLC values must be finite and positive;
- volume must be finite and non-negative;
- duplicate `(symbol, timestamp)` rows fail closed by default;
- rows are normalized into canonical `MarketBar` objects and sorted deterministically.

The ingestion manifest records the SHA-256 of the raw source file, row count, symbol coverage, time range, timezone policy and optional fields that were present.

## Deterministic dataset export

`prepare_trading_dataset()` performs the first complete reproducible path:

```text
raw CSV
  ↓
schema validation + SHA-256 provenance
  ↓
canonical MarketBar rows
  ↓
causal feature / future-target construction
  ↓
whole-market-date chronological split
  ↓
train.jsonl
validation.jsonl
test.jsonl
manifest.json
```

Each JSONL split receives its own SHA-256 in the manifest. Generated datasets belong under ignored local paths such as `data/processed/`; they must not be committed to Git.

## Targets

The initial target horizons are:

- 5 minutes
- 15 minutes
- 30 minutes
- 60 minutes

For each available horizon, the dataset stores:

- exact future timestamp
- future return as a decimal
- future return in percent
- `UP`, `DOWN`, or `NEUTRAL`
- the neutral/noise threshold used for that row

A future target is emitted only when a bar exists at the exact requested timestamp. A missing 5-minute bar cannot silently become a 6- or 7-minute target.

The default direction threshold is the larger of:

1. a configured absolute return floor, and
2. current quoted spread multiplied by a configurable factor.

This prevents tiny moves inside the estimated trading noise from being labelled as confident directional outcomes.

## Session boundaries

Bars are tagged as one of:

- `premarket`
- `regular`
- `afterhours`
- `offhours`

or may carry an explicit provider/session label.

By default a target may not cross from one named session into another. This keeps premarket and regular-hours evidence comparable to the existing OrionTrader incumbent engines. Cross-session targets can be explicitly enabled for a later preregistered experiment.

## Chronological splitting

`chronological_session_split()` splits by complete New York market dates, not random rows.

All symbols and all sessions from one market date stay in the same split. This prevents a model from training on one stock from a date while validating on another stock from that same date.

The initial split API exposes:

```text
train
validation
test
```

A purged/embargoed walk-forward layer will be added after point-in-time context joins are stable.

## External context is deliberately not joined yet

OrionTrader already has useful information from:

- news/catalyst analysis
- liquidity
- market-integrity anomaly scores
- market/sector regime
- session state

Those sources are valuable, but they must be joined with explicit `event_time` / `as_of` rules. The current foundation therefore does **not** silently forward-fill them into historical rows. A later point-in-time context joiner will reject context that was not known by the feature timestamp.

## Leakage and provenance invariants

The tests enforce several important rules:

- changing a future price changes the future target but not earlier features;
- missing exact future timestamps do not create fake horizons;
- targets do not cross market sessions unless explicitly enabled;
- whole market dates stay disjoint across train/validation/test;
- spread-aware neutral labels are deterministic;
- naive timestamps require an explicit timezone policy;
- duplicate timestamps fail closed;
- raw input and generated split hashes are reproducible.

These invariants are more important than achieving an attractive backtest number.

## Next steps

1. Add point-in-time joins for market, sector, liquidity, news and integrity context.
2. Add purged walk-forward folds and embargo around target horizons.
3. Add dataset/feature-set experiment identities on top of the existing source/output hashes.
4. Add a first real-data adapter/export path from OrionTrader into the canonical CSV contract.
5. Train the first tabular directional + return-magnitude baseline.
6. Compare it on matched timestamps against OrionTrader Premarket V5 and Regular-hours V3.1.
