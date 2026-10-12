# Trading dataset foundation

This document defines the first data contract for the trading-focused OurAI project.

## Objective

One training sample represents what OurAI could have known at one real market timestamp, plus one or more outcomes that become known only later.

Conceptually:

```text
(symbol, timestamp)
        │
        ├── causal bar/quote features known at timestamp
        ├── point-in-time external context available by timestamp
        └── future outcomes at exact horizons
              ├── 5m
              ├── 15m
              ├── 30m
              └── 60m
```

The causal builder lives in `trading/dataset.py`. Raw CSV ingestion and deterministic split export live in `trading/ingestion.py`. External point-in-time context joins live in `trading/context.py`.

## Current bar/quote feature set

The first pass uses timestamped market information whose causality is easy to verify:

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

`prepare_trading_dataset()` performs the first reproducible price-data path:

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

## Point-in-time external context

OrionTrader already has valuable non-price inputs such as:

- news/catalyst analysis
- liquidity metrics
- market-integrity anomaly scores
- market regime
- sector regime
- session/context state

These are now represented by `ContextObservation` and attached through `join_point_in_time_context()`.

Every observation has two separate time concepts:

```text
event_time      when the underlying event/state happened
available_at    when OurAI could actually know it
```

`available_at` is the hard causality boundary. For example, if an earnings-related headline describes an event at 10:00 but is only available to our system at 10:03, a sample at 10:01 cannot receive that information.

An observation can have one of three scopes:

```text
global   -> applies to every symbol
sector   -> applies only to symbols explicitly mapped to that sector
symbol   -> applies only to one ticker
```

Within one source the join applies matching global context first, then sector context, then symbol context. More specific values may override a feature with the same name. This lets us combine a broad market state with a ticker-specific refinement without creating separate production engines.

### Stable context schema

Context features use names such as:

```text
ctx__news__sentiment
ctx__news__catalyst_strength
ctx__liquidity__spread_score
ctx__market_regime__risk_on
```

For each source the join also emits:

```text
ctx__<source>__available
ctx__<source>__age_minutes
```

and each row receives:

```text
context_missing_fraction
```

The feature schema stays stable across rows: if a known context feature is unavailable at a timestamp, its value is `None` rather than disappearing from the row.

### Freshness and expiry

`ContextSourcePolicy` can impose a maximum allowed age per source. `ContextObservation.valid_until` can expire an observation at a specific timestamp. Both mechanisms fail closed: stale or expired context is treated as unavailable.

The join returns an audit record showing exactly which observation IDs/timestamps were used for each sample. This is intended to support reproducible evidence and later dataset lineage.

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

A purged/embargoed walk-forward layer is the next validation increment.

## Leakage and provenance invariants

The tests enforce several important rules:

- changing a future price changes the future target but not earlier features;
- missing exact future timestamps do not create fake horizons;
- targets do not cross market sessions unless explicitly enabled;
- whole market dates stay disjoint across train/validation/test;
- spread-aware neutral labels are deterministic;
- naive timestamps require an explicit timezone policy;
- duplicate timestamps fail closed;
- raw input and generated split hashes are reproducible;
- context with `available_at` after a sample timestamp is never joined;
- an old `event_time` cannot bypass a later publication/availability timestamp;
- stale and expired context fails closed;
- symbol/sector context requires an explicit scope match;
- the context feature schema remains stable when sources are missing.

These invariants are more important than achieving an attractive backtest number.

## Next steps

1. Add purged walk-forward folds and embargo around target horizons.
2. Add dataset/feature-set experiment identities on top of the existing source/output hashes and context audit.
3. Add a first real-data adapter/export path from OrionTrader into the canonical bar/context contracts.
4. Train the first tabular directional + return-magnitude baseline.
5. Calibrate probabilities and define a preregistered NO_TRADE policy.
6. Compare OurAI on matched timestamps against OrionTrader Premarket V5 and Regular-hours V3.1.
