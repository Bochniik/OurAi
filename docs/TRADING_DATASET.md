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

The implementation lives in `trading/dataset.py`.

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

A purged/embargoed walk-forward layer will be added after ingestion and point-in-time context joins are stable.

## External context is deliberately not joined yet

OrionTrader already has useful information from:

- news/catalyst analysis
- liquidity
- market-integrity anomaly scores
- market/sector regime
- session state

Those sources are valuable, but they must be joined with explicit `event_time` / `as_of` rules. The first foundation therefore does **not** silently forward-fill them into historical rows. A later point-in-time context joiner will reject context that was not known by the feature timestamp.

## Leakage invariants

The tests enforce several important rules:

- changing a future price changes the future target but not earlier features
- missing exact future timestamps do not create fake horizons
- targets do not cross market sessions unless explicitly enabled
- whole market dates stay disjoint across train/validation/test
- spread-aware neutral labels are deterministic

These invariants are more important than achieving an attractive backtest number.

## Next steps

1. Add provider-neutral CSV/Parquet ingestion with schema validation and provenance.
2. Add point-in-time joins for market, sector, liquidity, news and integrity context.
3. Add purged walk-forward folds and embargo around target horizons.
4. Add dataset hashes/manifests for reproducible experiments.
5. Train the first tabular directional + return-magnitude baseline.
6. Compare it on matched timestamps against OrionTrader Premarket V5 and Regular-hours V3.1.
