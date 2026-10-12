# OrionTrader market adapter

`trading/orion_adapter.py` is the boundary between OrionTrader market exports and the canonical OurAI `MarketBar` contract.

The adapter is deliberately dependency-light and does **not** import OrionTrader, pandas, SQLAlchemy, Yahoo Finance, MT5, secrets, or live infrastructure. OurAI can therefore train from exported rows while preserving reproducibility and keeping runtime systems decoupled.

## Supported row shapes

The adapter accepts common aliases already present in OrionTrader-style market data:

- timestamp: `timestamp`, `time`, `datetime`
- symbol: `symbol`, `ticker`, or `OrionAdapterConfig.default_symbol`
- OHLC: `open`, `high`, `low`, `close`
- volume: `volume`, `tick_volume`, `real_volume`
- quote fields: `bid` / `best_bid`, `ask` / `best_ask`
- session: `session` / `market_session`

Rows can be mappings, named rows exposing `_mapping`, or objects with the expected attributes.

## Yahoo/DataFrame export example

If Orion has a DataFrame whose index is `time` and whose columns are OHLCV:

```python
from trading import OrionAdapterConfig, normalize_orion_rows

rows = yahoo_df.reset_index().to_dict("records")
result = normalize_orion_rows(
    rows,
    config=OrionAdapterConfig(
        source_name="orion-yahoo",
        default_symbol="NVDA",
    ),
)

bars = result.bars
manifest = result.manifest
```

Timezone-aware timestamps are preferred. A naive timestamp is rejected unless the source timezone is explicitly supplied, for example `source_timezone="America/New_York"`.

## MT5-style export example

MT5 rate rows often expose Unix `time` plus `tick_volume`:

```python
result = normalize_orion_rows(
    mt5_rows,
    config=OrionAdapterConfig(
        source_name="orion-mt5",
        default_symbol="EURUSD",
    ),
)
```

The manifest records whether `tick_volume` or source `volume` was consumed. Those two meanings must not be treated as interchangeable without an explicit model decision.

## Reproducibility and safety

The adapter fails closed on:

- missing timestamps, symbols, OHLC fields, or volume;
- naive timestamps without an explicit source timezone;
- non-finite values rejected by the canonical `MarketBar` contract;
- duplicate `(symbol, timestamp)` rows by default.

`strict_duplicates=False` may be used only when an upstream export is known to contain byte-for-byte repeated bars. Skipped duplicates are counted in the manifest.

The resulting manifest includes symbol coverage, temporal coverage, aliases consumed, row counts, duplicate policy, and volume semantics. This metadata should become part of the dataset provenance used by `ExperimentIdentity`.

## Separation of responsibilities

The adapter only normalizes **market bars**. Orion-derived news, liquidity, regime, event, integrity, or other higher-level context belongs in the point-in-time context layer (`trading/context.py`) so every feature is joined according to its true publication/as-of time.

This separation prevents a convenient integration layer from silently introducing look-ahead leakage.
