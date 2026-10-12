# OurAI — Trading Intelligence

OurAI is now focused exclusively on **market prediction and trading decision support**. The goal is not to build a general-purpose chatbot. The goal is to build a local, self-hosted forecasting system that learns from market data and produces calibrated, testable predictions for liquid equities.

## Core objective

For a stock at time `t`, OurAI should estimate, for several horizons:

- `P(UP)` and `P(DOWN)`
- expected return in percent
- uncertainty / calibrated confidence
- expected return interval (for example 10th/50th/90th percentile)
- `LONG`, `SHORT`, or `NO_TRADE`

Initial horizons:

- 5 minutes
- 15 minutes
- 30 minutes
- 1 hour

Daily / swing horizons can be added after intraday evaluation is reliable.

> OurAI must never pretend that a market outcome is certain. A high-quality model should be able to abstain when evidence is weak. `NO_TRADE` is a first-class prediction.

## Why this direction

Trading prediction is a different machine-learning problem from text generation. The original v0.1 decoder-only Transformer proved that we can build and train neural networks locally on the GTX 1650, but a language model is not the best core architecture for forecasting stock returns.

The v0.1 language-model code is retained as an engineering prototype. New development prioritizes market data, leakage-safe validation, probabilistic forecasting, calibration, and live forward testing.

## Planned system

```text
Market data
    │
    ├── OHLCV
    ├── relative volume / liquidity
    ├── volatility
    ├── VWAP / EMA / momentum features
    ├── market + sector context
    ├── pre-market / session state
    └── later: news / order-flow features
            │
            ▼
      Feature pipeline
            │
      ┌─────┴───────────────┐
      ▼                     ▼
Tabular baseline       Temporal model
(XGBoost/LightGBM)      (PyTorch)
      │                     │
      └──────────┬──────────┘
                 ▼
        Calibration / ensemble
                 │
                 ▼
     Direction + magnitude + risk
                 │
                 ▼
      LONG / SHORT / NO_TRADE
```

## Development principles

1. **Out-of-sample first.** We do not judge models by training accuracy.
2. **No leakage.** Features and splits must not contain future information.
3. **Walk-forward evaluation.** Time-series validation must respect chronology.
4. **Calibration matters.** A reported 70% probability should behave like roughly 70% over many comparable predictions.
5. **Transaction costs matter.** Backtests must include spread/slippage assumptions.
6. **Liquidity first.** Initial universe should favor highly liquid equities where price formation and execution are more stable.
7. **Abstention is allowed.** The model should refuse low-confidence setups rather than force a direction.
8. **Live-forward evidence beats backtest optimism.** Candidate models must prove themselves on unseen future observations before being trusted.

## Target outputs

Example only:

```text
Ticker: NVDA
Horizon: 30m

P(UP):                 72.4%
P(DOWN):               27.6%
Expected return:       +0.63%
Predicted interval:    -0.28% ... +1.39%
Confidence:            0.74
Decision:              LONG
```

Or, when evidence conflicts:

```text
P(UP):      53.1%
P(DOWN):    46.9%
Decision:   NO_TRADE
Reason:     insufficient calibrated edge
```

## Evaluation metrics

Direction prediction:

- log loss
- Brier score
- ROC-AUC as a secondary diagnostic
- precision / hit rate in high-confidence buckets
- Expected Calibration Error (ECE)

Magnitude prediction:

- MAE / RMSE of future returns
- quantile / interval coverage

Trading usefulness:

- expected return after estimated costs
- drawdown
- profit factor / Sharpe-like diagnostics where appropriate
- performance by liquidity and market regime
- forward-test results separated from backtests

Raw hit rate alone is not enough.

## Hardware / execution

The current development machine has been validated with:

- NVIDIA GeForce GTX 1650 4 GB
- PyTorch 2.14 + CUDA 12.6
- Windows self-hosted GitHub Actions runner

The GPU runner is intentionally restricted to trusted/manual workflows because this repository is public.

## Current repository status

The original v0.1 language-model prototype includes:

- custom decoder-only Transformer
- Byte-Level BPE tokenizer
- CUDA mixed-precision training
- checkpointing and generation

This remains useful as proof that the local ML/CUDA stack works, but it is no longer the product direction.

## Trading AI roadmap

### Phase 1 — Market dataset foundation

- provider-agnostic market-data schema
- OHLCV ingestion
- multi-symbol / multi-session data layout
- data quality checks
- leakage-safe future-return target builder
- multi-horizon labels
- train/validation/test splits by time

### Phase 2 — Strong baseline

- feature engineering
- liquidity filters
- XGBoost/LightGBM direction model
- return-magnitude regressor
- calibrated probabilities
- reproducible walk-forward evaluation

### Phase 3 — Decision engine

- confidence thresholds
- uncertainty estimates
- LONG / SHORT / NO_TRADE policy
- realistic cost/slippage model
- regime-specific diagnostics

### Phase 4 — Temporal neural model

- compact sequence model compatible with 4 GB VRAM
- compare against the tabular baseline
- keep it only if it improves truly held-out performance

### Phase 5 — Ensemble + live forward testing

- ensemble specialist models
- calibration / meta-model
- persist every live prediction before the outcome is known
- score predictions after each horizon expires
- promote model versions only when forward evidence improves

### Phase 6 — Orion integration

Expose stable predictions to OrionTrader through a local API such as:

```text
/predict/{ticker}
```

with direction probability, magnitude, uncertainty, decision, model version, and timestamp.

## Important

Financial markets are noisy and non-stationary. No model can guarantee the future price of a stock. The engineering target is therefore **measurable probabilistic edge with calibrated uncertainty**, not fake certainty.
