# First unified trading baseline

`trading/baseline.py` is the first trainable shared prediction model in the trading-focused OurAI codebase.

It is intentionally a **scientific floor**, not the final architecture and not a claim of profitable edge. The purpose is to establish one reproducible model that consumes the same leakage-safe samples, emits the same forecast contract, and can be beaten by later boosted-tree, temporal, or ensemble models on untouched data.

## What it learns

For one configured horizon (5m, 15m, 30m, or 60m), the baseline fits two shared heads:

- regularized logistic regression for `P(UP)` / `P(DOWN)`;
- ridge regression for expected future return magnitude.

Both heads see the same standardized point-in-time numeric features. Missing numeric values receive explicit missingness indicators. Ticker and session are encoded as context inside the **same shared model**; they are not separate per-stock production engines.

An unseen ticker can still be scored from the shared numeric/session features. Its unknown ticker indicator simply remains zero.

## Leakage policy

Fit the model on the training split only.

Validation can be used to compare model settings and decision thresholds. The final test split is a lockbox and must not be repeatedly inspected or tuned against. For mature experiments, use the existing purged walk-forward machinery before the final lockbox.

No target field is admitted into the feature matrix.

## Run it

First prepare chronological JSONL splits with the existing ingestion pipeline. Then run:

```powershell
python scripts\train_trading_baseline.py `
  --train data\processed\train.jsonl `
  --validation data\processed\validation.jsonl `
  --test data\processed\test.jsonl `
  --manifest data\processed\manifest.json `
  --output-dir runs\baseline-15m `
  --horizon 15
```

The command writes:

- `model.json` — human-readable model/scaler/schema plus experiment identity;
- `report.json` — fit summary and held-out validation/test metrics.

Large datasets, generated reports, and model artifacts should remain local/ignored rather than being committed.

## Metrics

The baseline feeds the common forecast evaluator, so the report includes:

- Brier score and log loss;
- direction and balanced accuracy;
- probability reliability bins;
- expected-return MAE;
- LONG / SHORT / NO_TRADE coverage;
- trade-direction accuracy on selected forecasts;
- simple realized return after configured spread/slippage/fees.

These values are diagnostics, not a profitability guarantee.

## Why linear first

A linear baseline is cheap enough for the user's current Windows/GTX 1650/8 GB development machine, deterministic, easy to audit, and difficult to hide leakage behind. It gives us a hard reference point.

The next model family should be a stronger nonlinear tabular learner. It only earns promotion if it improves matched leakage-safe validation/walk-forward results without degrading calibration or relying on the test lockbox.
