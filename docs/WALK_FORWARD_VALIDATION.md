# Purged walk-forward validation

OurAI trading models must be evaluated chronologically. Random row splits are not valid for market forecasting because nearby samples can share overlapping future-return label windows and because later market regimes must never be used to train predictions for earlier periods.

The implementation is in `trading/validation.py`.

## Intended experiment structure

Reserve the final chronological `test` split as an untouched lockbox. Use only development data (normally `train + validation`) to create walk-forward folds:

```text
PAST ----------------------------------------------------------> FUTURE

Fold 0
TRAIN TRAIN TRAIN | embargo | VALID VALID

Fold 1
TRAIN TRAIN TRAIN TRAIN TRAIN | embargo | VALID VALID

Fold 2
TRAIN TRAIN TRAIN TRAIN TRAIN TRAIN TRAIN | embargo | VALID VALID

                                                        TEST
                                                   untouched lockbox
```

By default folds use an expanding training window. `max_train_market_dates` can be set to create a rolling window when we want to test adaptation to regime drift.

## Purging

Every `TradingSample` has a feature timestamp and one or more future outcomes. Its label interval ends at the latest resolved `future_timestamp` among its targets.

Before a validation block starts, a candidate training row is removed when:

```text
training_sample.latest_target_time >= validation_start
```

This prevents a row from being used for training when its label outcome reaches into the validation period.

Today the first target set is 5m/15m/30m/60m and targets normally stay inside a named market session. The purge logic intentionally uses actual target timestamps rather than assuming a fixed horizon, so it remains correct if we later add daily/swing horizons or explicitly preregister cross-session targets.

## Embargo

An optional embargo removes training samples whose **feature timestamp** is too close to validation start, even if their target has technically matured. This creates an extra separation gap around the boundary.

For example:

```text
validation starts: 2026-01-15 14:30 UTC
embargo:           60 minutes

training rows at or after 13:30 UTC are excluded
```

The default is 60 minutes, matching the longest initial intraday horizon.

## Complete market dates

Validation windows are selected by complete `session_date` values. All symbols from one New York market date stay in the same validation block. This avoids training on NVDA from a day while validating on AAPL from that same day.

Purging can remove individual boundary rows or even a whole boundary market date. The fold manifest records the actual surviving training dates, sample counts, purge count and embargo count.

## Expanding vs rolling

Expanding window:

```python
purged_walk_forward_folds(
    development_samples,
    min_train_market_dates=20,
    validation_market_dates=5,
    step_market_dates=5,
    embargo_minutes=60,
)
```

Rolling window:

```python
purged_walk_forward_folds(
    development_samples,
    min_train_market_dates=20,
    validation_market_dates=5,
    step_market_dates=5,
    embargo_minutes=60,
    max_train_market_dates=60,
)
```

The rolling variant is useful for testing whether recent history is more informative than the full historical sample under market regime drift. It is an experiment choice, not something to tune on the final lockbox.

## What this protects against

The tests enforce that:

- training is strictly earlier than validation;
- training label intervals do not overlap validation start;
- embargoed rows are excluded;
- rolling windows respect the configured maximum train history;
- fold manifests are deterministic;
- malformed/naive target timestamps fail closed;
- insufficient history fails closed.

## Next step

After this layer is stable, the next major milestone is the first unified tabular baseline. It should be trained fold-by-fold, produce out-of-fold probabilities/return forecasts, then calibrate only from development evidence. The final untouched test remains unopened until the model family, features and decision policy are preregistered.