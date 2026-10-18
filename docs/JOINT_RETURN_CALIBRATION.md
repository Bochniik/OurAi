# Experimental joint direction + magnitude distribution (v1)

**Scope:** new research-only module `trading/joint_return_calibration.py`. It does
not modify the frozen v3-15m champion, v3 forward protocol, shadow journal,
Alpaca fetch logic, or trading decisions. No claim of predictive improvement.

## Why

`P(UP)` from a conditional directional model and an expected absolute return
do **not** mathematically determine the expected signed return. The tempting
formula `(2 * p_up - 1) * predicted_abs_return` assumes equal positive and
negative move magnitudes, ignores neutrals and can give misleading prices.

This challenger uses past **train out-of-fold** direction/magnitude predictions
to look up an empirical joint distribution of signed 15m returns within fixed
probability/magnitude bins. All outputs come from the *same observations*:

- unconditional `P(UP)`, `P(DOWN)`, `P(NEUTRAL)` under a fixed +/- 3 bps band;
- expected signed return and expected absolute magnitude;
- empirical signed-return 10th, 50th and 90th percentiles;
- optional price projections from a strictly positive reference price;
- per-cell support; **abstain** (`None`) when fewer than 30 training examples
  support a cell.

The joint probabilities sum to one, quantiles are ordered, and
`abs(expected_signed_return) <= expected_abs_return` by construction.
These are *train empirical* summaries, not guaranteed calibrated or 80%-coverage
forecasts. Do not represent them as validated performance.

## Research/data contracts

1. Fit only with truly **out-of-fold predictions generated from TRAIN data**.
   No prediction may have been trained on its own realized target. A boolean
   `is_train_oof=True` must be provided, but **the caller must independently
   audit fold lineage**. This flag is a guardrail, not cryptographic proof.
2. The input record includes `horizon_minutes` and a timezone-aware UTC
   convertible timestamp. Predictions and actual signed future returns are
   finite fractions (0.004 = +0.4%). Mixing horizons or duplicates fails closed.
3. Only deterministic UTC epochs divisible by the horizon (e.g. 15 min) are
   included in fit. This avoids overlapping same-symbol 15m labels under
   standard forward-return targets. It is not a substitute for OOF embargo.
4. Both bin boundaries and minimum support are fixed before inspecting new
   validation or forward data. Do not tune them on untouched test/forward.
5. `predict()` takes only two raw prediction inputs and the horizon, never
   realized return or future market data. Sparse cells return `None`.
6. Never fit using v3 shadow journal or live-forward realized outcomes; those
   must remain independent tests. This module has no write access to journals.
7. Evaluate on a separate chronological evaluation set, including zero-coverage
   cases and stratification by market date/symbol/regime. The helper
   `evaluate_joint_return_distributions` reports three-class Brier,
   signed-return MAE, magnitude MAE, and 10–90% interval coverage *only among
   non-abstaining forecasts*. Report abstention separately.

## Minimal usage (illustrative)

```python
from trading.joint_return_calibration import (
    OOFReturnObservation, fit_joint_return_calibrator
)

# oof_rows are train-only, embargoed OOF direction + magnitude predictions.
# Each row must have timestamp, symbol, 15m horizon, p_up, predicted magnitude,
# realized signed return, and is_train_oof=True.
calibrator = fit_joint_return_calibrator(
    OOFReturnObservation(**row) for row in oof_rows
)
distribution = calibrator.predict(
    raw_p_up=0.58,
    predicted_abs_return=0.002,
    horizon_minutes=15,
)
if distribution is None:
    print("NO_PREDICTION: insufficient train-only empirical support")
else:
    print(distribution.expected_signed_return, distribution.q10_return,
          distribution.q90_return, distribution.p_up, distribution.p_down,
          distribution.p_neutral)
```

## Next experiment

Generate genuinely independent train-period OOF predictions from the direction
and magnitude models under time embargoes, fit this calibration layer from
those predictions only, and then evaluate *once* on a separated development
slice. Compare against train-only unconditional-return null, a signed ridge
baseline, and the pre-existing separate-head prediction; do not open v3 live
forward or any locked holdout for candidate selection.
