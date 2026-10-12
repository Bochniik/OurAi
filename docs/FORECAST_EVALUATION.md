# Forecast evaluation and NO-TRADE policy

OurAI must not turn every forecast into a trade. `trading/evaluation.py` keeps three concerns separate:

1. **forecast** — `P(UP)`, `P(DOWN)`, and expected return;
2. **evaluation** — probability quality, calibration, direction accuracy, return error;
3. **decision policy** — LONG, SHORT, or NO_TRADE after estimated spread, slippage, fees, and a safety edge.

That separation is intentional. A model can be directionally correct yet still produce an unprofitable trade if the expected move is too small relative to costs.

## Forecast contract

```python
ForecastPrediction(
    symbol="NVDA",
    timestamp=timestamp,
    horizon_minutes=15,
    p_up=0.76,
    expected_return=0.0062,
    actual_return=None,
    spread_bps=1.8,
)
```

`p_down` is derived as `1 - p_up`. `actual_return` is omitted for a live forecast and is attached later when the horizon resolves.

## Abstaining decision policy

The default policy requires:

- directional probability of at least 60%;
- expected-return sign to agree with the probability direction;
- expected move to clear estimated spread + slippage + fees;
- an additional minimum safety edge.

Example:

```python
decision = decide_forecast(forecast)
print(decision.action)  # LONG / SHORT / NO_TRADE
```

A high `P(UP)` is therefore not sufficient by itself. If the expected move is tiny, costs are high, or the expected-return model disagrees with the direction model, the output is `NO_TRADE`.

## Held-out metrics

`evaluate_forecasts(...)` reports, for one horizon at a time:

- Brier score;
- log loss;
- direction accuracy;
- balanced accuracy when both UP and DOWN are present;
- equal-width probability calibration bins;
- expected-return MAE in basis points;
- trade coverage;
- LONG/SHORT/NO_TRADE counts;
- accuracy conditional on taking a trade;
- average realized return after the configured cost estimate;
- positive net-trade rate.

Neutral outcomes can be excluded from the UP-vs-DOWN probability metrics with `neutral_threshold_bps`. This matches the intended architecture: the direction head estimates conditional UP-vs-DOWN probability, while expected return and the policy decide whether the opportunity is strong enough to trade.

## Scientific use

These metrics belong on validation/test/walk-forward/forward data, not on the same samples used to fit the model. Cost assumptions and policy thresholds are scientifically relevant experiment inputs and therefore belong in `ExperimentIdentity`.

No metric in this module is a claim of live profitability. Its purpose is to make model comparisons reproducible and to prevent a raw probability score from being mistaken for a trade recommendation.
