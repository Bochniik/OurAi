# Joint return: chronological TRAIN-only development experiment (v1)

**Experimental comparison; no production promotion.** This is not the V3
forward test and cannot establish a reliable trading edge by itself.

## Exact run

```powershell
git checkout main
git pull origin main
python -m unittest discover -s tests -p "test_joint_return_development.py" -v
python -m scripts.evaluate_joint_return_development --dataset-dir data/processed/market/first-real-direction-v4 --output runs/joint-return-development-v1/report.json
```

The script verifies the canonical v4 historical dataset contract and its
UNCHANGED validation/test declarations using existing metadata checks, then
**opens only `train.jsonl`** for outcomes. It does not open `validation.jsonl`,
`test.jsonl`, the V3 forward journal, or the V3 frozen bundle.

## Prespecified research design (before seeing this report)

- **Dataset:** canonical v4 train split, expected dataset ID
  `7d5f3b066a30ba878bba`. Already consulted for other v4/v5/v6
  research; treat scores as exploratory development only.
- **Horizon:** 15 minutes, exact future timestamps, UTC 15-minute
  non-overlapping anchors chosen independently of returns.
- **Development window:** last 20 whole market dates **within v4 train**,
  held out before calibration fitting. All earlier dates are training.
- **Purge/embargo:** latest training label must mature before development
  starts; 60-minute embargo. The inner OOF generator has its own 60-minute
  embargo and expanding 55 train / 15 held-out / 15 step market-date folds.
- **Raw heads:** existing unified logistic direction model
  `fit_unified_linear_baseline` and ridge absolute magnitude model
  `fit_unified_linear_magnitude_baseline`. No architecture or hyperparameter
  changes in this experiment.
- **Joint calibration:** fixed `JointCalibrationConfig`, fixed 3 bps
  neutral band, fixed probability/magnitude bins, minimum 30 TRAIN OOF
  examples per supported bin. Unsupported bins abstain. Do not tune
  bins, support threshold or window choice on this evaluation outcome.
- **No target peek:** development predictions are made on a
  `features + targets={}` object; true returns are used for scoring
  *only after* both heads emit forecasts.
- **Signed-return MAE (bps):** joint expected signed return versus
  same-row linear signed return and earlier-train median signed-return null.
- **Absolute-magnitude MAE (bps):** joint mean absolute return versus
  same-row linear magnitude and earlier-train median absolute-return null.
- **Three-class Brier:** joint (UP / DOWN / NEUTRAL) versus
  earlier-train unconditional class-frequency null; the direction head's
  conditional binary probability is **not** compared directly as a
  three-class forecast.
- **Interval coverage:** proportion of supported outcomes between
  predicted signed-return q10 and q90 (nominal 80%); report average
  interval width. No coverage guarantee.
- **Abstention:** report eligibility, number of predictions,
  number declined and the fraction covered. Paired model/null scores
  apply **only to the same covered rows**. Also report linear baselines
  over *all* eligible development rows as a coverage transparency check.
- **Decision:** `RESEARCH_ONLY_NO_PROMOTION` or
  `INSUFFICIENT_SUPPORT`; never automate trading or model promotion.

## Interpretation

Lower signed/absolute-return MAE is better. Lower three-class Brier is
better. Empirical coverage near 0.80 is desirable **only together with**
honestly reported interval width and adequate support. More accuracy
among a handful of accepted predictions cannot be called a general
improvement when the model abstains on most others.

Multiple symbols and dates are correlated, so per-row counts are not
independent trials. All model families already were explored on related
train data; fresh genuinely untouched test and future-forward observation
are required before any trusted deployment. This experiment does not
optimize or calculate PnL or alter the V3 test protocol.

**Report location:** `runs/joint-return-development-v1/report.json`
(local, gitignored). Share only the metrics/summary; never API credentials.

## Unit-test contract

`tests/test_joint_return_development.py` uses synthetic data, verifies
non-overlapping chronological training and evaluation dates, tests that
evaluation outcomes never enter fit/OOF training, and verifies that the
runner reads `train.jsonl` only (other split files are deliberately
malformed). GitHub CI runs the test suite for each PR.
