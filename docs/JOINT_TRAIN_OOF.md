# Train-only walk-forward OOF generation for joint return calibration (research v2)

## Purpose

The first joint-distribution prototype (`trading/joint_return_calibration.py`)
requires predictions made on unseen TRAIN-period observations. This new builder
(`trading/joint_oof.py`) generates those records from **existing baseline**
direction and magnitude models without changing V3, touching the real forward
journal, opening reserved validation or test splits, or making a trade.

Each expanding fold refits independently:
- `fit_unified_linear_baseline` for `P(UP | directional move)`;
- `fit_unified_linear_magnitude_baseline` for predicted absolute 15m return.

The fold uses the existing `purged_walk_forward_folds` with deterministic
training earlier than validation and a default 60-minute embargo. Rows whose
label intervals reach the validation block are removed from training.

## Rules / limitations

1. **Caller must load only `train.jsonl`** from the intended chronological
   dataset and verify the dataset manifest and split provenance. Pass
   `source_split="train"`. This is a code guard, not attestation of actual
   file provenance.
2. Every resolved 15m target must have exact timezone-aware future timestamp
   `t + 15 minutes`, valid direction and finite signed return. Duplicate
   symbol/time observations, mismatched model horizons, overlapping OOF date
   windows and too-short embargo are errors.
3. Predictions on each fold's held-out rows use a fresh **feature-only dict
   with `targets={}`**, and the resulting pair of raw forecasts is joined
   with the held-out training-period realized return *after* inference.
4. The OOF rows are subsampled at a fixed epoch-aligned 15-minute anchor,
   independent of realized labels, avoiding adjacent same-symbol 15-minute
   outcome overlap. Purging, consistent folds, and time embargo remain required.
5. **OOF is not a final evaluation or evidence of trading edge.** It may fit a
   joint calibration layer; any performance claim requires a separate later
   strictly chronological development evaluation, comparisons against nulls and
   directional/magnitude baselines, then locked independent test and future
   forward results. Shared market factors across symbols remain correlated.
6. Raw direction `P(UP)` is conditional on the UP/DOWN task, whereas the joint
   empirical calibration explicitly estimates UP, DOWN and NEUTRAL among all
   supported calibration examples; do not confuse these contracts.
7. No V3 model, frozen protocol, shadow journal, test lockbox or trading
   integration is updated by this research module. It has **no I/O**.

## Usage in Python (no data-file reading is performed by this module)

```python
from trading.joint_oof import generate_joint_train_oof
from trading.joint_return_calibration import fit_joint_return_calibrator

# train_samples: TradingSample instances loaded ONLY from verified train.jsonl.
out = generate_joint_train_oof(train_samples, source_split="train")
print(out.manifest)
calibrator = fit_joint_return_calibrator(out.observations)
```

This function is expensive for long, multi-symbol datasets because the two
existing baselines are refitted in each expanding fold. The default fold
layout is 55 training / 15 validation market dates, step 15, 60m embargo.
A held-out evaluation set must remain **separate** from the OOF records used
to fit calibration; the caller is responsible for not scoring on fit data.

Unit tests with synthetic TRAIN rows:

```powershell
python -m unittest discover -s tests -p "test_joint_oof.py" -v
```

### Follow-up research task

Implement a **read-only train-split experiment script** that verifies the
market dataset manifest, reserves an unseen development block before creating
OOF folds, trains calibration on earlier TRAIN dates only and scores a later
TRAIN-only block against signed-return null and benchmark heads. Freeze the
design before examining results. No live promotion based on a single run.
