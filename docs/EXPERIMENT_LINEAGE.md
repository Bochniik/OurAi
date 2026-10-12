# Experiment identity and lineage

A trading result is only meaningful if we can reproduce exactly what data, features, validation design, model settings and code produced it.

`trading/experiment.py` creates deterministic SHA-256 fingerprints for those inputs and combines them into one compact experiment ID.

## Inputs to an experiment identity

Every preregistered experiment should identify at least:

1. **dataset provenance** — normally the ingestion manifest containing raw-source and train/validation/test hashes;
2. **feature schema** — the exact set of feature names presented to the model;
3. **validation design** — walk-forward fold policy, embargo, fold layout and related settings;
4. **experiment configuration** — model family, hyperparameters, target horizon, calibration settings, transaction-cost assumptions and decision/abstention thresholds;
5. **code revision** — the exact Git commit SHA used to run the experiment.

Example:

```python
identity = build_experiment_identity(
    dataset_provenance=dataset_manifest,
    feature_names=feature_names,
    validation_design=walk_forward_manifest(folds),
    experiment_config={
        "model": "gradient_boosted_trees",
        "horizon": "15m",
        "calibration": "isotonic",
        "decision_threshold": 0.65,
        "cost_model": {"spread_bps": "observed", "slippage_bps": 2.0},
    },
    code_revision="<git-sha>",
)
```

The result contains component hashes plus an ID such as:

```text
exp_4d8a7c3b1f0e92aa
```

Changing any scientifically relevant component changes the experiment ID.

## Why this matters

Without lineage it is easy to compare two scores that were produced from different datasets, different leakage rules or different decision thresholds and mistakenly attribute the difference to the model.

The lineage hash lets us enforce a stronger rule:

> no metric belongs to a model name alone; it belongs to a specific experiment identity.

This will later be stored alongside checkpoints, calibration artifacts, forward predictions and OrionTrader comparison reports.

## Canonicalization

Mappings are sorted before hashing, feature names are deduplicated and sorted, and non-finite numbers such as NaN/Infinity are rejected. This makes identities deterministic across machines while failing closed on ambiguous metadata.

## Next step

The first unified tabular baseline should emit its metrics under one `ExperimentIdentity`. If features, folds, costs, thresholds or data change, the new run must receive a new identity rather than silently overwriting the old result.