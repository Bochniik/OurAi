"""Train and evaluate the first unified OurAI trading baseline from prepared JSONL splits."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any

from trading import (
    BaselineConfig,
    DecisionPolicyConfig,
    build_experiment_identity,
    evaluate_forecasts,
    fit_unified_linear_baseline,
    forecast_samples,
    sha256_file,
)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                payload = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"expected JSON object at {path}:{line_number}")
            rows.append(payload)
    if not rows:
        raise ValueError(f"{path} contains no samples")
    return rows


def _code_revision(explicit: str | None) -> str:
    if explicit:
        return explicit.strip()
    github_sha = os.environ.get("GITHUB_SHA", "").strip()
    if github_sha:
        return github_sha
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return "unresolved-local-revision"


def _dataset_provenance(
    train: Path,
    validation: Path,
    test: Path | None,
    manifest: Path | None,
) -> dict[str, Any]:
    if manifest is not None:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("dataset manifest must be a JSON object")
        return payload

    splits: dict[str, Any] = {
        "train": {"path": train.name, "sha256": sha256_file(train)},
        "validation": {"path": validation.name, "sha256": sha256_file(validation)},
    }
    if test is not None:
        splits["test"] = {"path": test.name, "sha256": sha256_file(test)}
    return {
        "schema_version": 1,
        "source": "prepared_jsonl_splits",
        "splits": splits,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the first shared direction + return baseline for OurAI."
    )
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument(
        "--test",
        type=Path,
        help="Optional untouched test lockbox. It is read only with --evaluate-test.",
    )
    parser.add_argument(
        "--evaluate-test",
        action="store_true",
        help="Explicitly unlock and evaluate the test split. Do this only after model/policy choices are frozen.",
    )
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--horizon", type=int, default=15, choices=(5, 15, 30, 60))
    parser.add_argument("--neutral-threshold-bps", type=float, default=3.0)
    parser.add_argument("--code-revision")
    parser.add_argument("--probability-epochs", type=int, default=120)
    parser.add_argument("--probability-learning-rate", type=float, default=0.05)
    parser.add_argument("--probability-l2", type=float, default=1e-3)
    parser.add_argument("--regression-l2", type=float, default=1.0)
    parser.add_argument("--min-direction-probability", type=float, default=0.60)
    parser.add_argument("--minimum-edge-bps", type=float, default=2.0)
    parser.add_argument("--slippage-bps", type=float, default=1.5)
    parser.add_argument("--fees-bps", type=float, default=0.5)
    parser.add_argument("--fallback-spread-bps", type=float, default=5.0)
    args = parser.parse_args()

    if args.evaluate_test and args.test is None:
        parser.error("--evaluate-test requires --test")

    training_rows = _load_jsonl(args.train)
    validation_rows = _load_jsonl(args.validation)
    test_rows = _load_jsonl(args.test) if args.evaluate_test else None

    model_config = BaselineConfig(
        horizon_minutes=args.horizon,
        probability_l2=args.probability_l2,
        probability_learning_rate=args.probability_learning_rate,
        probability_epochs=args.probability_epochs,
        regression_l2=args.regression_l2,
    )
    policy = DecisionPolicyConfig(
        min_direction_probability=args.min_direction_probability,
        minimum_edge_bps=args.minimum_edge_bps,
        slippage_bps=args.slippage_bps,
        fees_bps=args.fees_bps,
        fallback_spread_bps=args.fallback_spread_bps,
    )

    model, fit_summary = fit_unified_linear_baseline(
        training_rows,
        config=model_config,
    )
    validation_forecasts = forecast_samples(model, validation_rows)
    validation_report = evaluate_forecasts(
        validation_forecasts,
        policy=policy,
        neutral_threshold_bps=args.neutral_threshold_bps,
    )

    test_report = None
    if test_rows is not None:
        test_forecasts = forecast_samples(model, test_rows)
        test_report = evaluate_forecasts(
            test_forecasts,
            policy=policy,
            neutral_threshold_bps=args.neutral_threshold_bps,
        )

    dataset_provenance = _dataset_provenance(
        args.train,
        args.validation,
        args.test if args.evaluate_test else None,
        args.manifest,
    )
    experiment_config = {
        "model_family": "unified_linear_baseline",
        "horizon_minutes": args.horizon,
        "probability_l2": args.probability_l2,
        "probability_learning_rate": args.probability_learning_rate,
        "probability_epochs": args.probability_epochs,
        "regression_l2": args.regression_l2,
        "neutral_threshold_bps": args.neutral_threshold_bps,
        "test_lockbox_evaluated": bool(args.evaluate_test),
        "decision_policy": {
            "min_direction_probability": policy.min_direction_probability,
            "minimum_edge_bps": policy.minimum_edge_bps,
            "slippage_bps": policy.slippage_bps,
            "fees_bps": policy.fees_bps,
            "fallback_spread_bps": policy.fallback_spread_bps,
        },
    }
    identity = build_experiment_identity(
        dataset_provenance=dataset_provenance,
        feature_names=model.schema.expanded_feature_names,
        validation_design={
            "policy": "chronological_train_validation_test_holdout",
            "training_split": "train_only",
            "threshold_selection_split": "validation_only",
            "final_report_split": "test" if args.evaluate_test else "untouched_test_lockbox",
        },
        experiment_config=experiment_config,
        code_revision=_code_revision(args.code_revision),
    )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    model_payload = model.to_dict()
    model_payload["experiment_identity"] = identity.to_dict()
    (output_dir / "model.json").write_text(
        json.dumps(model_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report = {
        "schema_version": 1,
        "experiment_identity": identity.to_dict(),
        "fit": fit_summary.to_dict(),
        "validation": validation_report,
        "test": test_report,
        "test_lockbox_evaluated": bool(args.evaluate_test),
        "notes": [
            "This is a shared linear scientific baseline, not a claim of predictive edge.",
            "The test split is not read unless --evaluate-test is supplied explicitly.",
            "Test metrics must not be used to tune model or decision thresholds.",
            "A boosted-tree model should only replace this baseline after matched held-out comparison.",
        ],
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(f"Experiment: {identity.experiment_id}")
    print(f"Model:      {output_dir / 'model.json'}")
    print(f"Report:     {output_dir / 'report.json'}")
    print("Validation Brier:", validation_report["probability"]["brier_score"])
    print("Validation coverage:", validation_report["decision_policy"]["coverage"])
    if test_report is None:
        print("Test lockbox: UNTOUCHED")
    else:
        print("Test Brier:", test_report["probability"]["brier_score"])
        print("Test coverage:", test_report["decision_policy"]["coverage"])


if __name__ == "__main__":
    main()
