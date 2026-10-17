"""Train-only development gate for the preregistered v4 nonlinear direction head.

Only ``train.jsonl`` is opened. The fresh v4 validation and test splits remain
untouched until separate, later gates are committed. Promotion is based on proper
probability scores only; PnL is deliberately excluded.

The original v4 preregistration expected eight walk-forward folds. After the real
dataset was built, the first development invocation deterministically produced seven
folds and stopped on the fold-count assertion before the model-fit loop started. A
committed amendment therefore changes only the expected fold count from eight to
seven. Model configuration, metrics, thresholds, split boundaries and holdouts remain
unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from statistics import fmean
from typing import Any, Mapping

from scripts.build_v4_direction_dataset import preregistered_protocol
from scripts.evaluate_magnitude_head import _load_train
from trading.baseline import BaselineConfig, fit_unified_linear_baseline
from trading.direction import (
    HistGradientDirectionConfig,
    evaluate_direction_predictions,
    fit_hist_gradient_direction,
    forecast_hist_gradient_directions,
    forecast_linear_directions,
)
from trading.validation import purged_walk_forward_folds, walk_forward_manifest


HORIZON_MINUTES = 15
MIN_TRAIN_MARKET_DATES = 55
VALIDATION_MARKET_DATES = 15
STEP_MARKET_DATES = 15
EMBARGO_MINUTES = 60.0
BASE_EXPECTED_FOLD_COUNT = 8
EXPECTED_FOLD_COUNT = 7
EXPECTED_TRAIN_MARKET_DATES = 174
MINIMUM_BRIER_FOLD_WINS_VS_LINEAR = 6
MINIMUM_BRIER_FOLD_WINS_VS_NULL = 6
EXPECTED_BUILDER = "ourai-historical-market-v2"
EXPECTED_DATASET_ID = "7d5f3b066a30ba878bba"
AMENDMENT_PATH = (
    Path(__file__).resolve().parents[1]
    / "research"
    / "v4_direction_development_amendment_v1.json"
)

NONLINEAR_CONFIG = HistGradientDirectionConfig(
    horizon_minutes=HORIZON_MINUTES,
    learning_rate=0.05,
    max_iter=150,
    max_leaf_nodes=31,
    min_samples_leaf=100,
    l2_regularization=1.0,
    max_bins=255,
    random_state=42,
    minimum_training_samples=100,
)
LINEAR_CONFIG = BaselineConfig(
    horizon_minutes=HORIZON_MINUTES,
    probability_l2=1e-3,
    probability_learning_rate=0.05,
    probability_epochs=120,
    probability_batch_size=4096,
    regression_l2=1.0,
    minimum_training_samples=40,
    seed=42,
)


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _canonical_json_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_amendment() -> dict[str, Any]:
    if not AMENDMENT_PATH.is_file():
        raise FileNotFoundError(f"required v4 development amendment not found: {AMENDMENT_PATH}")
    payload = json.loads(AMENDMENT_PATH.read_text(encoding="utf-8"))
    amendment = _mapping(payload, "v4 direction development amendment")

    expected_protocol = preregistered_protocol()
    if amendment.get("amendment_name") != "ourai-v4-direction-development-fold-count-amendment-v1":
        raise ValueError("unexpected v4 direction development amendment identity")
    if amendment.get("base_protocol_name") != expected_protocol["protocol_name"]:
        raise ValueError("v4 amendment base protocol name mismatch")
    if amendment.get("base_protocol_sha256") != expected_protocol["protocol_sha256"]:
        raise ValueError("v4 amendment base protocol SHA256 mismatch")
    if amendment.get("applies_to_dataset_id") != EXPECTED_DATASET_ID:
        raise ValueError("v4 amendment dataset identity mismatch")
    if amendment.get("created_before_model_outcomes") is not True:
        raise ValueError("v4 amendment must state it was frozen before model outcomes")

    observed = _mapping(amendment.get("observed_before_amendment"), "observed_before_amendment")
    if observed.get("train_market_date_count") != EXPECTED_TRAIN_MARKET_DATES:
        raise ValueError("v4 amendment train market-date count mismatch")
    if observed.get("computed_fold_count") != EXPECTED_FOLD_COUNT:
        raise ValueError("v4 amendment computed fold count mismatch")
    if observed.get("model_fit_loop_started") is not False:
        raise ValueError("v4 amendment must precede the model-fit loop")
    if observed.get("development_metrics_observed") is not False:
        raise ValueError("v4 amendment must precede development metrics")
    if observed.get("validation_split_accessed") is not False:
        raise ValueError("v4 amendment must keep validation untouched")
    if observed.get("test_split_accessed") is not False:
        raise ValueError("v4 amendment must keep test untouched")

    change = _mapping(amendment.get("change"), "change")
    if dict(change) != {
        "field": "development_gate.expected_fold_count",
        "from": BASE_EXPECTED_FOLD_COUNT,
        "to": EXPECTED_FOLD_COUNT,
    }:
        raise ValueError("v4 amendment changes more than the expected fold count")

    unchanged = _mapping(amendment.get("unchanged_contract"), "unchanged_contract")
    expected_unchanged = {
        "source_interval": True,
        "dataset_and_split_boundaries": True,
        "direction_contract": True,
        "nonlinear_model_family_and_hyperparameters": True,
        "linear_incumbent": True,
        "walk_forward_window_sizes_and_embargo": True,
        "minimum_brier_fold_wins_vs_linear": MINIMUM_BRIER_FOLD_WINS_VS_LINEAR,
        "minimum_brier_fold_wins_vs_null": MINIMUM_BRIER_FOLD_WINS_VS_NULL,
        "mean_brier_requirements": True,
        "mean_log_loss_requirements": True,
        "pnl_used": False,
    }
    if dict(unchanged) != expected_unchanged:
        raise ValueError("v4 amendment altered a frozen development requirement")
    return dict(amendment)


def _expected_nonlinear_protocol_config() -> dict[str, Any]:
    return {
        "family": "HistGradientBoostingClassifier",
        "loss": "log_loss",
        "learning_rate": NONLINEAR_CONFIG.learning_rate,
        "max_iter": NONLINEAR_CONFIG.max_iter,
        "max_leaf_nodes": NONLINEAR_CONFIG.max_leaf_nodes,
        "min_samples_leaf": NONLINEAR_CONFIG.min_samples_leaf,
        "l2_regularization": NONLINEAR_CONFIG.l2_regularization,
        "max_bins": NONLINEAR_CONFIG.max_bins,
        "early_stopping": False,
        "random_state": NONLINEAR_CONFIG.random_state,
    }


def _validate_v4_contract(
    root: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    manifest_path = root / "manifest.json"
    protocol_path = root / "v4_direction_preregistration.json"
    result_path = root / "v4_direction_protocol_result.json"
    for path in (manifest_path, protocol_path, result_path):
        if not path.is_file():
            raise FileNotFoundError(f"required v4 direction artifact not found: {path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, Mapping) or not isinstance(protocol, Mapping) or not isinstance(result, Mapping):
        raise ValueError("v4 direction artifacts must contain JSON objects")

    expected = preregistered_protocol()
    if protocol != expected:
        raise ValueError("v4 direction preregistration does not match committed base protocol")
    if result.get("protocol_sha256") != expected["protocol_sha256"]:
        raise ValueError("v4 direction protocol result hash mismatch")
    if result.get("validation_status") != "UNTOUCHED":
        raise ValueError("v4 direction validation is not marked UNTOUCHED")
    if result.get("test_status") != "UNTOUCHED":
        raise ValueError("v4 direction test is not marked UNTOUCHED")
    if result.get("overlap_with_v3") is not False:
        raise ValueError("v4 direction source overlaps v3")
    if manifest.get("builder") != EXPECTED_BUILDER:
        raise ValueError("v4 direction experiment requires historical-v2 causal builder")
    if result.get("dataset_id") != manifest.get("dataset_id"):
        raise ValueError("v4 direction dataset_id mismatch")
    if manifest.get("dataset_id") != EXPECTED_DATASET_ID:
        raise ValueError("v4 direction amendment is frozen to the canonical dataset")
    source = _mapping(manifest.get("source"), "manifest.source")
    if bool(source.get("partial")):
        raise ValueError("partial source history is not allowed")
    if expected["frozen_challenger"] != _expected_nonlinear_protocol_config():
        raise ValueError("committed nonlinear config drifted from v4 preregistration")

    gate = _mapping(expected.get("development_gate"), "development_gate")
    base_expected_gate = {
        "walk_forward_min_train_market_dates": MIN_TRAIN_MARKET_DATES,
        "walk_forward_validation_market_dates": VALIDATION_MARKET_DATES,
        "walk_forward_step_market_dates": STEP_MARKET_DATES,
        "embargo_minutes": EMBARGO_MINUTES,
        "expected_fold_count": BASE_EXPECTED_FOLD_COUNT,
        "minimum_brier_fold_wins_vs_linear": MINIMUM_BRIER_FOLD_WINS_VS_LINEAR,
        "minimum_brier_fold_wins_vs_null": MINIMUM_BRIER_FOLD_WINS_VS_NULL,
        "require_lower_mean_brier_than_linear": True,
        "require_lower_mean_brier_than_null": True,
        "require_lower_mean_log_loss_than_linear": True,
        "require_lower_mean_log_loss_than_null": True,
        "pnl_used": False,
    }
    if dict(gate) != base_expected_gate:
        raise ValueError("v4 direction base development gate drifted from preregistration")

    amendment = _load_amendment()
    return dict(manifest), dict(result), amendment


def run_experiment(dataset_dir: str | Path, output_path: str | Path) -> dict[str, Any]:
    root = Path(dataset_dir)
    manifest, protocol_result, amendment = _validate_v4_contract(root)
    train_path = root / "train.jsonl"
    if not train_path.is_file():
        raise FileNotFoundError(f"train split not found: {train_path}")

    # Fresh validation.jsonl and test.jsonl are intentionally never opened here.
    samples = _load_train(train_path)
    train_market_dates = sorted({sample.session_date for sample in samples})
    if len(train_market_dates) != EXPECTED_TRAIN_MARKET_DATES:
        raise ValueError(
            "v4 direction amendment requires exactly "
            f"{EXPECTED_TRAIN_MARKET_DATES} train market dates, got {len(train_market_dates)}"
        )

    folds = purged_walk_forward_folds(
        samples,
        min_train_market_dates=MIN_TRAIN_MARKET_DATES,
        validation_market_dates=VALIDATION_MARKET_DATES,
        step_market_dates=STEP_MARKET_DATES,
        embargo_minutes=EMBARGO_MINUTES,
        max_train_market_dates=None,
    )
    if len(folds) != EXPECTED_FOLD_COUNT:
        raise ValueError(
            f"amended v4 direction experiment requires exactly {EXPECTED_FOLD_COUNT} folds, got {len(folds)}"
        )

    fold_results: list[dict[str, Any]] = []
    for fold in folds:
        nonlinear_model, nonlinear_fit = fit_hist_gradient_direction(
            fold.train,
            config=NONLINEAR_CONFIG,
        )
        linear_model, linear_fit = fit_unified_linear_baseline(
            fold.train,
            config=LINEAR_CONFIG,
        )
        null_p_up = nonlinear_fit.training_up_rate
        nonlinear_predictions = forecast_hist_gradient_directions(
            nonlinear_model,
            fold.validation,
            include_actual=True,
        )
        linear_predictions = forecast_linear_directions(
            linear_model,
            fold.validation,
            include_actual=True,
        )
        nonlinear_metrics = evaluate_direction_predictions(
            nonlinear_predictions,
            null_p_up=null_p_up,
        )
        linear_metrics = evaluate_direction_predictions(
            linear_predictions,
            null_p_up=null_p_up,
        )

        nonlinear_brier = float(nonlinear_metrics["brier_score"])
        linear_brier = float(linear_metrics["brier_score"])
        null_brier = float(nonlinear_metrics["null_brier_score"])
        nonlinear_log_loss = float(nonlinear_metrics["log_loss"])
        linear_log_loss = float(linear_metrics["log_loss"])
        null_log_loss = float(nonlinear_metrics["null_log_loss"])
        fold_results.append(
            {
                "fold": fold.index,
                "train_market_dates": list(fold.train_market_dates),
                "validation_market_dates": list(fold.validation_market_dates),
                "train_samples": len(fold.train),
                "validation_samples": len(fold.validation),
                "purged_overlap_count": fold.purged_overlap_count,
                "embargoed_count": fold.embargoed_count,
                "directional_validation_samples": nonlinear_metrics["evaluated_count"],
                "nonlinear_fit_summary": nonlinear_fit.to_dict(),
                "linear_fit_summary": linear_fit.to_dict(),
                "null_p_up": null_p_up,
                "nonlinear_brier": nonlinear_brier,
                "linear_brier": linear_brier,
                "null_brier": null_brier,
                "delta_brier_vs_linear": nonlinear_brier - linear_brier,
                "delta_brier_vs_null": nonlinear_brier - null_brier,
                "nonlinear_log_loss": nonlinear_log_loss,
                "linear_log_loss": linear_log_loss,
                "null_log_loss": null_log_loss,
                "delta_log_loss_vs_linear": nonlinear_log_loss - linear_log_loss,
                "delta_log_loss_vs_null": nonlinear_log_loss - null_log_loss,
                "nonlinear_balanced_accuracy": nonlinear_metrics["balanced_accuracy"],
                "linear_balanced_accuracy": linear_metrics["balanced_accuracy"],
                "wins_brier_vs_linear": nonlinear_brier < linear_brier,
                "wins_brier_vs_null": nonlinear_brier < null_brier,
            }
        )

    wins_vs_linear = sum(bool(row["wins_brier_vs_linear"]) for row in fold_results)
    wins_vs_null = sum(bool(row["wins_brier_vs_null"]) for row in fold_results)
    mean_nonlinear_brier = fmean(row["nonlinear_brier"] for row in fold_results)
    mean_linear_brier = fmean(row["linear_brier"] for row in fold_results)
    mean_null_brier = fmean(row["null_brier"] for row in fold_results)
    mean_nonlinear_log_loss = fmean(row["nonlinear_log_loss"] for row in fold_results)
    mean_linear_log_loss = fmean(row["linear_log_loss"] for row in fold_results)
    mean_null_log_loss = fmean(row["null_log_loss"] for row in fold_results)

    supported = bool(
        wins_vs_linear >= MINIMUM_BRIER_FOLD_WINS_VS_LINEAR
        and wins_vs_null >= MINIMUM_BRIER_FOLD_WINS_VS_NULL
        and mean_nonlinear_brier < mean_linear_brier
        and mean_nonlinear_brier < mean_null_brier
        and mean_nonlinear_log_loss < mean_linear_log_loss
        and mean_nonlinear_log_loss < mean_null_log_loss
    )

    payload = {
        "schema_version": 1,
        "experiment": "v4_hist_gradient_direction_15m_v1_amended_fold_count",
        "dataset_id": manifest.get("dataset_id"),
        "dataset_builder": manifest.get("builder"),
        "protocol_name": protocol_result.get("protocol_name"),
        "protocol_sha256": protocol_result.get("protocol_sha256"),
        "development_amendment": {
            "amendment_name": amendment["amendment_name"],
            "path": str(AMENDMENT_PATH.as_posix()),
            "sha256": _canonical_json_sha256(amendment),
            "base_expected_fold_count": BASE_EXPECTED_FOLD_COUNT,
            "amended_expected_fold_count": EXPECTED_FOLD_COUNT,
            "created_before_model_outcomes": True,
        },
        "horizon_minutes": HORIZON_MINUTES,
        "direction_contract": "conditional_up_vs_down",
        "source": "train_split_only",
        "validation_used": False,
        "test_lockbox_accessed": False,
        "nonlinear_model": _expected_nonlinear_protocol_config(),
        "incumbent": preregistered_protocol()["linear_incumbent"],
        "null": "training_fold_directional_up_rate",
        "walk_forward": {
            "min_train_market_dates": MIN_TRAIN_MARKET_DATES,
            "validation_market_dates": VALIDATION_MARKET_DATES,
            "step_market_dates": STEP_MARKET_DATES,
            "embargo_minutes": EMBARGO_MINUTES,
            "expected_fold_count": EXPECTED_FOLD_COUNT,
            "manifest": walk_forward_manifest(folds),
        },
        "promotion_rule": {
            "minimum_brier_fold_wins_vs_linear": MINIMUM_BRIER_FOLD_WINS_VS_LINEAR,
            "minimum_brier_fold_wins_vs_null": MINIMUM_BRIER_FOLD_WINS_VS_NULL,
            "require_lower_mean_brier_than_linear": True,
            "require_lower_mean_brier_than_null": True,
            "require_lower_mean_log_loss_than_linear": True,
            "require_lower_mean_log_loss_than_null": True,
            "validation_split_used_for_tuning": False,
            "pnl_used": False,
        },
        "fold_results": fold_results,
        "summary": {
            "fold_count": len(fold_results),
            "brier_fold_wins_vs_linear": wins_vs_linear,
            "brier_fold_wins_vs_null": wins_vs_null,
            "mean_nonlinear_brier": mean_nonlinear_brier,
            "mean_linear_brier": mean_linear_brier,
            "mean_null_brier": mean_null_brier,
            "mean_delta_brier_vs_linear": mean_nonlinear_brier - mean_linear_brier,
            "mean_delta_brier_vs_null": mean_nonlinear_brier - mean_null_brier,
            "mean_nonlinear_log_loss": mean_nonlinear_log_loss,
            "mean_linear_log_loss": mean_linear_log_loss,
            "mean_null_log_loss": mean_null_log_loss,
            "mean_delta_log_loss_vs_linear": mean_nonlinear_log_loss - mean_linear_log_loss,
            "mean_delta_log_loss_vs_null": mean_nonlinear_log_loss - mean_null_log_loss,
            "nonlinear_direction_supported": supported,
        },
    }

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Run the amended train-only v4 nonlinear 15m direction challenger."
    )
    result.add_argument("--dataset-dir", required=True)
    result.add_argument(
        "--output",
        default="runs/first-real-direction-v4/nonlinear_direction_development_v1.json",
    )
    return result


def main() -> int:
    args = parser().parse_args()
    payload = run_experiment(args.dataset_dir, args.output)
    amendment = payload["development_amendment"]
    print(
        "Development amendment:",
        amendment["amendment_name"],
        f"({amendment['base_expected_fold_count']} -> {amendment['amended_expected_fold_count']} folds)",
    )
    for row in payload["fold_results"]:
        dates = row["validation_market_dates"]
        print(
            f"FOLD {row['fold']} {dates[0]} -> {dates[-1]} "
            f"brier nonlinear={row['nonlinear_brier']:.6f} "
            f"linear={row['linear_brier']:.6f} null={row['null_brier']:.6f} "
            f"d_linear={row['delta_brier_vs_linear']:+.6f} "
            f"d_null={row['delta_brier_vs_null']:+.6f}"
        )

    summary = payload["summary"]
    print()
    print("=== V4 NONLINEAR DIRECTION DEVELOPMENT ===")
    print("Dataset ID:", payload["dataset_id"])
    print("Brier fold wins vs linear:", f"{summary['brier_fold_wins_vs_linear']}/{summary['fold_count']}")
    print("Brier fold wins vs null:", f"{summary['brier_fold_wins_vs_null']}/{summary['fold_count']}")
    print("Mean nonlinear Brier:", summary["mean_nonlinear_brier"])
    print("Mean linear Brier:", summary["mean_linear_brier"])
    print("Mean null Brier:", summary["mean_null_brier"])
    print("Mean delta Brier vs linear:", summary["mean_delta_brier_vs_linear"])
    print("Mean delta Brier vs null:", summary["mean_delta_brier_vs_null"])
    print("Mean nonlinear log loss:", summary["mean_nonlinear_log_loss"])
    print("Mean linear log loss:", summary["mean_linear_log_loss"])
    print("Mean null log loss:", summary["mean_null_log_loss"])
    print("NONLINEAR DIRECTION SUPPORTED:", summary["nonlinear_direction_supported"])
    print("Report:", args.output)
    print("V4 validation: NOT ACCESSED")
    print("V4 test lockbox: NOT ACCESSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
