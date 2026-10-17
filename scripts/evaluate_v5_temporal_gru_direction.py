"""Train-only v5 GRU direction development with market/sector context.

Only the v4 target train split and v5 market-context train split are opened. The
frozen 60-observation GRU receives target-stock features plus same-timestamp SPY,
QQQ and sector-ETF features. The magnitude selector is carried forward unchanged
from v4 selective development. No validation/test split and no PnL is used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from statistics import fmean
from typing import Any, Mapping

import numpy as np

from scripts.evaluate_magnitude_head import _load_train
from scripts.evaluate_v4_nonlinear_direction import (
    EMBARGO_MINUTES,
    EXPECTED_DATASET_ID,
    EXPECTED_FOLD_COUNT,
    EXPECTED_TRAIN_MARKET_DATES,
    HORIZON_MINUTES,
    LINEAR_CONFIG,
    MIN_TRAIN_MARKET_DATES,
    STEP_MARKET_DATES,
    VALIDATION_MARKET_DATES,
    _validate_v4_contract,
)
from scripts.evaluate_v4_selective_direction import (
    GATE_PREDICTION_QUANTILE,
    MAGNITUDE_CONFIG,
    TAIL_QUANTILE,
    _mean_actual_magnitude_bps,
    _resolved_magnitudes_bps,
    _tail_directional_samples,
    _up_rate,
)
from trading.baseline import fit_unified_linear_baseline
from trading.cross_sectional import enrich_cross_sectional_context
from trading.direction import evaluate_direction_predictions, forecast_linear_directions
from trading.gru_direction import (
    GRUDirectionConfig,
    fit_gru_direction,
    forecast_gru_directions,
)
from trading.nonlinear import fit_hist_gradient_magnitude, forecast_hist_gradient_magnitudes
from trading.temporal_sequences import (
    SECTOR_MAP,
    build_temporal_sequence_corpus,
    fit_sequence_scaler,
    sample_key,
)
from trading.temporal_state import enrich_temporal_state
from trading.validation import purged_walk_forward_folds


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "research" / "v5_temporal_market_context_protocol_v1.json"
AMENDMENT_PATH = ROOT / "research" / "v5_temporal_gru_implementation_amendment_v1.json"
SEQUENCE_LENGTH = 60
MINIMUM_WINS = 5
MINIMUM_MEAN_COVERAGE = 0.10

GRU_CONFIG = GRUDirectionConfig(
    horizon_minutes=15,
    hidden_size=64,
    num_layers=2,
    dropout=0.1,
    learning_rate=0.001,
    weight_decay=0.0001,
    epochs=5,
    batch_size=256,
    gradient_clip_norm=1.0,
    seed=42,
)


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_json_object(path: Path, name: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"required {name} not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{name} must contain a JSON object")
    return payload


def _validate_v5_protocol() -> tuple[dict[str, Any], dict[str, Any]]:
    protocol = _load_json_object(PROTOCOL_PATH, "v5 protocol")
    amendment = _load_json_object(AMENDMENT_PATH, "v5 implementation amendment")
    if protocol.get("protocol_name") != "ourai-v5-temporal-market-context-direction-v1":
        raise ValueError("unexpected v5 protocol")
    if amendment.get("amendment_name") != "ourai-v5-temporal-gru-implementation-amendment-v1":
        raise ValueError("unexpected v5 implementation amendment")
    if amendment.get("parent_protocol") != protocol.get("protocol_name"):
        raise ValueError("v5 implementation amendment parent mismatch")

    candidate = protocol.get("temporal_candidate") or {}
    expected_candidate = {
        "family": "GRU",
        "horizon_minutes": 15,
        "sequence_length_observations": 60,
        "hidden_size": 64,
        "num_layers": 2,
        "dropout": 0.1,
        "optimizer": "AdamW",
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "epochs": 5,
        "batch_size": 256,
        "gradient_clip_norm": 1.0,
        "seed": 42,
    }
    for key, expected in expected_candidate.items():
        if candidate.get(key) != expected:
            raise ValueError(f"v5 temporal candidate drifted at {key}")

    gate = protocol.get("development") or {}
    expected_gate = {
        "walk_forward_min_train_market_dates": MIN_TRAIN_MARKET_DATES,
        "walk_forward_validation_market_dates": VALIDATION_MARKET_DATES,
        "walk_forward_step_market_dates": STEP_MARKET_DATES,
        "embargo_minutes": EMBARGO_MINUTES,
        "expected_fold_count": EXPECTED_FOLD_COUNT,
        "minimum_brier_fold_wins_vs_original_linear": MINIMUM_WINS,
        "minimum_brier_fold_wins_vs_tail_null": MINIMUM_WINS,
        "minimum_mean_selected_coverage": MINIMUM_MEAN_COVERAGE,
        "pnl_used": False,
    }
    for key, expected in expected_gate.items():
        if gate.get(key) != expected:
            raise ValueError(f"v5 development gate drifted at {key}")

    data_access = amendment.get("data_access") or {}
    if data_access.get("v4_validation_accessed") is not False:
        raise ValueError("v5 amendment must keep v4 validation unopened")
    if data_access.get("v4_test_accessed") is not False:
        raise ValueError("v5 amendment must keep v4 test unopened")
    if data_access.get("context_validation_accessed") is not False:
        raise ValueError("v5 amendment must keep context validation unopened")
    if data_access.get("context_test_accessed") is not False:
        raise ValueError("v5 amendment must keep context test unopened")
    return protocol, amendment


def _validate_context_dataset(root: Path, protocol: Mapping[str, Any]) -> dict[str, Any]:
    manifest = _load_json_object(root / "manifest.json", "context manifest")
    result = _load_json_object(root / "v5_market_context_result.json", "context build result")
    protocol_copy = _load_json_object(root / "v5_temporal_market_context_protocol.json", "context protocol copy")
    if protocol_copy != protocol:
        raise ValueError("context dataset protocol copy differs from committed protocol")
    if result.get("protocol_name") != protocol.get("protocol_name"):
        raise ValueError("context dataset protocol identity mismatch")
    if result.get("protocol_sha256") != _canonical_sha256(protocol):
        raise ValueError("context dataset protocol hash mismatch")
    if result.get("base_target_dataset_id") != EXPECTED_DATASET_ID:
        raise ValueError("context dataset base target identity mismatch")
    if result.get("context_dataset_id") != manifest.get("dataset_id"):
        raise ValueError("context dataset_id mismatch")
    if result.get("source_partial") is not False:
        raise ValueError("partial context history is not allowed")
    if result.get("context_targets_used_by_direction_research") is not False:
        raise ValueError("context targets must not be used")
    if result.get("v4_validation_accessed") is not False or result.get("v4_test_accessed") is not False:
        raise ValueError("context build reports v4 holdout access")
    if result.get("market_date_count") != 249:
        raise ValueError("unexpected context market-date count")
    if result.get("split_market_date_counts") != {"train": 174, "validation": 37, "test": 38}:
        raise ValueError("context split dates do not align with v4")
    return result


def _direction(sample) -> str | None:
    target = sample.targets.get(f"{HORIZON_MINUTES}m")
    if not isinstance(target, Mapping):
        return None
    value = str(target.get("direction") or "").strip().upper()
    return value if value in {"UP", "DOWN"} else None


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    return fmean(float(row[key]) for row in rows)


def run_experiment(
    target_dataset_dir: str | Path,
    context_dataset_dir: str | Path,
    output_path: str | Path,
    *,
    device: str | None = None,
) -> dict[str, Any]:
    target_root = Path(target_dataset_dir)
    context_root = Path(context_dataset_dir)
    manifest, protocol_result, amendment_v4 = _validate_v4_contract(target_root)
    protocol, amendment = _validate_v5_protocol()
    context_result = _validate_context_dataset(context_root, protocol)

    target_train_path = target_root / "train.jsonl"
    context_train_path = context_root / "train.jsonl"
    if not target_train_path.is_file() or not context_train_path.is_file():
        raise FileNotFoundError("v5 development requires both train.jsonl files")

    # No validation.jsonl or test.jsonl file is opened anywhere in this experiment.
    target_samples = _load_train(target_train_path)
    context_samples = _load_train(context_train_path)
    target_dates = sorted({sample.session_date for sample in target_samples})
    context_dates = sorted({sample.session_date for sample in context_samples})
    if len(target_dates) != EXPECTED_TRAIN_MARKET_DATES:
        raise ValueError("unexpected v4 train date count")
    if target_dates != context_dates:
        raise ValueError("target and market-context train dates are not aligned")

    print("Building causal 60-observation sequence corpus...")
    corpus = build_temporal_sequence_corpus(
        target_samples,
        context_samples,
        sequence_length=SEQUENCE_LENGTH,
        sector_map=SECTOR_MAP,
    )
    print("Sequence-eligible samples:", len(corpus.ref_by_key))
    print("Raw features per observation:", corpus.raw_feature_count)
    print("Model features incl. missing masks:", corpus.model_feature_count)

    state_samples = enrich_cross_sectional_context(enrich_temporal_state(target_samples))
    original_folds = purged_walk_forward_folds(
        target_samples,
        min_train_market_dates=MIN_TRAIN_MARKET_DATES,
        validation_market_dates=VALIDATION_MARKET_DATES,
        step_market_dates=STEP_MARKET_DATES,
        embargo_minutes=EMBARGO_MINUTES,
        max_train_market_dates=None,
    )
    state_folds = purged_walk_forward_folds(
        state_samples,
        min_train_market_dates=MIN_TRAIN_MARKET_DATES,
        validation_market_dates=VALIDATION_MARKET_DATES,
        step_market_dates=STEP_MARKET_DATES,
        embargo_minutes=EMBARGO_MINUTES,
        max_train_market_dates=None,
    )
    if len(original_folds) != EXPECTED_FOLD_COUNT or len(state_folds) != EXPECTED_FOLD_COUNT:
        raise ValueError("v5 GRU fold count mismatch")

    original_by_key = {sample_key(sample): sample for sample in target_samples}
    rows: list[dict[str, Any]] = []

    for original_fold, state_fold in zip(original_folds, state_folds):
        if original_fold.validation_market_dates != state_fold.validation_market_dates:
            raise AssertionError("state enrichment changed v5 fold boundaries")

        train_magnitudes = _resolved_magnitudes_bps(state_fold.train)
        actual_tail_threshold_bps = float(np.quantile(train_magnitudes, TAIL_QUANTILE))
        tail_train = _tail_directional_samples(state_fold.train, actual_tail_threshold_bps)
        tail_null_p_up = _up_rate(tail_train)

        magnitude_model, _ = fit_hist_gradient_magnitude(
            state_fold.train,
            config=MAGNITUDE_CONFIG,
        )
        train_magnitude_predictions = forecast_hist_gradient_magnitudes(
            magnitude_model,
            state_fold.train,
            include_actual=False,
        )
        predicted_gate_threshold_bps = float(
            np.quantile(
                [prediction.expected_magnitude_bps for prediction in train_magnitude_predictions],
                GATE_PREDICTION_QUANTILE,
            )
        )

        train_keys = [sample_key(sample) for sample in original_fold.train]
        scaler = fit_sequence_scaler(corpus, train_keys)
        tail_refs = [
            corpus.ref_by_key[sample_key(sample)]
            for sample in tail_train
            if sample_key(sample) in corpus.ref_by_key
        ]
        if len(tail_refs) < 100:
            raise ValueError("too few tail sequences for frozen GRU training")

        print()
        print(
            f"FOLD {state_fold.index} {state_fold.validation_market_dates[0]} -> "
            f"{state_fold.validation_market_dates[-1]} | training GRU on {len(tail_refs)} sequences"
        )
        gru_model, gru_fit = fit_gru_direction(
            corpus,
            tail_refs,
            scaler=scaler,
            config=GRU_CONFIG,
            device=device,
            progress_prefix=f"fold {state_fold.index}",
        )
        original_model, _ = fit_unified_linear_baseline(
            original_fold.train,
            config=LINEAR_CONFIG,
        )

        validation_magnitude_predictions = forecast_hist_gradient_magnitudes(
            magnitude_model,
            state_fold.validation,
            include_actual=False,
        )
        selected_state = [
            sample
            for sample, prediction in zip(state_fold.validation, validation_magnitude_predictions)
            if float(prediction.expected_magnitude_bps) >= predicted_gate_threshold_bps
        ]
        selected_refs = [
            corpus.ref_by_key[sample_key(sample)]
            for sample in selected_state
            if sample_key(sample) in corpus.ref_by_key
        ]
        selected_original = [
            original_by_key[sample_key(ref.sample)]
            for ref in selected_refs
        ]
        if not selected_refs:
            raise ValueError("v5 magnitude gate plus sequence requirement selected no rows")

        gru_predictions = forecast_gru_directions(gru_model, selected_refs, include_actual=True)
        original_predictions = forecast_linear_directions(
            original_model,
            selected_original,
            include_actual=True,
        )
        gru_metrics = evaluate_direction_predictions(gru_predictions, null_p_up=tail_null_p_up)
        original_metrics = evaluate_direction_predictions(original_predictions, null_p_up=tail_null_p_up)
        if int(gru_metrics["evaluated_count"]) != int(original_metrics["evaluated_count"]):
            raise AssertionError("GRU and incumbent were not evaluated on the same directional rows")

        coverage = len(selected_refs) / len(state_fold.validation)
        selected_mean_magnitude = _mean_actual_magnitude_bps([ref.sample for ref in selected_refs])
        overall_mean_magnitude = _mean_actual_magnitude_bps(state_fold.validation)
        enrichment = selected_mean_magnitude / overall_mean_magnitude

        row = {
            "fold": state_fold.index,
            "validation_market_dates": list(state_fold.validation_market_dates),
            "device": gru_fit.device,
            "gru_training_sequences": gru_fit.training_sequences,
            "gru_input_features": gru_fit.input_features,
            "gru_epoch_losses": list(gru_fit.epoch_losses),
            "actual_tail_threshold_bps": actual_tail_threshold_bps,
            "predicted_gate_threshold_bps": predicted_gate_threshold_bps,
            "selected_samples": len(selected_refs),
            "validation_samples": len(state_fold.validation),
            "selected_coverage": coverage,
            "selected_directional_samples": int(gru_metrics["evaluated_count"]),
            "magnitude_enrichment_ratio": enrichment,
            "gru_brier": float(gru_metrics["brier_score"]),
            "original_linear_brier_on_selected": float(original_metrics["brier_score"]),
            "tail_null_brier": float(gru_metrics["null_brier_score"]),
            "gru_log_loss": float(gru_metrics["log_loss"]),
            "original_linear_log_loss_on_selected": float(original_metrics["log_loss"]),
            "tail_null_log_loss": float(gru_metrics["null_log_loss"]),
            "gru_balanced_accuracy": gru_metrics["balanced_accuracy"],
            "original_balanced_accuracy_on_selected": original_metrics["balanced_accuracy"],
            "wins_brier_vs_original": float(gru_metrics["brier_score"]) < float(original_metrics["brier_score"]),
            "wins_brier_vs_null": float(gru_metrics["brier_score"]) < float(gru_metrics["null_brier_score"]),
        }
        rows.append(row)
        print(
            f"FOLD {row['fold']} coverage={coverage:.3f} magnitude_x={enrichment:.3f} "
            f"GRU={row['gru_brier']:.6f} original={row['original_linear_brier_on_selected']:.6f} "
            f"null={row['tail_null_brier']:.6f}"
        )

    wins_original = sum(bool(row["wins_brier_vs_original"]) for row in rows)
    wins_null = sum(bool(row["wins_brier_vs_null"]) for row in rows)
    mean_gru_brier = _mean(rows, "gru_brier")
    mean_original_brier = _mean(rows, "original_linear_brier_on_selected")
    mean_null_brier = _mean(rows, "tail_null_brier")
    mean_gru_log_loss = _mean(rows, "gru_log_loss")
    mean_original_log_loss = _mean(rows, "original_linear_log_loss_on_selected")
    mean_null_log_loss = _mean(rows, "tail_null_log_loss")
    mean_coverage = _mean(rows, "selected_coverage")
    mean_enrichment = _mean(rows, "magnitude_enrichment_ratio")

    supported = bool(
        wins_original >= MINIMUM_WINS
        and wins_null >= MINIMUM_WINS
        and mean_gru_brier < mean_original_brier
        and mean_gru_brier < mean_null_brier
        and mean_gru_log_loss < mean_original_log_loss
        and mean_gru_log_loss < mean_null_log_loss
        and mean_coverage >= MINIMUM_MEAN_COVERAGE
    )

    payload = {
        "schema_version": 1,
        "experiment": "v5_temporal_market_context_gru_direction_train_only_v1",
        "dataset_id": manifest.get("dataset_id"),
        "context_dataset_id": context_result.get("context_dataset_id"),
        "protocol_name": protocol.get("protocol_name"),
        "protocol_sha256": _canonical_sha256(protocol),
        "implementation_amendment": amendment.get("amendment_name"),
        "implementation_amendment_sha256": _canonical_sha256(amendment),
        "v4_development_amendment": amendment_v4.get("amendment_name"),
        "horizon_minutes": HORIZON_MINUTES,
        "source": "train_splits_only",
        "validation_used": False,
        "test_lockbox_accessed": False,
        "context_validation_used": False,
        "context_test_used": False,
        "context_targets_used": False,
        "pnl_used": False,
        "fold_results": rows,
        "summary": {
            "fold_count": len(rows),
            "brier_wins_vs_original_linear": wins_original,
            "brier_wins_vs_tail_null": wins_null,
            "mean_selected_coverage": mean_coverage,
            "mean_magnitude_enrichment_ratio": mean_enrichment,
            "mean_gru_brier": mean_gru_brier,
            "mean_original_linear_brier_on_selected": mean_original_brier,
            "mean_tail_null_brier": mean_null_brier,
            "mean_gru_log_loss": mean_gru_log_loss,
            "mean_original_linear_log_loss_on_selected": mean_original_log_loss,
            "mean_tail_null_log_loss": mean_null_log_loss,
            "supported": supported,
            "eligible_for_validation_without_separate_freeze": False,
        },
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Run frozen v5 train-only temporal GRU direction development.")
    result.add_argument("--target-dataset-dir", required=True)
    result.add_argument("--context-dataset-dir", required=True)
    result.add_argument(
        "--output",
        default="runs/first-real-direction-v5/temporal_gru_development_v1.json",
    )
    result.add_argument("--device", choices=("cpu", "cuda"), default=None)
    return result


def main() -> int:
    args = parser().parse_args()
    payload = run_experiment(
        args.target_dataset_dir,
        args.context_dataset_dir,
        args.output,
        device=args.device,
    )
    summary = payload["summary"]
    print()
    print("=== V5 TEMPORAL MARKET-CONTEXT GRU DEVELOPMENT ===")
    print("Dataset ID:", payload["dataset_id"])
    print("Context dataset ID:", payload["context_dataset_id"])
    print("Brier fold wins vs original linear:", f"{summary['brier_wins_vs_original_linear']}/{summary['fold_count']}")
    print("Brier fold wins vs tail null:", f"{summary['brier_wins_vs_tail_null']}/{summary['fold_count']}")
    print("Mean selected coverage:", summary["mean_selected_coverage"])
    print("Mean magnitude enrichment ratio:", summary["mean_magnitude_enrichment_ratio"])
    print("Mean GRU Brier:", summary["mean_gru_brier"])
    print("Mean original Brier on selected:", summary["mean_original_linear_brier_on_selected"])
    print("Mean tail null Brier:", summary["mean_tail_null_brier"])
    print("Mean GRU log loss:", summary["mean_gru_log_loss"])
    print("Mean original log loss on selected:", summary["mean_original_linear_log_loss_on_selected"])
    print("Mean tail null log loss:", summary["mean_tail_null_log_loss"])
    print("V5 TEMPORAL DIRECTION SUPPORTED:", summary["supported"])
    print("V4 validation: NOT ACCESSED")
    print("V4 test lockbox: NOT ACCESSED")
    print("Eligible for validation now: NO - separate freeze required")
    print("Report:", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
