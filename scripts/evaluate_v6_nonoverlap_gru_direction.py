"""V6 preregistered nonoverlap GRU: development folds inside v4 TRAIN only.

Selection for GRU training uses fixed UTC 15-minute time anchors, applied
BEFORE any direction label is read. This eliminates outcome-tail selection
and adjacent overlapping 15m target intervals for each symbol/session.
The magnitude gate remains inference-only and uses known features.
Never opens global validation/test or context instrument targets.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import fmean

import numpy as np

from scripts.evaluate_magnitude_head import _load_train
from scripts.evaluate_v4_nonlinear_direction import (
    EMBARGO_MINUTES, EXPECTED_DATASET_ID, EXPECTED_FOLD_COUNT,
    EXPECTED_TRAIN_MARKET_DATES, HORIZON_MINUTES, LINEAR_CONFIG,
    MIN_TRAIN_MARKET_DATES, STEP_MARKET_DATES, VALIDATION_MARKET_DATES,
    _validate_v4_contract,
)
from scripts.evaluate_v4_selective_direction import (
    GATE_PREDICTION_QUANTILE, MAGNITUDE_CONFIG, _mean_actual_magnitude_bps,
    _up_rate,
)
from scripts.evaluate_v5_temporal_gru_direction import (
    _canonical_sha256, _validate_context_dataset, _validate_v5_protocol,
)
from trading.baseline import fit_unified_linear_baseline
from trading.cross_sectional import enrich_cross_sectional_context
from trading.direction import evaluate_direction_predictions, forecast_linear_directions
from trading.gru_direction import GRUDirectionConfig, fit_gru_direction, forecast_gru_directions
from trading.nonlinear import fit_hist_gradient_magnitude, forecast_hist_gradient_magnitudes
from trading.nonoverlap_selection import anchored_training_refs
from trading.temporal_sequences import (
    SECTOR_MAP, build_temporal_sequence_corpus, fit_sequence_scaler, sample_key,
)
from trading.temporal_state import enrich_temporal_state
from trading.validation import purged_walk_forward_folds


PROTOCOL_PATH = Path(__file__).resolve().parents[1] / "research" / "v6_nonoverlap_direction_protocol_v1.json"
SEQUENCE_LENGTH = 60
ANCHOR_MINUTES = 15
MINIMUM_WINS = 5
MINIMUM_COVERAGE = 0.10
EXPECTED_CONTEXT_ID = "3ce0425124fbda4e3fb3"
GRU_CONFIG = GRUDirectionConfig(
    horizon_minutes=15, hidden_size=32, num_layers=1, dropout=0.0,
    learning_rate=0.0003, weight_decay=0.001, epochs=2, batch_size=128,
    gradient_clip_norm=1.0, seed=42,
)


def _load_frozen_protocol() -> dict:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if protocol.get("protocol_name") != "ourai-v6-nonoverlap-direction-train-only-v1":
        raise ValueError("v6 protocol identity mismatch")
    datasets = protocol.get("datasets") or {}
    if datasets.get("target_dataset_id") != EXPECTED_DATASET_ID:
        raise ValueError("v6 target dataset identity mismatch")
    if datasets.get("context_dataset_id") != EXPECTED_CONTEXT_ID:
        raise ValueError("v6 context dataset identity mismatch")
    for flag in ("target_holdouts_accessed", "context_holdouts_accessed", "context_targets_used"):
        if datasets.get(flag) is not False:
            raise ValueError(f"v6 {flag} must be false")
    model = protocol.get("model") or {}
    frozen = {
        "horizon_minutes": GRU_CONFIG.horizon_minutes,
        "sequence_length": SEQUENCE_LENGTH,
        "hidden_size": GRU_CONFIG.hidden_size,
        "num_layers": GRU_CONFIG.num_layers,
        "dropout": GRU_CONFIG.dropout,
        "learning_rate": GRU_CONFIG.learning_rate,
        "weight_decay": GRU_CONFIG.weight_decay,
        "epochs": GRU_CONFIG.epochs,
        "batch_size": GRU_CONFIG.batch_size,
        "gradient_clip_norm": GRU_CONFIG.gradient_clip_norm,
        "seed": GRU_CONFIG.seed,
    }
    if model.get("family") != "GRU" or model.get("loss") != "BCEWithLogitsLoss":
        raise ValueError("v6 GRU model family/loss mismatch")
    if any(model.get(key) != value for key, value in frozen.items()):
        raise ValueError("v6 frozen model parameters drift")
    training = protocol.get("training") or {}
    if (training.get("anchor_modulus_minutes"), training.get("anchor_phase")) != (ANCHOR_MINUTES, 0):
        raise ValueError("v6 frozen time anchor drift")
    if training.get("no_label_selection_prior_to_time_anchors") is not True:
        raise ValueError("v6 training cannot select anchors by labels")
    if training.get("do_not_apply_magnitude_gate_to_training") is not True:
        raise ValueError("v6 training cannot use magnitude selection")
    gate = protocol.get("evaluation") or {}
    expected_gate = {
        "folds": EXPECTED_FOLD_COUNT,
        "train_market_dates_min": MIN_TRAIN_MARKET_DATES,
        "validation_market_dates": VALIDATION_MARKET_DATES,
        "step_market_dates": STEP_MARKET_DATES,
        "embargo_minutes": EMBARGO_MINUTES,
        "minimum_mean_coverage": MINIMUM_COVERAGE,
        "minimum_wins_over_linear": MINIMUM_WINS,
        "minimum_wins_over_null": MINIMUM_WINS,
        "pnl_used": False,
    }
    if any(gate.get(key) != value for key, value in expected_gate.items()):
        raise ValueError("v6 frozen development gate drift")
    if gate.get("lower_mean_brier_than_both") is not True or gate.get("lower_mean_log_loss_than_both") is not True:
        raise ValueError("v6 proper scoring requirements drift")
    if protocol.get("promotion", {}).get("validation_or_test_opened") is not False:
        raise ValueError("v6 global holdouts must stay closed")
    return protocol


def _fold_mean(rows: list[dict], key: str) -> float:
    return fmean(float(row[key]) for row in rows)


def run_experiment(
    target_dataset_dir: str | Path, context_dataset_dir: str | Path,
    output_path: str | Path, *, device: str | None = None,
) -> dict:
    target_root, context_root = Path(target_dataset_dir), Path(context_dataset_dir)
    manifest, _, _ = _validate_v4_contract(target_root)
    parent_protocol, _ = _validate_v5_protocol()
    protocol = _load_frozen_protocol()
    context_build = _validate_context_dataset(context_root, parent_protocol)
    if context_build.get("context_dataset_id") != EXPECTED_CONTEXT_ID:
        raise ValueError("canonical context dataset mismatch")

    target_train_path = target_root / "train.jsonl"
    context_train_path = context_root / "train.jsonl"
    if not target_train_path.is_file() or not context_train_path.is_file():
        raise FileNotFoundError("both train-only splits are required")
    targets = _load_train(target_train_path)
    contexts = _load_train(context_train_path)
    target_dates = sorted({sample.session_date for sample in targets})
    if len(target_dates) != EXPECTED_TRAIN_MARKET_DATES:
        raise ValueError("v6 expected 174 target training market dates")
    if target_dates != sorted({sample.session_date for sample in contexts}):
        raise ValueError("v6 target and context train dates differ")
    print("Building causal 60-observation corpus (train splits only)...")
    corpus = build_temporal_sequence_corpus(
        targets, contexts, sequence_length=SEQUENCE_LENGTH, sector_map=SECTOR_MAP,
    )
    print("Sequence-eligible samples:", len(corpus.ref_by_key))
    state_samples = enrich_cross_sectional_context(enrich_temporal_state(targets))
    original_folds = purged_walk_forward_folds(
        targets, min_train_market_dates=MIN_TRAIN_MARKET_DATES,
        validation_market_dates=VALIDATION_MARKET_DATES, step_market_dates=STEP_MARKET_DATES,
        embargo_minutes=EMBARGO_MINUTES, max_train_market_dates=None,
    )
    state_folds = purged_walk_forward_folds(
        state_samples, min_train_market_dates=MIN_TRAIN_MARKET_DATES,
        validation_market_dates=VALIDATION_MARKET_DATES, step_market_dates=STEP_MARKET_DATES,
        embargo_minutes=EMBARGO_MINUTES, max_train_market_dates=None,
    )
    if len(original_folds) != EXPECTED_FOLD_COUNT or len(state_folds) != EXPECTED_FOLD_COUNT:
        raise ValueError("unexpected v6 fold count")
    original_lookup = {sample_key(sample): sample for sample in targets}
    results: list[dict] = []
    for original_fold, state_fold in zip(original_folds, state_folds):
        if original_fold.validation_market_dates != state_fold.validation_market_dates:
            raise AssertionError("v6 enrichment shifted validation dates")
        # All model-selection machinery remains within the current training fold.
        prior = _up_rate(original_fold.train)
        magnitude, _ = fit_hist_gradient_magnitude(state_fold.train, config=MAGNITUDE_CONFIG)
        train_mag = forecast_hist_gradient_magnitudes(
            magnitude, state_fold.train, include_actual=False,
        )
        threshold = float(np.quantile(
            [pred.expected_magnitude_bps for pred in train_mag], GATE_PREDICTION_QUANTILE,
        ))
        train_keys = [sample_key(sample) for sample in original_fold.train]
        scaler = fit_sequence_scaler(corpus, train_keys)

        # Anchor selection occurs before direction/realized-magnitude label lookup.
        eligible_refs = [
            corpus.ref_by_key[sample_key(sample)]
            for sample in original_fold.train
            if sample_key(sample) in corpus.ref_by_key
        ]
        anchors = anchored_training_refs(eligible_refs)
        if len(anchors) < 100:
            raise ValueError("too few outcome-independent training anchors")
        dates = state_fold.validation_market_dates
        print(
            f"FOLD {state_fold.index} {dates[0]} -> {dates[-1]} "
            f"| {len(anchors)} fixed time anchors (before neutral exclusion)"
        )
        gru, fit = fit_gru_direction(
            corpus, anchors, scaler=scaler, config=GRU_CONFIG,
            device=device, progress_prefix=f"fold {state_fold.index}",
        )
        linear, _ = fit_unified_linear_baseline(original_fold.train, config=LINEAR_CONFIG)
        dev_magnitude = forecast_hist_gradient_magnitudes(
            magnitude, state_fold.validation, include_actual=False,
        )
        selected_state = [
            sample for sample, estimate in zip(state_fold.validation, dev_magnitude)
            if float(estimate.expected_magnitude_bps) >= threshold
        ]
        selected_refs = [
            corpus.ref_by_key[sample_key(sample)]
            for sample in selected_state if sample_key(sample) in corpus.ref_by_key
        ]
        if not selected_refs:
            raise ValueError("inference-only magnitude gate selected zero eligible rows")
        selected_original = [original_lookup[sample_key(ref.sample)] for ref in selected_refs]
        gru_metrics = evaluate_direction_predictions(
            forecast_gru_directions(gru, selected_refs, include_actual=True),
            null_p_up=prior,
        )
        linear_metrics = evaluate_direction_predictions(
            forecast_linear_directions(linear, selected_original, include_actual=True),
            null_p_up=prior,
        )
        if gru_metrics["evaluated_count"] != linear_metrics["evaluated_count"]:
            raise AssertionError("GRU and linear comparison must use identical rows")
        coverage = len(selected_refs) / len(state_fold.validation)
        enrichment = (
            _mean_actual_magnitude_bps([ref.sample for ref in selected_refs])
            / _mean_actual_magnitude_bps(state_fold.validation)
        )
        row = {
            "fold": state_fold.index,
            "validation_market_dates": list(dates),
            "train_fixed_anchor_count": len(anchors),
            "gru_training_sequences": fit.training_sequences,
            "gru_train_up_fraction": fit.up_rate,
            "gru_epoch_losses": list(fit.epoch_losses),
            "prior_up_probability": prior,
            "gate_threshold_bps": threshold,
            "selected_samples": len(selected_refs),
            "selected_directional_samples": int(gru_metrics["evaluated_count"]),
            "selected_coverage": coverage,
            "magnitude_enrichment_ratio": enrichment,
            "gru_brier": float(gru_metrics["brier_score"]),
            "linear_brier": float(linear_metrics["brier_score"]),
            "null_brier": float(gru_metrics["null_brier_score"]),
            "gru_log_loss": float(gru_metrics["log_loss"]),
            "linear_log_loss": float(linear_metrics["log_loss"]),
            "null_log_loss": float(gru_metrics["null_log_loss"]),
        }
        row["win_vs_linear"] = row["gru_brier"] < row["linear_brier"]
        row["win_vs_null"] = row["gru_brier"] < row["null_brier"]
        results.append(row)
        print(
            f"FOLD {row['fold']} anchored_directional={fit.training_sequences} "
            f"coverage={coverage:.3f} magnitude_x={enrichment:.3f} "
            f"GRU={row['gru_brier']:.6f} linear={row['linear_brier']:.6f} null={row['null_brier']:.6f}"
        )

    summary = {
        "fold_count": len(results),
        "brier_wins_vs_linear": sum(row["win_vs_linear"] for row in results),
        "brier_wins_vs_null": sum(row["win_vs_null"] for row in results),
        "mean_training_sequences": _fold_mean(results, "gru_training_sequences"),
        "mean_coverage": _fold_mean(results, "selected_coverage"),
        "mean_magnitude_enrichment": _fold_mean(results, "magnitude_enrichment_ratio"),
        "mean_gru_brier": _fold_mean(results, "gru_brier"),
        "mean_linear_brier": _fold_mean(results, "linear_brier"),
        "mean_null_brier": _fold_mean(results, "null_brier"),
        "mean_gru_log_loss": _fold_mean(results, "gru_log_loss"),
        "mean_linear_log_loss": _fold_mean(results, "linear_log_loss"),
        "mean_null_log_loss": _fold_mean(results, "null_log_loss"),
        "eligible_for_global_validation_without_new_freeze": False,
    }
    summary["supported"] = bool(
        summary["brier_wins_vs_linear"] >= MINIMUM_WINS
        and summary["brier_wins_vs_null"] >= MINIMUM_WINS
        and summary["mean_gru_brier"] < summary["mean_linear_brier"]
        and summary["mean_gru_brier"] < summary["mean_null_brier"]
        and summary["mean_gru_log_loss"] < summary["mean_linear_log_loss"]
        and summary["mean_gru_log_loss"] < summary["mean_null_log_loss"]
        and summary["mean_coverage"] >= MINIMUM_COVERAGE
    )
    payload = {
        "schema_version": 1,
        "experiment": "v6_outcome_independent_anchor_gru_direction_train_only_v1",
        "dataset_id": manifest["dataset_id"],
        "context_dataset_id": context_build["context_dataset_id"],
        "protocol_name": protocol["protocol_name"],
        "protocol_sha256": _canonical_sha256(protocol),
        "parent_v5_protocol_name": parent_protocol["protocol_name"],
        "source": "train_splits_only",
        "validation_used": False,
        "test_lockbox_accessed": False,
        "context_validation_used": False,
        "context_test_used": False,
        "context_targets_used": False,
        "pnl_used": False,
        "training_selection": "fixed UTC minute modulo 15 == 0, before label filtering",
        "fold_results": results,
        "summary": summary,
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Run frozen v6 anchored GRU, train-only development.")
    parser.add_argument("--target-dataset-dir", required=True)
    parser.add_argument("--context-dataset-dir", required=True)
    parser.add_argument("--output", default="runs/first-real-direction-v6/nonoverlap_gru_development_v1.json")
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)
    args = parser.parse_args()
    payload = run_experiment(
        args.target_dataset_dir, args.context_dataset_dir,
        args.output, device=args.device,
    )
    s = payload["summary"]
    print("\n=== V6 OUTCOME-INDEPENDENT ANCHORED GRU DEVELOPMENT ===")
    print("Brier fold wins vs linear:", f"{s['brier_wins_vs_linear']}/{s['fold_count']}")
    print("Brier fold wins vs null:", f"{s['brier_wins_vs_null']}/{s['fold_count']}")
    print("Mean GRU / linear / null Brier:", s["mean_gru_brier"], "/", s["mean_linear_brier"], "/", s["mean_null_brier"])
    print("Mean GRU / linear / null log loss:", s["mean_gru_log_loss"], "/", s["mean_linear_log_loss"], "/", s["mean_null_log_loss"])
    print("Mean selected coverage:", s["mean_coverage"])
    print("Mean magnitude enrichment:", s["mean_magnitude_enrichment"])
    print("V6 NONOVERLAP DIRECTION SUPPORTED:", s["supported"])
    print("V4 validation: NOT ACCESSED")
    print("V4 test lockbox: NOT ACCESSED")
    print("Eligible for global validation: NO - separate freeze required")
    print("Report:", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
