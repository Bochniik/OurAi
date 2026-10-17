"""Prepare the frozen v3 magnitude champion for future-forward shadow evaluation.

This command never opens the historical v3 test split. It verifies the already-frozen
nonlinear champion and the committed forward protocol, then fits and serializes the
fixed linear comparator on v3 train + validation. The training median magnitude from
that same source is frozen as the constant MAE null comparator.
"""

from __future__ import annotations

import argparse
import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

from scripts.evaluate_magnitude_head import _load_train
from scripts.evaluate_v3_nonlinear_magnitude import LINEAR_CONFIG, NONLINEAR_CONFIG
from scripts.validate_magnitude_head import _load_validation
from trading.forward_shadow import canonical_json_sha256, sha256_file
from trading.magnitude import fit_unified_linear_magnitude_baseline
from trading.nonlinear import UnifiedHistGradientMagnitude


EXPECTED_PROTOCOL_NAME = "ourai-v3-15m-magnitude-future-forward-v2"
ORIGINAL_PROTOCOL_PATH = Path(__file__).resolve().parents[1] / "research" / "v3_15m_magnitude_forward_protocol.json"
V2_FORWARD_START = "2026-10-19T13:30:00Z"


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _parse_aware(value: object, name: str) -> datetime:
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")
    return parsed


def _expected_champion_protocol_config() -> dict[str, Any]:
    return {
        "family": "HistGradientBoostingRegressor",
        "loss": "absolute_error",
        "learning_rate": NONLINEAR_CONFIG.learning_rate,
        "max_iter": NONLINEAR_CONFIG.max_iter,
        "max_leaf_nodes": NONLINEAR_CONFIG.max_leaf_nodes,
        "min_samples_leaf": NONLINEAR_CONFIG.min_samples_leaf,
        "l2_regularization": NONLINEAR_CONFIG.l2_regularization,
        "max_bins": NONLINEAR_CONFIG.max_bins,
        "early_stopping": False,
        "random_state": NONLINEAR_CONFIG.random_state,
        "training_source": "v3_train_plus_validation_only",
        "historical_test_in_training": False,
        "updates_during_forward_period": False,
    }


def _validate_protocol(protocol: Mapping[str, Any]) -> None:
    if protocol.get("protocol_name") != EXPECTED_PROTOCOL_NAME:
        raise ValueError("unexpected forward protocol identity")
    if protocol.get("frozen_before_forward_start") is not True:
        raise ValueError("forward protocol must be frozen before start")
    _parse_aware(protocol.get("forward_start_utc"), "forward_start_utc")
    if protocol.get("forward_start_utc") != V2_FORWARD_START:
        raise ValueError("v2 requires a preregistered future-only shadow start")
    if protocol.get("supersedes_protocol_name") != "ourai-v3-15m-magnitude-future-forward-v1":
        raise ValueError("unexpected v2 parent protocol identity")
    if protocol.get("prediction_logging", {}).get("never_backfill_before_forward_start") is not True:
        raise ValueError("v2 forward shadow must never backfill")
    original = json.loads(ORIGINAL_PROTOCOL_PATH.read_text(encoding="utf-8"))
    allowed_changes = {"protocol_name", "forward_start_utc", "supersedes_protocol_name", "revision_reason"}
    before = {key: value for key, value in original.items() if key not in allowed_changes}
    after = {key: value for key, value in protocol.items() if key not in allowed_changes}
    revised_logging = dict(after["prediction_logging"])
    revised_logging.pop("never_backfill_before_forward_start", None)
    after["prediction_logging"] = revised_logging
    if before != after:
        raise ValueError("v2 protocol must preserve all champion/evaluation/safety fields from v1")
    if str(protocol.get("session")) != "regular":
        raise ValueError("v3 forward protocol requires regular session")
    if str(protocol.get("bar_timeframe")) != "1Min":
        raise ValueError("v3 forward protocol requires 1Min bars")
    if float(protocol.get("bar_availability_delay_minutes", -1)) != 1.0:
        raise ValueError("v3 forward protocol requires one-minute bar availability delay")
    if int(protocol.get("horizon_minutes", -1)) != NONLINEAR_CONFIG.horizon_minutes:
        raise ValueError("forward horizon does not match frozen champion")
    champion = dict(_mapping(protocol.get("champion"), "protocol.champion"))
    if champion != _expected_champion_protocol_config():
        raise ValueError("forward protocol champion configuration does not match frozen model")
    logging = _mapping(protocol.get("prediction_logging"), "protocol.prediction_logging")
    if logging.get("prediction_must_be_written_before_target_resolution") is not True:
        raise ValueError("forward predictions must precede target resolution")
    if logging.get("append_only") is not True:
        raise ValueError("forward journal must be append-only")


def _validate_champion_manifest(
    manifest: Mapping[str, Any],
    *,
    model_path: Path,
) -> str:
    if manifest.get("status") != "retrospective_champion":
        raise ValueError("champion manifest has unexpected status")
    if manifest.get("production_ready") is not False:
        raise ValueError("forward preparation requires a non-production retrospective champion")
    if manifest.get("future_forward_required") is not True:
        raise ValueError("champion manifest must require future-forward evidence")
    if manifest.get("post_test_tuning_allowed") is not False:
        raise ValueError("post-test tuning must remain prohibited")
    if manifest.get("historical_test_in_training") is not False:
        raise ValueError("historical test must not be in champion training")
    if int(manifest.get("horizon_minutes", -1)) != NONLINEAR_CONFIG.horizon_minutes:
        raise ValueError("champion manifest horizon mismatch")
    if manifest.get("model_family") != "HistGradientBoostingRegressor":
        raise ValueError("champion manifest model family mismatch")
    artifacts = _mapping(manifest.get("artifacts"), "champion.artifacts")
    expected_sha = str(artifacts.get("model_sha256") or "")
    if not expected_sha:
        raise ValueError("champion manifest is missing model SHA256")
    actual_sha = sha256_file(model_path)
    if actual_sha != expected_sha:
        raise ValueError("champion model SHA256 does not match manifest")
    return actual_sha


def prepare_forward_shadow(
    *,
    dataset_dir: str | Path,
    champion_manifest_path: str | Path,
    champion_model_path: str | Path,
    protocol_path: str | Path,
    linear_model_output: str | Path,
    bundle_output: str | Path,
) -> dict[str, Any]:
    root = Path(dataset_dir)
    champion_manifest_file = Path(champion_manifest_path)
    champion_model_file = Path(champion_model_path)
    protocol_file = Path(protocol_path)

    champion_manifest = json.loads(champion_manifest_file.read_text(encoding="utf-8"))
    protocol = json.loads(protocol_file.read_text(encoding="utf-8"))
    if not isinstance(champion_manifest, Mapping) or not isinstance(protocol, Mapping):
        raise ValueError("champion manifest and forward protocol must contain JSON objects")
    _validate_protocol(protocol)
    champion_sha = _validate_champion_manifest(
        champion_manifest,
        model_path=champion_model_file,
    )

    # Loading confirms the serialized artifact is the expected frozen model type/config.
    champion = pickle.loads(champion_model_file.read_bytes())
    if not isinstance(champion, UnifiedHistGradientMagnitude):
        raise ValueError("champion pickle is not a UnifiedHistGradientMagnitude")
    if champion.config != NONLINEAR_CONFIG:
        raise ValueError("champion pickle configuration changed after lockbox")

    train_path = root / "train.jsonl"
    validation_path = root / "validation.jsonl"
    if not train_path.is_file() or not validation_path.is_file():
        raise FileNotFoundError("v3 train.jsonl and validation.jsonl are required")

    # Historical test.jsonl is intentionally never referenced or opened here.
    training_samples = [*_load_train(train_path), *_load_validation(validation_path)]
    linear_model, linear_fit = fit_unified_linear_magnitude_baseline(
        training_samples,
        config=LINEAR_CONFIG,
    )
    linear_path = Path(linear_model_output)
    linear_path.parent.mkdir(parents=True, exist_ok=True)
    linear_path.write_bytes(pickle.dumps(linear_model, protocol=pickle.HIGHEST_PROTOCOL))
    linear_sha = sha256_file(linear_path)

    universe = [str(value).upper() for value in protocol.get("universe", [])]
    if not universe:
        raise ValueError("forward protocol universe is empty")

    payload = {
        "schema_version": 1,
        "status": "forward_shadow_ready",
        "production_ready": False,
        "protocol_name": protocol["protocol_name"],
        "protocol_sha256": canonical_json_sha256(protocol),
        "protocol_path": str(protocol_file),
        "forward_start_utc": protocol["forward_start_utc"],
        "universe": universe,
        "session": protocol["session"],
        "bar_timeframe": protocol["bar_timeframe"],
        "bar_availability_delay_minutes": protocol["bar_availability_delay_minutes"],
        "horizon_minutes": protocol["horizon_minutes"],
        "feed": "iex",
        "champion": {
            "model_path": str(champion_model_file),
            "model_sha256": champion_sha,
            "manifest_path": str(champion_manifest_file),
            "manifest_sha256": sha256_file(champion_manifest_file),
        },
        "linear_incumbent": {
            "model_path": str(linear_path),
            "model_sha256": linear_sha,
            "fit_summary": linear_fit.to_dict(),
        },
        "null": {
            "family": "frozen_train_plus_validation_median_absolute_return",
            "magnitude_bps": linear_fit.training_median_magnitude_bps,
        },
        "historical_test_accessed_during_prepare": False,
        "model_updates_during_forward_period": False,
    }
    destination = Path(bundle_output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Prepare frozen v3 15m magnitude models for future-forward shadow evaluation.")
    result.add_argument("--dataset-dir", required=True)
    result.add_argument("--champion-manifest", default="runs/first-real-15m-v3/v3_15m_magnitude_champion.json")
    result.add_argument("--champion-model", default="runs/first-real-15m-v3/v3_15m_magnitude_champion.pkl")
    result.add_argument("--protocol", default="research/v3_15m_magnitude_forward_protocol_v2.json")
    result.add_argument("--linear-model-output", default="runs/first-real-15m-v3/v3_15m_forward_linear_incumbent.pkl")
    result.add_argument("--bundle-output", default="runs/first-real-15m-v3/v3_15m_forward_bundle.json")
    return result


def main() -> int:
    args = parser().parse_args()
    payload = prepare_forward_shadow(
        dataset_dir=args.dataset_dir,
        champion_manifest_path=args.champion_manifest,
        champion_model_path=args.champion_model,
        protocol_path=args.protocol,
        linear_model_output=args.linear_model_output,
        bundle_output=args.bundle_output,
    )
    print("=== V3 15M FORWARD SHADOW PREPARED ===")
    print("Protocol:", payload["protocol_name"])
    print("Forward start UTC:", payload["forward_start_utc"])
    print("Champion SHA256:", payload["champion"]["model_sha256"])
    print("Linear SHA256:", payload["linear_incumbent"]["model_sha256"])
    print("Frozen null magnitude bps:", payload["null"]["magnitude_bps"])
    print("Historical test accessed during prepare: NO")
    print("Production ready: NO - shadow evaluation only")
    print("Bundle:", args.bundle_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
