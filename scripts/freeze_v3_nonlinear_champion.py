"""Freeze the accepted OurAI v3 nonlinear 15m magnitude model for forward shadow use.

This script does not reopen the historical test split. It validates the already-written
successful development, validation, and test reports, refits the exact frozen nonlinear
model on train + validation only, serializes it locally, and writes a tamper-evident
champion manifest. The resulting model is a retrospective champion, not production-ready;
future-forward evidence is still required.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path
from typing import Any, Mapping

from scripts.evaluate_magnitude_head import _load_train
from scripts.evaluate_v3_nonlinear_magnitude import NONLINEAR_CONFIG, _validate_v3_contract
from scripts.validate_magnitude_head import _load_validation
from scripts.validate_v3_nonlinear_magnitude import (
    _expected_nonlinear_config,
    _validate_development_report,
)
from scripts.test_v3_nonlinear_lockbox import _validate_validation_report
from trading.nonlinear import fit_hist_gradient_magnitude

EXPECTED_TEST_EXPERIMENT = "v3_hist_gradient_absolute_return_magnitude_test_lockbox_v1"


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_test_report(
    report: Mapping[str, Any],
    *,
    dataset_id: object,
    protocol_name: object,
    protocol_sha256: object,
    development_report_path: str | Path,
    validation_report_path: str | Path,
) -> None:
    if report.get("experiment") != EXPECTED_TEST_EXPERIMENT:
        raise ValueError("unexpected v3 nonlinear test experiment identity")
    if report.get("dataset_id") != dataset_id:
        raise ValueError("test report dataset_id does not match v3 dataset")
    if report.get("protocol_name") != protocol_name:
        raise ValueError("test report protocol_name does not match v3 protocol")
    if report.get("protocol_sha256") != protocol_sha256:
        raise ValueError("test report protocol hash does not match v3 protocol")
    if report.get("development_report") != str(development_report_path):
        raise ValueError("test report references a different development report")
    if report.get("validation_report") != str(validation_report_path):
        raise ValueError("test report references a different validation report")
    if report.get("test_supported") is not True:
        raise ValueError("v3 nonlinear challenger did not pass the test lockbox")
    if report.get("test_lockbox_accessed") is not True:
        raise ValueError("test report must record that the lockbox was accessed")
    if report.get("post_test_tuning_allowed") is not False:
        raise ValueError("post-test tuning must remain prohibited")

    frozen = _mapping(report.get("frozen_nonlinear_model"), "frozen_nonlinear_model")
    if frozen.get("family") != "HistGradientBoostingRegressor":
        raise ValueError("unexpected frozen nonlinear model family")
    if frozen.get("loss") != "absolute_error":
        raise ValueError("unexpected frozen nonlinear loss")
    if dict(_mapping(frozen.get("config"), "frozen_nonlinear_model.config")) != _expected_nonlinear_config():
        raise ValueError("test report changed the frozen nonlinear configuration")

    rule = _mapping(report.get("test_rule"), "test_rule")
    for key in (
        "overall_nonlinear_mae_must_beat_linear",
        "overall_nonlinear_mae_must_beat_final_training_median_null",
        "require_strict_majority_market_date_wins_vs_linear",
        "require_strict_majority_market_date_wins_vs_null",
    ):
        if rule.get(key) is not True:
            raise ValueError(f"test report used an unexpected rule for {key}")
    if rule.get("tuning_after_test") is not False:
        raise ValueError("test report must prohibit post-test tuning")
    if rule.get("pnl_used") is not False:
        raise ValueError("test acceptance must not use PnL")


def freeze_champion(
    dataset_dir: str | Path,
    development_report_path: str | Path,
    validation_report_path: str | Path,
    test_report_path: str | Path,
    model_output_path: str | Path,
    manifest_output_path: str | Path,
) -> dict[str, Any]:
    root = Path(dataset_dir)
    manifest, protocol_result = _validate_v3_contract(root)

    development_path = Path(development_report_path)
    validation_path_report = Path(validation_report_path)
    test_path_report = Path(test_report_path)
    development = json.loads(development_path.read_text(encoding="utf-8"))
    validation = json.loads(validation_path_report.read_text(encoding="utf-8"))
    test = json.loads(test_path_report.read_text(encoding="utf-8"))
    if not isinstance(development, Mapping) or not isinstance(validation, Mapping) or not isinstance(test, Mapping):
        raise ValueError("development, validation, and test reports must contain JSON objects")

    _validate_development_report(
        development,
        dataset_id=manifest.get("dataset_id"),
        protocol_name=protocol_result.get("protocol_name"),
        protocol_sha256=protocol_result.get("protocol_sha256"),
    )
    _validate_validation_report(
        validation,
        dataset_id=manifest.get("dataset_id"),
        protocol_name=protocol_result.get("protocol_name"),
        protocol_sha256=protocol_result.get("protocol_sha256"),
        development_report_path=development_report_path,
    )
    _validate_test_report(
        test,
        dataset_id=manifest.get("dataset_id"),
        protocol_name=protocol_result.get("protocol_name"),
        protocol_sha256=protocol_result.get("protocol_sha256"),
        development_report_path=development_report_path,
        validation_report_path=validation_report_path,
    )

    train_path = root / "train.jsonl"
    validation_split_path = root / "validation.jsonl"
    if not train_path.is_file() or not validation_split_path.is_file():
        raise FileNotFoundError("train.jsonl and validation.jsonl are required to freeze the champion")

    # The historical test split is intentionally not referenced or opened here.
    final_training_samples = [
        *_load_train(train_path),
        *_load_validation(validation_split_path),
    ]
    model, fit_summary = fit_hist_gradient_magnitude(
        final_training_samples,
        config=NONLINEAR_CONFIG,
    )

    model_path = Path(model_output_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_bytes(pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL))

    payload = {
        "schema_version": 1,
        "status": "retrospective_champion",
        "production_ready": False,
        "future_forward_required": True,
        "post_test_tuning_allowed": False,
        "dataset_id": manifest.get("dataset_id"),
        "protocol_name": protocol_result.get("protocol_name"),
        "protocol_sha256": protocol_result.get("protocol_sha256"),
        "horizon_minutes": NONLINEAR_CONFIG.horizon_minutes,
        "model_family": "HistGradientBoostingRegressor",
        "model_config": _expected_nonlinear_config(),
        "final_training_source": "v3_train_plus_validation_only",
        "historical_test_in_training": False,
        "fit_summary": fit_summary.to_dict(),
        "artifacts": {
            "model_path": str(model_path),
            "model_sha256": _sha256_file(model_path),
            "development_report": str(development_path),
            "development_report_sha256": _sha256_file(development_path),
            "validation_report": str(validation_path_report),
            "validation_report_sha256": _sha256_file(validation_path_report),
            "test_report": str(test_path_report),
            "test_report_sha256": _sha256_file(test_path_report),
        },
        "historical_test_summary": test.get("test"),
    }

    manifest_path = Path(manifest_output_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Freeze the accepted v3 nonlinear 15m magnitude champion for forward shadow evaluation.")
    result.add_argument("--dataset-dir", required=True)
    result.add_argument("--development-report", required=True)
    result.add_argument("--validation-report", required=True)
    result.add_argument("--test-report", required=True)
    result.add_argument("--model-output", default="runs/first-real-15m-v3/v3_15m_magnitude_champion.pkl")
    result.add_argument("--manifest-output", default="runs/first-real-15m-v3/v3_15m_magnitude_champion.json")
    return result


def main() -> int:
    args = parser().parse_args()
    payload = freeze_champion(
        args.dataset_dir,
        args.development_report,
        args.validation_report,
        args.test_report,
        args.model_output,
        args.manifest_output,
    )
    print("=== V3 15M MAGNITUDE CHAMPION FROZEN ===")
    print("Dataset ID:", payload["dataset_id"])
    print("Status:", payload["status"])
    print("Model SHA256:", payload["artifacts"]["model_sha256"])
    print("Model:", payload["artifacts"]["model_path"])
    print("Manifest:", args.manifest_output)
    print("Historical test in training: NO")
    print("Post-test tuning allowed: NO")
    print("Production ready: NO - future-forward evidence required")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
