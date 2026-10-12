"""Reproducible experiment lineage for trading-model development.

An experiment must be identifiable from the data provenance, feature schema,
validation design, model/decision configuration and exact code revision. This module
produces deterministic SHA-256 fingerprints for those components and one compact
experiment ID that changes whenever any scientifically relevant input changes.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
import hashlib
import json
from math import isfinite
from typing import Any, Mapping, Sequence


def _normalize(value: Any) -> Any:
    """Convert supported values into deterministic JSON-compatible structures."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not isfinite(value):
            raise ValueError("experiment metadata cannot contain NaN/Infinity")
        return value
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {
            str(key): _normalize(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_normalize(item) for item in value]
    if isinstance(value, (set, frozenset)):
        normalized = [_normalize(item) for item in value]
        return sorted(normalized, key=lambda item: json.dumps(item, sort_keys=True))
    raise TypeError(f"unsupported experiment metadata type: {type(value).__name__}")


def canonical_json(value: Any) -> str:
    """Serialize metadata in a stable representation used for hashing."""

    return json.dumps(
        _normalize(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def sha256_metadata(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ExperimentIdentity:
    experiment_id: str
    dataset_hash: str
    feature_schema_hash: str
    validation_hash: str
    config_hash: str
    code_revision: str
    lineage_hash: str

    def to_dict(self) -> dict[str, str]:
        return {
            "experiment_id": self.experiment_id,
            "dataset_hash": self.dataset_hash,
            "feature_schema_hash": self.feature_schema_hash,
            "validation_hash": self.validation_hash,
            "config_hash": self.config_hash,
            "code_revision": self.code_revision,
            "lineage_hash": self.lineage_hash,
        }


def build_experiment_identity(
    *,
    dataset_provenance: Mapping[str, Any],
    feature_names: Sequence[str],
    validation_design: Mapping[str, Any],
    experiment_config: Mapping[str, Any],
    code_revision: str,
) -> ExperimentIdentity:
    """Build a deterministic identity for one preregistered experiment.

    ``dataset_provenance`` should normally be the ingestion/split manifest containing
    raw-source and generated-split hashes. ``validation_design`` should contain the
    walk-forward manifest/config. ``experiment_config`` should include model family,
    hyperparameters, horizons, calibration choices, costs and decision thresholds
    that could affect reported results.
    """

    revision = str(code_revision or "").strip()
    if not revision:
        raise ValueError("code_revision is required")

    normalized_features = sorted({str(name).strip() for name in feature_names if str(name).strip()})
    if not normalized_features:
        raise ValueError("feature_names must not be empty")

    dataset_hash = sha256_metadata(dataset_provenance)
    feature_schema_hash = sha256_metadata(normalized_features)
    validation_hash = sha256_metadata(validation_design)
    config_hash = sha256_metadata(experiment_config)

    lineage_payload = {
        "dataset_hash": dataset_hash,
        "feature_schema_hash": feature_schema_hash,
        "validation_hash": validation_hash,
        "config_hash": config_hash,
        "code_revision": revision,
    }
    lineage_hash = sha256_metadata(lineage_payload)

    return ExperimentIdentity(
        experiment_id=f"exp_{lineage_hash[:16]}",
        dataset_hash=dataset_hash,
        feature_schema_hash=feature_schema_hash,
        validation_hash=validation_hash,
        config_hash=config_hash,
        code_revision=revision,
        lineage_hash=lineage_hash,
    )
