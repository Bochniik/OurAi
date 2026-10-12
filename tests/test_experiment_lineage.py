from __future__ import annotations

import unittest

from trading.experiment import (
    build_experiment_identity,
    canonical_json,
    sha256_metadata,
)


class ExperimentLineageTests(unittest.TestCase):
    def _identity(self, **overrides):
        arguments = {
            "dataset_provenance": {
                "source_sha256": "abc",
                "splits": {
                    "train": {"sha256": "train-hash"},
                    "validation": {"sha256": "val-hash"},
                    "test": {"sha256": "test-hash"},
                },
            },
            "feature_names": ["price", "return_5m", "ctx__news__sentiment"],
            "validation_design": {
                "policy": "strictly_chronological_purged_walk_forward",
                "embargo_minutes": 60,
                "fold_count": 4,
            },
            "experiment_config": {
                "model": "gradient_boosted_trees",
                "horizon": "15m",
                "neutral_threshold_bps": 3,
            },
            "code_revision": "0123456789abcdef",
        }
        arguments.update(overrides)
        return build_experiment_identity(**arguments)

    def test_mapping_key_order_does_not_change_hash(self) -> None:
        left = {"a": 1, "b": {"x": 2, "y": 3}}
        right = {"b": {"y": 3, "x": 2}, "a": 1}
        self.assertEqual(canonical_json(left), canonical_json(right))
        self.assertEqual(sha256_metadata(left), sha256_metadata(right))

    def test_feature_order_and_duplicates_do_not_change_identity(self) -> None:
        first = self._identity()
        second = self._identity(
            feature_names=[
                "ctx__news__sentiment",
                "price",
                "return_5m",
                "price",
            ]
        )
        self.assertEqual(first.feature_schema_hash, second.feature_schema_hash)
        self.assertEqual(first.experiment_id, second.experiment_id)

    def test_dataset_change_changes_experiment_identity(self) -> None:
        first = self._identity()
        changed = self._identity(
            dataset_provenance={
                "source_sha256": "different-source",
                "splits": {
                    "train": {"sha256": "train-hash"},
                    "validation": {"sha256": "val-hash"},
                    "test": {"sha256": "test-hash"},
                },
            }
        )
        self.assertNotEqual(first.dataset_hash, changed.dataset_hash)
        self.assertNotEqual(first.experiment_id, changed.experiment_id)

    def test_validation_change_changes_experiment_identity(self) -> None:
        first = self._identity()
        changed = self._identity(
            validation_design={
                "policy": "strictly_chronological_purged_walk_forward",
                "embargo_minutes": 120,
                "fold_count": 4,
            }
        )
        self.assertNotEqual(first.validation_hash, changed.validation_hash)
        self.assertNotEqual(first.experiment_id, changed.experiment_id)

    def test_model_or_decision_config_change_changes_identity(self) -> None:
        first = self._identity()
        changed = self._identity(
            experiment_config={
                "model": "gradient_boosted_trees",
                "horizon": "15m",
                "neutral_threshold_bps": 5,
            }
        )
        self.assertNotEqual(first.config_hash, changed.config_hash)
        self.assertNotEqual(first.experiment_id, changed.experiment_id)

    def test_code_revision_change_changes_identity(self) -> None:
        first = self._identity()
        changed = self._identity(code_revision="fedcba9876543210")
        self.assertNotEqual(first.experiment_id, changed.experiment_id)

    def test_identity_has_full_component_hashes_and_compact_id(self) -> None:
        identity = self._identity()
        payload = identity.to_dict()
        self.assertTrue(payload["experiment_id"].startswith("exp_"))
        self.assertEqual(len(payload["experiment_id"]), 20)
        for key in (
            "dataset_hash",
            "feature_schema_hash",
            "validation_hash",
            "config_hash",
            "lineage_hash",
        ):
            self.assertEqual(len(payload[key]), 64)

    def test_empty_features_and_revision_fail_closed(self) -> None:
        with self.assertRaises(ValueError):
            self._identity(feature_names=[])
        with self.assertRaises(ValueError):
            self._identity(code_revision="")

    def test_non_finite_metadata_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            sha256_metadata({"loss": float("nan")})


if __name__ == "__main__":
    unittest.main()
