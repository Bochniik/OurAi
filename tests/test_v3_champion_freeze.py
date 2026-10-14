from __future__ import annotations

import json
import unittest
from pathlib import Path

from scripts.freeze_v3_nonlinear_champion import _validate_test_report
from scripts.validate_v3_nonlinear_magnitude import _expected_nonlinear_config


class V3ChampionFreezeTests(unittest.TestCase):
    def _report(self) -> dict:
        return {
            "experiment": "v3_hist_gradient_absolute_return_magnitude_test_lockbox_v1",
            "dataset_id": "dataset-v3",
            "protocol_name": "protocol-v3",
            "protocol_sha256": "abc",
            "development_report": "development.json",
            "validation_report": "validation.json",
            "frozen_nonlinear_model": {
                "family": "HistGradientBoostingRegressor",
                "loss": "absolute_error",
                "config": _expected_nonlinear_config(),
            },
            "test_rule": {
                "overall_nonlinear_mae_must_beat_linear": True,
                "overall_nonlinear_mae_must_beat_final_training_median_null": True,
                "require_strict_majority_market_date_wins_vs_linear": True,
                "require_strict_majority_market_date_wins_vs_null": True,
                "tuning_after_test": False,
                "pnl_used": False,
            },
            "test_supported": True,
            "test_lockbox_accessed": True,
            "post_test_tuning_allowed": False,
        }

    def test_successful_test_report_is_accepted_for_freeze(self) -> None:
        _validate_test_report(
            self._report(),
            dataset_id="dataset-v3",
            protocol_name="protocol-v3",
            protocol_sha256="abc",
            development_report_path="development.json",
            validation_report_path="validation.json",
        )

    def test_failed_lockbox_cannot_be_frozen_as_champion(self) -> None:
        report = self._report()
        report["test_supported"] = False
        with self.assertRaisesRegex(ValueError, "did not pass"):
            _validate_test_report(
                report,
                dataset_id="dataset-v3",
                protocol_name="protocol-v3",
                protocol_sha256="abc",
                development_report_path="development.json",
                validation_report_path="validation.json",
            )

    def test_future_forward_protocol_starts_after_historical_research(self) -> None:
        protocol_path = Path(__file__).parents[1] / "research" / "v3_15m_magnitude_forward_protocol.json"
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        self.assertEqual(protocol["forward_start_utc"], "2026-10-15T13:30:00Z")
        self.assertTrue(protocol["champion"]["updates_during_forward_period"] is False)
        self.assertEqual(protocol["evaluation"]["minimum_complete_market_dates_before_decision"], 20)
        self.assertTrue(protocol["prediction_logging"]["prediction_must_be_written_before_target_resolution"])
        self.assertFalse(protocol["research_safety"]["production_promotion_before_forward_gate"])


if __name__ == "__main__":
    unittest.main()
