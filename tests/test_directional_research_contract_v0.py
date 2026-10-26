"""Synthetic-only tests for the isolated research directional shape contract."""
from __future__ import annotations

import copy
import unittest

from research.directional_research_contract_v0 import (
    REQUIRED_FIELDS, validate_directional_research_v0,
)


def synthetic_payload():
    # Deliberately invented numbers: never a benchmark, forecast, or journal.
    return {
        "contract": "ourai_directional_research_v0",
        "schema_version": 0,
        "mode": "RESEARCH_ONLY",
        "validation_status": "NOT_INDEPENDENTLY_VALIDATED",
        "evidence_status": "UNVERIFIED_MODEL_OUTPUT",
        "research_only": True,
        "trading_enabled": False,
        "experiment_id": "synthetic-shape-test",
        "symbol": "NVDA",
        "horizon_minutes": 15,
        "as_of_utc": "2026-10-20T14:00:00Z",
        "target_due_at_utc": "2026-10-20T14:15:00Z",
        "probability_conditioning": "up_vs_down_given_nonzero_15m_return",
        "p_up": 0.6,
        "p_down": 0.4,
        "expected_signed_change_pct": -0.12,
    }


class DirectionalResearchContractV0Tests(unittest.TestCase):
    def test_synthetic_research_shape_is_accepted_without_promotion(self):
        row = synthetic_payload()
        self.assertEqual(set(row), REQUIRED_FIELDS)
        checked = validate_directional_research_v0(row)
        self.assertEqual(checked["expected_signed_change_pct"], -0.12)
        self.assertEqual(checked["validation_status"], "NOT_INDEPENDENTLY_VALIDATED")
        self.assertFalse(checked["trading_enabled"])
        self.assertTrue(checked["research_only"])
        self.assertEqual(row, synthetic_payload())  # no input mutation

    def test_invalid_directional_probabilities_fail_closed(self):
        for up, down in ((0.8, 0.8), (-0.1, 1.1), (True, 0.0),
                         (float("nan"), 0.5), (float("inf"), 0.0),
                         ("0.6", 0.4), (10 ** 400, 0.0)):
            with self.subTest(up=up, down=down):
                row = synthetic_payload()
                row["p_up"], row["p_down"] = up, down
                with self.assertRaises(ValueError):
                    validate_directional_research_v0(row)

    def test_signed_return_must_be_finite_number_but_can_be_negative(self):
        for invalid in (True, "BUY", float("nan"), float("inf"), 10 ** 400):
            with self.subTest(value=invalid):
                row = synthetic_payload()
                row["expected_signed_change_pct"] = invalid
                with self.assertRaises(ValueError):
                    validate_directional_research_v0(row)

    def test_no_live_trading_or_unearned_validation_label(self):
        overrides = {
            "research_only": False,
            "trading_enabled": True,
            "mode": "PRODUCTION",
            "validation_status": "PASSED",
            "evidence_status": "VERIFIED_FORWARD",
            "schema_version": 1,
            "probability_conditioning": "unconditional",
            "horizon_minutes": 30,
        }
        for key, invalid in overrides.items():
            with self.subTest(field=key):
                row = synthetic_payload()
                row[key] = invalid
                with self.assertRaises(ValueError):
                    validate_directional_research_v0(row)

    def test_no_trading_or_journal_fields_can_be_smuggled(self):
        for field, value in (("trade_signal", "BUY"), ("order", {"side": "buy"}),
                             ("journal_valid", True), ("direction_available", True),
                             ("accuracy", 0.99), ("source", "ourai_verified_forward_journal")):
            with self.subTest(field=field):
                row = synthetic_payload()
                row[field] = value
                with self.assertRaises(ValueError):
                    validate_directional_research_v0(row)

    def test_missing_fields_and_bad_identity_fail_closed(self):
        row = synthetic_payload()
        del row["p_down"]
        with self.assertRaises(ValueError):
            validate_directional_research_v0(row)
        for field, value in (("symbol", "IBM"), ("symbol", ["NVDA"]),
                             ("experiment_id", "../model"), ("experiment_id", "")):
            with self.subTest(field=field):
                row = synthetic_payload()
                row[field] = value
                with self.assertRaises((ValueError, TypeError)):
                    validate_directional_research_v0(row)

    def test_horizon_and_utc_chronology_are_explicit(self):
        for field, value in (
            ("as_of_utc", "2026-10-20T14:00:00"),
            ("as_of_utc", "2026-10-20T16:00:00+02:00"),
            ("target_due_at_utc", "2026-10-20T14:14:59Z"),
            ("target_due_at_utc", "2026-10-20T14:30:00Z"),
            ("target_due_at_utc", "not-a-date"),
        ):
            with self.subTest(field=field, value=value):
                row = synthetic_payload()
                row[field] = value
                with self.assertRaises(ValueError):
                    validate_directional_research_v0(row)


if __name__ == "__main__":
    unittest.main()
