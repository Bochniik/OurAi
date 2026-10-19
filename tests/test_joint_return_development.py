"""Synthetic contracts for chronological TRAIN-only joint-return comparison."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from datetime import datetime, timedelta, timezone

from trading.baseline import BaselineConfig
from trading.dataset import TradingSample
from trading.joint_oof import JointOOFConfig
from trading.joint_return_calibration import JointCalibrationConfig
from scripts.evaluate_joint_return_development import (
    JointDevelopmentConfig, evaluate_train_only_joint, run_experiment,
)


class JointReturnDevelopmentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = JointDevelopmentConfig(
            horizon_minutes=15, evaluation_market_dates=2, embargo_minutes=60,
        )
        self.oof = JointOOFConfig(
            horizon_minutes=15, min_train_market_dates=2,
            validation_market_dates=1, step_market_dates=1,
            embargo_minutes=60,
        )
        self.cal = JointCalibrationConfig(
            horizon_minutes=15, minimum_cell_support=2,
        )
        self.baseline = BaselineConfig(
            horizon_minutes=15, probability_epochs=5,
            probability_batch_size=128, minimum_training_samples=40,
        )

    def rows(self):
        first = datetime(2026, 1, 5, 14, 30, tzinfo=timezone.utc)
        rows = []
        for day in range(9):
            first_of_day = first + timedelta(days=day)
            for minute in range(45):
                stamp = first_of_day + timedelta(minutes=minute)
                for symbol_index, symbol in enumerate(("NVDA", "AMD")):
                    upward = (day + minute + symbol_index) % 3 != 0
                    actual = 0.001 if upward else -0.002
                    rows.append(TradingSample(
                        symbol=symbol,
                        timestamp=stamp,
                        session_date=first_of_day.date().isoformat(),
                        session="regular",
                        features={
                            "price": 100 + day + minute / 100,
                            "return_1m": 0.0005 if upward else -0.0005,
                            "relative_volume_20": 1 + (minute % 5) / 10,
                        },
                        targets={"15m": {
                            "future_timestamp": (
                                stamp + timedelta(minutes=15)
                            ).isoformat(),
                            "future_return": actual,
                            "direction": "UP" if upward else "DOWN",
                        }},
                    ))
        return rows

    def run_small(self, samples=None):
        return evaluate_train_only_joint(
            self.rows() if samples is None else samples,
            source_split="train", config=self.config,
            oof_config=self.oof, calibration_config=self.cal,
            direction_config=self.baseline, magnitude_config=self.baseline,
        )

    def test_end_to_end_reports_coverage_and_fair_paired_scores(self):
        result = self.run_small()
        self.assertEqual(result["source"], "v4_train_split_only")
        self.assertEqual(len(result["development_market_dates"]), 2)
        self.assertTrue(set(result["development_market_dates"]).isdisjoint(
            result["train_market_dates"]
        ))
        self.assertEqual(result["eligible_nonoverlap_development_count"], 12)
        self.assertEqual(
            result["joint_forecast_count"] + result["abstained_count"],
            result["eligible_nonoverlap_development_count"],
        )
        self.assertGreater(result["oof_count"], 0)
        self.assertGreater(result["joint_forecast_count"], 0)
        self.assertIsNotNone(result["covered_only_metrics"])
        self.assertIn("signed_linear_mae_bps", result["covered_only_metrics"])
        self.assertIn("train_prior_three_class_brier", result["covered_only_metrics"])
        self.assertIn("joint_mae_optimal_magnitude_mae_bps", result["covered_only_metrics"])
        self.assertGreaterEqual(
            result["covered_only_metrics"]["joint_mae_optimal_magnitude_mae_bps"], 0,
        )
        self.assertGreaterEqual(
            result["covered_only_metrics"]["q10_q90_empirical_coverage"], 0
        )
        self.assertLessEqual(
            result["covered_only_metrics"]["q10_q90_empirical_coverage"], 1
        )
        self.assertFalse(result["test_lockbox_accessed"])
        self.assertFalse(result["original_validation_used"])
        self.assertFalse(result["v3_forward_used"])
        self.assertFalse(result["production_promotion_allowed"])

    def test_eval_rows_never_enter_oof_fit_or_incumbent_train(self):
        rows = self.rows()
        last_dates = set(sorted({r.session_date for r in rows})[-2:])
        import scripts.evaluate_joint_return_development as module
        original_oof = module.generate_joint_train_oof
        original_fit = module.fit_unified_linear_baseline
        seen = []

        def checked_oof(samples, **kwargs):
            self.assertFalse(any(r.session_date in last_dates for r in samples))
            seen.append("oof")
            return original_oof(samples, **kwargs)

        def checked_fit(samples, **kwargs):
            self.assertFalse(any(r.session_date in last_dates for r in samples))
            seen.append("fit")
            return original_fit(samples, **kwargs)

        with patch.object(module, "generate_joint_train_oof", checked_oof), patch.object(
            module, "fit_unified_linear_baseline", checked_fit
        ):
            self.run_small(rows)
        self.assertEqual(seen, ["oof", "fit"])

    def test_fails_closed_for_wrong_split_or_insufficient_days(self):
        with self.assertRaisesRegex(ValueError, "source_split"):
            evaluate_train_only_joint(self.rows(), source_split="validation")
        with self.assertRaisesRegex(ValueError, "not enough TRAIN"):
            self.run_small(self.rows()[:90])
        with self.assertRaisesRegex(ValueError, "horizons must match"):
            evaluate_train_only_joint(
                self.rows(), source_split="train", config=self.config,
                oof_config=self.oof,
                calibration_config=JointCalibrationConfig(horizon_minutes=5),
            )

    def test_runner_never_opens_original_validation_or_test_splits(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            rows = self.rows()
            (root / "train.jsonl").write_text(
                "\n".join(json.dumps(r.to_dict()) for r in rows) + "\n",
                encoding="utf-8",
            )
            # Deliberately malformed reserved splits: opening/reading would fail.
            (root / "validation.jsonl").write_text("{INVALID", encoding="utf-8")
            (root / "test.jsonl").write_text("{INVALID", encoding="utf-8")
            output = root / "outputs" / "report.json"
            with patch(
                "scripts.evaluate_joint_return_development._validate_v4_contract",
                return_value=(
                    {"dataset_id": "synthetic"},
                    {"protocol_name": "synthetic-train-only"},
                    {},
                ),
            ), patch(
                "scripts.evaluate_joint_return_development.evaluate_train_only_joint",
                return_value={"source": "v4_train_split_only"},
            ) as evaluator:
                report = run_experiment(root, output)
            self.assertTrue(output.is_file())
            self.assertEqual(report["dataset_id"], "synthetic")
            self.assertEqual(report["protocol_name"], "synthetic-train-only")
            passed_samples = evaluator.call_args.args[0]
            self.assertEqual(len(passed_samples), len(rows))
            self.assertEqual(evaluator.call_args.kwargs["source_split"], "train")


if __name__ == "__main__":
    unittest.main()
