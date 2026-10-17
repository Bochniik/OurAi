from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "research" / "v5_temporal_market_context_protocol_v1.json"
AMENDMENT = ROOT / "research" / "v5_temporal_gru_implementation_amendment_v1.json"
SCRIPT = ROOT / "scripts" / "evaluate_v5_temporal_gru_direction.py"


class V5TemporalGRUProtocolTests(unittest.TestCase):
    def test_frozen_gru_contract(self):
        protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
        candidate = protocol["temporal_candidate"]
        self.assertEqual(candidate["family"], "GRU")
        self.assertEqual(candidate["sequence_length_observations"], 60)
        self.assertEqual(candidate["hidden_size"], 64)
        self.assertEqual(candidate["num_layers"], 2)
        self.assertEqual(candidate["epochs"], 5)
        self.assertEqual(candidate["batch_size"], 256)
        self.assertFalse(protocol["market_context"]["targets_from_context_symbols_used"])
        self.assertTrue(protocol["market_context"]["same_or_past_timestamp_only"])
        self.assertFalse(protocol["development"]["pnl_used"])

    def test_implementation_amendment_keeps_holdouts_closed(self):
        amendment = json.loads(AMENDMENT.read_text(encoding="utf-8"))
        access = amendment["data_access"]
        self.assertFalse(access["v4_validation_accessed"])
        self.assertFalse(access["v4_test_accessed"])
        self.assertFalse(access["context_validation_accessed"])
        self.assertFalse(access["context_test_accessed"])
        self.assertEqual(amendment["magnitude_gate"]["lineage"], "carry_forward_exact_v4_selective_magnitude_gate")
        self.assertFalse(amendment["development_gate"]["pnl_used"])
        self.assertTrue(amendment["promotion"]["separate_frozen_validation_gate_required"])

    def test_development_script_never_constructs_holdout_split_paths(self):
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertNotIn('/ "validation.jsonl"', source)
        self.assertNotIn('/ "test.jsonl"', source)
        self.assertIn('target_root / "train.jsonl"', source)
        self.assertIn('context_root / "train.jsonl"', source)
        self.assertIn('"context_targets_used": False', source)
        self.assertIn('"pnl_used": False', source)


if __name__ == "__main__":
    unittest.main()
