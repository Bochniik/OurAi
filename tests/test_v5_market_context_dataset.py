from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.build_v5_market_context_dataset import (
    CONTEXT_SYMBOLS,
    EXPECTED_SPLIT_DATE_COUNTS,
    build_v5_market_context_dataset,
)


class V5MarketContextDatasetTests(unittest.TestCase):
    def test_builder_freezes_context_only_contract(self) -> None:
        captured = {}

        def fake_fetcher(destination, *, market_config, horizons_minutes, train_fraction, validation_fraction):
            captured["destination"] = Path(destination)
            captured["market_config"] = market_config
            captured["horizons_minutes"] = tuple(horizons_minutes)
            captured["train_fraction"] = train_fraction
            captured["validation_fraction"] = validation_fraction
            splits = {}
            cursor = 0
            for name, count in EXPECTED_SPLIT_DATE_COUNTS.items():
                splits[name] = {
                    "market_dates": [f"d{index:03d}" for index in range(cursor, cursor + count)],
                    "sample_count": count * 100,
                }
                cursor += count
            return {
                "builder": "ourai-historical-market-v2",
                "dataset_id": "context-dataset-id",
                "manifest_sha256": "manifest-sha",
                "source": {"partial": False},
                "source_cache": {"row_count": 12345},
                "dataset": {
                    "sample_count": 12000,
                    "market_date_count": 249,
                    "splits": splits,
                },
            }

        with tempfile.TemporaryDirectory() as temporary:
            result = build_v5_market_context_dataset(temporary, dataset_fetcher=fake_fetcher)
            root = Path(temporary)
            protocol = json.loads(
                (root / "v5_temporal_market_context_protocol.json").read_text(encoding="utf-8")
            )
            persisted = json.loads(
                (root / "v5_market_context_result.json").read_text(encoding="utf-8")
            )

        self.assertEqual(tuple(captured["market_config"].symbols), CONTEXT_SYMBOLS)
        self.assertEqual(captured["horizons_minutes"], (15,))
        self.assertEqual(protocol["market_context"]["targets_from_context_symbols_used"], False)
        self.assertEqual(result["context_targets_used_by_direction_research"], False)
        self.assertEqual(result["v4_validation_accessed"], False)
        self.assertEqual(result["v4_test_accessed"], False)
        self.assertEqual(result["split_market_date_counts"], EXPECTED_SPLIT_DATE_COUNTS)
        self.assertEqual(persisted, result)


if __name__ == "__main__":
    unittest.main()
