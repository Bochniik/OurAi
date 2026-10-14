from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from trading.dataset import MarketBar, build_causal_samples
from trading.forward_shadow import (
    append_journal_event,
    build_feature_only_samples,
    journal_state,
    prediction_id,
    read_journal,
    verify_journal,
)


class ForwardShadowTests(unittest.TestCase):
    def _bars(self, count: int = 45) -> list[MarketBar]:
        start = datetime(2026, 10, 15, 13, 30, tzinfo=timezone.utc)
        bars: list[MarketBar] = []
        price = 100.0
        for index in range(count):
            close = price * (1.0 + (0.0002 if index % 2 == 0 else -0.0001))
            bars.append(
                MarketBar(
                    symbol="AAPL",
                    timestamp=start + timedelta(minutes=index),
                    open=price,
                    high=max(price, close) + 0.02,
                    low=min(price, close) - 0.02,
                    close=close,
                    volume=1000 + index * 10,
                    session="regular",
                )
            )
            price = close
        return bars

    def test_feature_only_builder_matches_causal_feature_contract(self) -> None:
        bars = self._bars()
        feature_only = build_feature_only_samples(
            bars,
            bar_availability_delay_minutes=1.0,
        )
        resolved = build_causal_samples(
            bars,
            horizons_minutes=(5,),
            bar_availability_delay_minutes=1.0,
        )
        feature_by_timestamp = {row.timestamp: row for row in feature_only}
        self.assertGreater(len(resolved), 10)
        for row in resolved[:20]:
            live = feature_by_timestamp[row.timestamp]
            self.assertEqual(live.symbol, row.symbol)
            self.assertEqual(live.features, row.features)
            self.assertEqual(live.targets, {})

    def test_hash_chained_journal_rejects_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "events.jsonl"
            append_journal_event(
                path,
                {
                    "event_id": "prediction:abc",
                    "event_type": "prediction",
                    "prediction_id": "abc",
                },
                recorded_at=datetime(2026, 10, 15, 13, 31, tzinfo=timezone.utc),
            )
            append_journal_event(
                path,
                {
                    "event_id": "resolution:abc",
                    "event_type": "resolution",
                    "prediction_id": "abc",
                },
                recorded_at=datetime(2026, 10, 15, 13, 46, tzinfo=timezone.utc),
            )
            events = read_journal(path)
            self.assertIsNotNone(verify_journal(events))
            state = journal_state(events)
            self.assertIn("abc", state["predictions"])
            self.assertIn("abc", state["resolutions"])

            tampered = [dict(row) for row in events]
            tampered[0]["prediction_id"] = "changed"
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                verify_journal(tampered)

    def test_prediction_id_is_deterministic(self) -> None:
        timestamp = datetime(2026, 10, 15, 13, 31, tzinfo=timezone.utc)
        first = prediction_id(
            protocol_name="protocol",
            model_sha256="abc123",
            symbol="aapl",
            sample_timestamp=timestamp,
        )
        second = prediction_id(
            protocol_name="protocol",
            model_sha256="abc123",
            symbol="AAPL",
            sample_timestamp=timestamp,
        )
        self.assertEqual(first, second)
        self.assertEqual(len(first), 32)


if __name__ == "__main__":
    unittest.main()
