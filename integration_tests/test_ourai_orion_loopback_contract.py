"""Cross-repository OurAI producer -> Orion consumer smoke tests.

Run only in the dedicated workflow after checking out Orion-Trader alongside
OurAI. No real journal, market data, model execution, credentials or trading.
The server and client communicate over an actual ephemeral 127.0.0.1 socket.
"""
from __future__ import annotations

from contextlib import contextmanager
from http.server import ThreadingHTTPServer
import importlib.util
from pathlib import Path
from threading import Thread
import unittest
from unittest.mock import patch

from scripts import serve_orion_forward_bridge as producer


ORION_BRIDGE_FILE = (
    Path(__file__).resolve().parents[1]
    / "orion-trader/backend/app/services/ourai_bridge.py"
)


def load_orion_consumer():
    if not ORION_BRIDGE_FILE.is_file():
        raise FileNotFoundError(
            "Orion-Trader checkout missing: cross-repository contract was not tested"
        )
    spec = importlib.util.spec_from_file_location(
        "orion_read_only_contract_consumer", ORION_BRIDGE_FILE
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot import the checked-out Orion consumer")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def synthetic_verified_report():
    """Structurally valid *synthetic* analytics, never real forward evidence."""
    return {
        "journal_exists": True,
        "journal_valid": True,
        "evaluated_at_utc": "2026-10-20T12:00:00Z",
        "prediction_count": 2,
        "resolution_count": 1,
        "pending_count": 1,
        "overdue_pending_count": 0,
        "champion_mae_bps": 8.7,
        "linear_mae_bps": 9.8,
        "null_mae_bps": 10.7,
        "symbols": {
            "NVDA": {
                "predictions": 2,
                "resolved": 1,
                "pending": 1,
                "overdue_pending": 0,
                "champion_mae_bps": 8.7,
                "linear_mae_bps": 9.8,
                "null_mae_bps": 10.7,
            }
        },
        "latest_by_symbol": {
            "NVDA": {
                "symbol": "NVDA",
                "predicted_at": "2026-10-20T11:59:00Z",
                "sample_timestamp": "2026-10-20T11:58:00Z",
                "target_due_at": "2026-10-20T12:13:00Z",
                "expected_magnitude_bps": 12.0,
                "resolved": False,
            }
        },
    }


@contextmanager
def real_producer():
    server = ThreadingHTTPServer(
        (producer.HOST, 0), producer.make_handler("synthetic-not-opened.jsonl")
    )
    worker = Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        yield server.server_port
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=5)


class CrossRepoLoopbackContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.consumer = load_orion_consumer()

    def test_real_producer_response_is_sanitized_by_real_consumer(self):
        report = synthetic_verified_report()
        report["secret"] = "synthetic-never-publish"
        report["p_up"] = 0.99
        report["symbols"]["NVDA"]["p_down"] = 0.01
        report["latest_by_symbol"]["NVDA"]["trade_signal"] = "BUY"
        with patch.object(producer, "read_forward_report", return_value=report) as read:
            with real_producer() as port:
                result = self.consumer.fetch_ourai_forward_evidence(
                    enabled=True, port=port
                )
        read.assert_called_once_with("synthetic-not-opened.jsonl")
        self.assertEqual(result["status"], "available")
        self.assertEqual(result["source"], "ourai_verified_forward_journal")
        self.assertEqual(result["resolution_count"], 1)
        self.assertEqual(result["pending_count"], 1)
        self.assertEqual(result["symbols"]["NVDA"]["predictions"], 2)
        self.assertFalse(result["direction_available"])
        self.assertFalse(result["trading_enabled"])
        self.assertNotIn("p_up", str(result))
        self.assertNotIn("p_down", str(result))
        self.assertNotIn("trade_signal", str(result))
        self.assertNotIn("synthetic-never-publish", str(result))

    def test_invalid_journal_fails_closed_then_fresh_request_recovers(self):
        with patch.object(
            producer,
            "read_forward_report",
            side_effect=[ValueError("synthetic hash-chain error"), synthetic_verified_report()],
        ) as read:
            with real_producer() as port:
                unavailable = self.consumer.fetch_ourai_forward_evidence(
                    enabled=True, port=port
                )
                recovered = self.consumer.fetch_ourai_forward_evidence(
                    enabled=True, port=port
                )
        self.assertEqual(read.call_count, 2)
        self.assertEqual(unavailable["status"], "unavailable")
        self.assertEqual(unavailable["unavailable_reason"], "upstream_http_failure")
        self.assertFalse(unavailable["direction_available"])
        self.assertFalse(unavailable["trading_enabled"])
        self.assertNotIn("symbols", unavailable)
        self.assertNotIn("champion_mae_bps", unavailable)
        self.assertNotIn("hash-chain", str(unavailable))
        self.assertEqual(recovered["status"], "available")
        self.assertEqual(recovered["prediction_count"], 2)

    def test_inconsistent_analytics_is_not_forward_evidence(self):
        report = synthetic_verified_report()
        report["prediction_count"] = 3  # Conflicts with NVDA's two predictions.
        with patch.object(producer, "read_forward_report", return_value=report):
            with real_producer() as port:
                result = self.consumer.fetch_ourai_forward_evidence(
                    enabled=True, port=port
                )
        self.assertEqual(result["status"], "unavailable")
        self.assertEqual(result["unavailable_reason"], "invalid_upstream_evidence")
        self.assertFalse(result["direction_available"])
        self.assertFalse(result["trading_enabled"])
        self.assertNotIn("symbols", result)

    def test_disabled_consumer_never_contacts_producer(self):
        with patch.object(
            producer, "read_forward_report", side_effect=AssertionError("unexpected read")
        ) as read:
            with real_producer() as port:
                result = self.consumer.fetch_ourai_forward_evidence(
                    enabled=False, port=port
                )
        read.assert_not_called()
        self.assertEqual(result["status"], "disabled")
        self.assertFalse(result["trading_enabled"])


if __name__ == "__main__":
    unittest.main()
