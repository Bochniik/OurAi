"""Loopback OurAI-to-Orion bridge: synthetic, read-only contract tests."""
from __future__ import annotations

from http.server import ThreadingHTTPServer
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import ProxyHandler, Request, build_opener

from scripts.serve_orion_forward_bridge import (
    HOST, ROUTE, forward_evidence_snapshot, make_handler,
)


class OrionForwardBridgeTests(unittest.TestCase):
    def fake_report(self):
        return {
            "journal_exists": True, "journal_valid": True,
            "evaluated_at_utc": "2026-10-20T12:00:00Z",
            "prediction_count": 2, "resolution_count": 1,
            "pending_count": 1, "overdue_pending_count": 0,
            "champion_mae_bps": 8.7, "linear_mae_bps": 9.8,
            "null_mae_bps": 10.7,
            "symbols": {"NVDA": {
                "predictions": 2, "resolved": 1,
                "pending": 1, "overdue_pending": 0,
                "champion_mae_bps": 8.7, "linear_mae_bps": 9.8,
                "null_mae_bps": 10.7,
            }},
            "latest_by_symbol": {"NVDA": {
                "symbol": "NVDA", "predicted_at": "2026-10-20T11:59:00Z",
                "sample_timestamp": "2026-10-20T11:58:00Z",
                "target_due_at": "2026-10-20T12:13:00Z",
                "expected_magnitude_bps": 12.0, "resolved": False,
            }},
        }

    def test_snapshot_whitelists_magnitude_evidence_only(self):
        report = self.fake_report()
        report["raw_credential"] = "must-never-cross-boundary"
        report["p_up"] = 0.9
        with patch("scripts.serve_orion_forward_bridge.read_forward_report", return_value=report):
            result = forward_evidence_snapshot("not-opened-in-mock.jsonl")
        self.assertEqual(result["schema_version"], 1)
        self.assertEqual(result["resolution_count"], 1)
        self.assertFalse(result["direction_available"])
        self.assertFalse(result["trading_enabled"])
        self.assertNotIn("p_up", result)
        self.assertNotIn("raw_credential", result)
        self.assertNotIn("signal", result)
        self.assertIsInstance(result["symbols"], dict)

    def test_missing_journal_fails_closed(self):
        report = self.fake_report()
        report["journal_exists"] = False
        with patch("scripts.serve_orion_forward_bridge.read_forward_report", return_value=report):
            with self.assertRaises(FileNotFoundError):
                forward_evidence_snapshot("missing.jsonl")

    def test_http_contract_and_invalid_host(self):
        with patch(
            "scripts.serve_orion_forward_bridge.read_forward_report",
            return_value=self.fake_report(),
        ):
            server = ThreadingHTTPServer((HOST, 0), make_handler("mock-journal.jsonl"))
            runner = threading.Thread(target=server.serve_forever, daemon=True)
            runner.start()
            try:
                base = f"http://127.0.0.1:{server.server_port}"
                opener = build_opener(ProxyHandler({}))
                with opener.open(base + ROUTE, timeout=5) as response:
                    self.assertEqual(response.status, 200)
                    self.assertEqual(response.headers["Cache-Control"], "no-store")
                    self.assertIsNone(response.headers.get("Access-Control-Allow-Origin"))
                    data = json.load(response)
                    self.assertEqual(data["experiment"], "frozen_v3_magnitude_15m")
                    self.assertEqual(data["champion_mae_bps"], 8.7)

                with self.assertRaises(HTTPError) as ctx:
                    opener.open(Request(base + ROUTE, headers={"Host": "malicious.invalid"}), timeout=5)
                self.assertEqual(ctx.exception.code, 403)
                with self.assertRaises(HTTPError) as missing:
                    opener.open(base + "/v1/predict/NVDA", timeout=5)
                self.assertEqual(missing.exception.code, 404)
            finally:
                server.shutdown()
                server.server_close()
                runner.join(timeout=5)

    def test_http_unavailable_never_leaks_exception(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "missing.jsonl"
            server = ThreadingHTTPServer((HOST, 0), make_handler(path))
            runner = threading.Thread(target=server.serve_forever, daemon=True)
            runner.start()
            try:
                opener = build_opener(ProxyHandler({}))
                with self.assertRaises(HTTPError) as ctx:
                    opener.open(f"http://127.0.0.1:{server.server_port}{ROUTE}", timeout=5)
                self.assertEqual(ctx.exception.code, 503)
                body = json.load(ctx.exception)
                self.assertEqual(body, {"status": "forward_evidence_unavailable"})
                self.assertNotIn(str(path), json.dumps(body))
                self.assertFalse(path.exists())
            finally:
                server.shutdown()
                server.server_close()
                runner.join(timeout=5)


if __name__ == "__main__":
    unittest.main()
