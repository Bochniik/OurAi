"""Loopback OurAI-to-Orion bridge: synthetic, read-only contract tests."""
from __future__ import annotations

from http.server import ThreadingHTTPServer
import json
from pathlib import Path
import tempfile
import threading
from threading import Event
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

    def test_nested_report_metadata_is_not_exported(self):
        # The analytics report can gain private or research-only fields.
        # Neither nested symbol metrics nor latest forecasts may leak them.
        report = self.fake_report()
        report["symbols"]["NVDA"]["raw_credential"] = "synthetic-private-value"
        report["symbols"]["NVDA"]["p_up"] = 0.99
        report["latest_by_symbol"]["NVDA"]["raw_journal_record"] = {
            "local_path": "synthetic-private-value",
        }
        report["latest_by_symbol"]["NVDA"]["p_down"] = 0.01
        with patch("scripts.serve_orion_forward_bridge.read_forward_report", return_value=report):
            snapshot = forward_evidence_snapshot("not-opened-in-mock.jsonl")
        self.assertEqual(set(snapshot["symbols"]["NVDA"]), {
            "predictions", "resolved", "pending", "overdue_pending",
            "champion_mae_bps", "linear_mae_bps", "null_mae_bps",
        })
        self.assertEqual(set(snapshot["latest_by_symbol"]["NVDA"]), {
            "symbol", "predicted_at", "sample_timestamp", "target_due_at",
            "expected_magnitude_bps", "resolved",
        })
        encoded = json.dumps(snapshot)
        self.assertNotIn("synthetic-private-value", encoded)
        self.assertNotIn("raw_journal_record", encoded)
        self.assertNotIn("p_up", encoded)
        self.assertNotIn("p_down", encoded)

    def test_missing_required_nested_field_fails_closed(self):
        report = self.fake_report()
        del report["symbols"]["NVDA"]["predictions"]
        with patch("scripts.serve_orion_forward_bridge.read_forward_report", return_value=report):
            with self.assertRaises(KeyError):
                forward_evidence_snapshot("not-opened-in-mock.jsonl")

    def test_missing_journal_fails_closed(self):
        report = self.fake_report()
        report["journal_exists"] = False
        with patch("scripts.serve_orion_forward_bridge.read_forward_report", return_value=report):
            with self.assertRaises(FileNotFoundError):
                forward_evidence_snapshot("missing.jsonl")

    def test_journal_verification_flags_require_literal_true(self):
        """Truthy report metadata must not become verified forward evidence."""
        for key, error_type in (
            ("journal_exists", FileNotFoundError),
            ("journal_valid", ValueError),
        ):
            for invalid in (1, "true", "false", [], None):
                with self.subTest(field=key, invalid=invalid):
                    report = self.fake_report()
                    report[key] = invalid
                    with patch(
                        "scripts.serve_orion_forward_bridge.read_forward_report",
                        return_value=report,
                    ):
                        with self.assertRaises(error_type):
                            forward_evidence_snapshot("synthetic-journal.jsonl")

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

    def test_host_header_requires_actual_loopback_port_and_no_duplicates(self):
        # Host checks must run BEFORE touching a potentially large hash-chain
        # journal. A spoofed port or ambiguous Host cannot trigger verification.
        import http.client

        with patch(
            "scripts.serve_orion_forward_bridge.read_forward_report",
            side_effect=AssertionError("invalid Host reached journal"),
        ) as read_report:
            server = ThreadingHTTPServer((HOST, 0), make_handler("mock-journal.jsonl"))
            runner = threading.Thread(target=server.serve_forever, daemon=True)
            runner.start()
            url = f"http://127.0.0.1:{server.server_port}{ROUTE}"
            opener = build_opener(ProxyHandler({}))
            try:
                for host in ("127.0.0.1:0", "localhost:not-a-port", "127.0.0.1"):
                    with self.subTest(host=host):
                        with self.assertRaises(HTTPError) as caught:
                            opener.open(Request(url, headers={"Host": host}), timeout=5)
                        self.assertEqual(caught.exception.code, 403)
                        self.assertEqual(json.load(caught.exception), {"status": "forbidden_host"})

                connection = http.client.HTTPConnection(HOST, server.server_port, timeout=5)
                try:
                    connection.putrequest("GET", ROUTE, skip_host=True)
                    connection.putheader("Host", f"127.0.0.1:{server.server_port}")
                    connection.putheader("Host", f"localhost:{server.server_port}")
                    connection.endheaders()
                    response = connection.getresponse()
                    self.assertEqual(response.status, 403)
                    self.assertEqual(json.loads(response.read()), {"status": "forbidden_host"})
                finally:
                    connection.close()
                read_report.assert_not_called()
            finally:
                server.shutdown()
                server.server_close()
                runner.join(timeout=5)

    def test_cross_origin_browser_requests_do_not_read_forward_journal(self):
        # No CORS headers is insufficient: browser GETs still trigger work.
        # Even a direct navigation (Sec-Fetch-Site: none) must not verify the
        # full journal; only backend-to-backend requests are supported.
        with patch(
            "scripts.serve_orion_forward_bridge.read_forward_report",
            side_effect=AssertionError("cross-origin request reached journal"),
        ) as read_report:
            server = ThreadingHTTPServer((HOST, 0), make_handler("mock-journal.jsonl"))
            runner = threading.Thread(target=server.serve_forever, daemon=True)
            runner.start()
            try:
                url = f"http://127.0.0.1:{server.server_port}{ROUTE}"
                opener = build_opener(ProxyHandler({}))
                for headers in (
                    {"Origin": "https://example.invalid"},
                    {"Origin": "null"},
                    {"Sec-Fetch-Site": "cross-site"},
                    {"Sec-Fetch-Site": "same-site"},
                    {"Sec-Fetch-Site": "same-origin"},
                    {"Sec-Fetch-Site": "none", "Sec-Fetch-Mode": "navigate"},
                    {"Sec-Fetch-Site": "unexpected"},
                ):
                    with self.subTest(headers=headers):
                        with self.assertRaises(HTTPError) as caught:
                            opener.open(Request(url, headers=headers), timeout=5)
                        self.assertEqual(caught.exception.code, 403)
                        self.assertEqual(
                            json.load(caught.exception),
                            {"status": "forbidden_browser_origin"},
                        )
                read_report.assert_not_called()
            finally:
                server.shutdown()
                server.server_close()
                runner.join(timeout=5)


    def test_concurrent_refresh_does_not_duplicate_journal_verification(self):
        # Two loopback readers must not cause simultaneous full hash-chain
        # verification. The second fails closed while the first is blocked.
        started, release = Event(), Event()
        results = {}
        def slow_report(_path):
            started.set()
            if not release.wait(timeout=5):
                raise TimeoutError("synthetic blocked journal verification")
            return self.fake_report()

        with patch(
            "scripts.serve_orion_forward_bridge.read_forward_report",
            side_effect=slow_report,
        ) as read_report:
            server = ThreadingHTTPServer((HOST, 0), make_handler("mock-journal.jsonl"))
            runner = threading.Thread(target=server.serve_forever, daemon=True)
            runner.start()
            url = f"http://127.0.0.1:{server.server_port}{ROUTE}"

            def first_reader():
                try:
                    with build_opener(ProxyHandler({})).open(url, timeout=5) as response:
                        results["first_status"] = response.status
                        results["first_body"] = json.load(response)
                except Exception as error:
                    results["first_error"] = repr(error)

            first = threading.Thread(target=first_reader, daemon=True)
            try:
                first.start()
                self.assertTrue(started.wait(timeout=3), "first verification did not start")
                with self.assertRaises(HTTPError) as busy:
                    build_opener(ProxyHandler({})).open(url, timeout=5)
                self.assertEqual(busy.exception.code, 503)
                self.assertEqual(
                    json.load(busy.exception),
                    {"status": "forward_evidence_unavailable"},
                )
                read_report.assert_called_once()
            finally:
                release.set()
                first.join(timeout=5)
                server.shutdown()
                server.server_close()
                runner.join(timeout=5)
            self.assertFalse(first.is_alive())
            self.assertNotIn("first_error", results)
            self.assertEqual(results["first_status"], 200)
            self.assertEqual(results["first_body"]["resolution_count"], 1)
            read_report.assert_called_once()

    def test_failed_verification_releases_gate_for_next_refresh(self):
        # A bad/missing journal cannot permanently block later requests.
        with patch(
            "scripts.serve_orion_forward_bridge.read_forward_report",
            side_effect=[FileNotFoundError("synthetic missing"), self.fake_report()],
        ) as read_report:
            server = ThreadingHTTPServer((HOST, 0), make_handler("mock-journal.jsonl"))
            runner = threading.Thread(target=server.serve_forever, daemon=True)
            runner.start()
            url = f"http://127.0.0.1:{server.server_port}{ROUTE}"
            opener = build_opener(ProxyHandler({}))
            try:
                with self.assertRaises(HTTPError) as unavailable:
                    opener.open(url, timeout=5)
                self.assertEqual(unavailable.exception.code, 503)
                self.assertEqual(
                    json.load(unavailable.exception),
                    {"status": "forward_evidence_unavailable"},
                )
                with opener.open(url, timeout=5) as response:
                    self.assertEqual(response.status, 200)
                    self.assertEqual(json.load(response)["resolution_count"], 1)
                self.assertEqual(read_report.call_count, 2)
            finally:
                server.shutdown()
                server.server_close()
                runner.join(timeout=5)

    def test_verification_exhaustion_returns_safe_503_then_recovers(self):
        # Deeply nested or oversized journal data can exhaust parser limits.
        # A failed verification is an unavailable snapshot, not a broken socket,
        # and it must release the gate for the next legitimate refresh.
        with patch(
            "scripts.serve_orion_forward_bridge.read_forward_report",
            side_effect=[
                RecursionError("synthetic nested journal"),
                OverflowError("synthetic oversized numeric value"),
                self.fake_report(),
            ],
        ) as read_report:
            server = ThreadingHTTPServer((HOST, 0), make_handler("mock-journal.jsonl"))
            runner = threading.Thread(target=server.serve_forever, daemon=True)
            runner.start()
            opener = build_opener(ProxyHandler({}))
            url = f"http://127.0.0.1:{server.server_port}{ROUTE}"
            try:
                for _ in range(2):
                    with self.assertRaises(HTTPError) as caught:
                        opener.open(url, timeout=5)
                    self.assertEqual(caught.exception.code, 503)
                    self.assertEqual(caught.exception.headers["Cache-Control"], "no-store")
                    self.assertEqual(
                        json.load(caught.exception),
                        {"status": "forward_evidence_unavailable"},
                    )
                with opener.open(url, timeout=5) as response:
                    self.assertEqual(response.status, 200)
                    self.assertEqual(json.load(response)["resolution_count"], 1)
                self.assertEqual(read_report.call_count, 3)
            finally:
                server.shutdown()
                server.server_close()
                runner.join(timeout=5)

    def test_unserializable_snapshot_returns_safe_503_then_recovers(self):
        # Corrupt/non-JSON analytics must be classified as unavailable, not
        # streamed partially or surfaced as an unexplained broken connection.
        invalid_number = self.fake_report()
        invalid_number["champion_mae_bps"] = float("nan")
        invalid_object = self.fake_report()
        invalid_object["symbols"]["NVDA"]["champion_mae_bps"] = object()

        with patch(
            "scripts.serve_orion_forward_bridge.read_forward_report",
            side_effect=[invalid_number, invalid_object, self.fake_report()],
        ) as read_report:
            server = ThreadingHTTPServer((HOST, 0), make_handler("mock-journal.jsonl"))
            runner = threading.Thread(target=server.serve_forever, daemon=True)
            runner.start()
            opener = build_opener(ProxyHandler({}))
            url = f"http://127.0.0.1:{server.server_port}{ROUTE}"
            try:
                for _ in range(2):
                    with self.assertRaises(HTTPError) as caught:
                        opener.open(url, timeout=5)
                    self.assertEqual(caught.exception.code, 503)
                    self.assertEqual(caught.exception.headers["Cache-Control"], "no-store")
                    self.assertEqual(
                        json.load(caught.exception),
                        {"status": "forward_evidence_unavailable"},
                    )
                with opener.open(url, timeout=5) as response:
                    self.assertEqual(response.status, 200)
                    self.assertEqual(json.load(response)["resolution_count"], 1)
                self.assertEqual(read_report.call_count, 3)
            finally:
                server.shutdown()
                server.server_close()
                runner.join(timeout=5)

    def test_disconnected_loopback_reader_does_not_crash_handler_or_poison_next_get(self):
        # A short Orion timeout can close the socket after verification but
        # before HTTP headers are sent. Simulate the write failure reliably,
        # without touching a real journal or creating any forward evidence.
        with patch(
            "scripts.serve_orion_forward_bridge.read_forward_report",
            return_value=self.fake_report(),
        ) as read_report:
            handler = make_handler("mock-journal.jsonl")
            original_end_headers = handler.end_headers
            disconnected = Event()

            def disconnected_first_reader(self):
                if not disconnected.is_set():
                    disconnected.set()
                    raise BrokenPipeError("synthetic closed loopback socket")
                return original_end_headers(self)

            server = ThreadingHTTPServer((HOST, 0), handler)
            runner = threading.Thread(target=server.serve_forever, daemon=True)
            runner.start()
            opener = build_opener(ProxyHandler({}))
            url = f"http://127.0.0.1:{server.server_port}{ROUTE}"
            try:
                with patch.object(handler, "end_headers", disconnected_first_reader), \
                        patch.object(server, "handle_error") as unhandled_error:
                    with self.assertRaises(OSError):
                        opener.open(url, timeout=5)
                    self.assertTrue(disconnected.is_set())
                    with opener.open(url, timeout=5) as response:
                        self.assertEqual(response.status, 200)
                        self.assertEqual(json.load(response)["resolution_count"], 1)
                    unhandled_error.assert_not_called()
                self.assertEqual(read_report.call_count, 2)
            finally:
                server.shutdown()
                server.server_close()
                runner.join(timeout=5)

    def test_disconnected_reader_during_verification_does_not_poison_gate(self):
        # Orion may hit its HTTP timeout while the producer is still
        # verifying the append-only journal. The next refresh must not
        # inherit a locked gate or receive a cached/stale snapshot.
        import socket

        started, release, completed = Event(), Event(), Event()

        def blocked_report(_path):
            started.set()
            if not release.wait(timeout=5):
                raise TimeoutError("synthetic verification timeout")
            return self.fake_report()

        with patch(
            "scripts.serve_orion_forward_bridge.read_forward_report",
            side_effect=blocked_report,
        ) as read_report:
            handler = make_handler("mock-journal.jsonl")
            original_json = handler._json

            def track_response(self, code, payload):
                try:
                    return original_json(self, code, payload)
                finally:
                    if code == 200:
                        completed.set()

            server = ThreadingHTTPServer((HOST, 0), handler)
            runner = threading.Thread(target=server.serve_forever, daemon=True)
            runner.start()
            url = f"http://127.0.0.1:{server.server_port}{ROUTE}"
            opener = build_opener(ProxyHandler({}))
            reader = None
            try:
                with patch.object(handler, "_json", track_response), \
                        patch.object(server, "handle_error") as unhandled_error:
                    reader = socket.create_connection((HOST, server.server_port), timeout=3)
                    reader.sendall(
                        f"GET {ROUTE} HTTP/1.1\r\n"
                        f"Host: 127.0.0.1:{server.server_port}\r\n"
                        "Connection: close\r\n\r\n"
                    .encode("ascii"))
                    self.assertTrue(started.wait(timeout=3))
                    # Close the first client's socket BEFORE verification
                    # finishes, not merely during response serialization.
                    reader.close()
                    reader = None
                    with self.assertRaises(HTTPError) as busy:
                        opener.open(url, timeout=5)
                    self.assertEqual(busy.exception.code, 503)
                    self.assertEqual(
                        json.load(busy.exception),
                        {"status": "forward_evidence_unavailable"},
                    )
                    read_report.assert_called_once()

                    release.set()
                    self.assertTrue(completed.wait(timeout=5))
                    with opener.open(url, timeout=5) as recovered:
                        self.assertEqual(recovered.status, 200)
                        self.assertEqual(json.load(recovered)["resolution_count"], 1)
                    self.assertEqual(read_report.call_count, 2)
                    unhandled_error.assert_not_called()
            finally:
                release.set()
                if reader is not None:
                    reader.close()
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
