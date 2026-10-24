"""Synthetic, offline diagnostics for shadow collector transport failures."""
from __future__ import annotations

import socket
import unittest
from urllib.error import HTTPError, URLError

from scripts.run_v3_forward_shadow import shadow_failure_category


class ShadowFailureCategoryTests(unittest.TestCase):
    def test_dns_getaddrinfo_failure_is_not_a_model_error(self):
        error = URLError(socket.gaierror(11001, "getaddrinfo failed"))
        self.assertEqual(shadow_failure_category(error), "provider_dns_failure")

    def test_windows_10060_and_socket_timeouts_are_provider_timeouts(self):
        for error in (
            URLError(OSError(10060, "connection timed out")),
            TimeoutError("timed out"),
        ):
            with self.subTest(error=type(error).__name__):
                self.assertEqual(shadow_failure_category(error), "provider_timeout")

    def test_http_and_connection_failures_are_distinct(self):
        self.assertEqual(
            shadow_failure_category(HTTPError("https://example.invalid", 503, "unavailable", {}, None)),
            "provider_http_error",
        )
        self.assertEqual(
            shadow_failure_category(URLError("connection reset")),
            "provider_connection_failure",
        )

    def test_unknown_cycle_failure_does_not_claim_model_failure(self):
        error = ValueError("sensitive-local-path-or-token")
        self.assertEqual(shadow_failure_category(error), "cycle_failure_unclassified")

    def test_exception_cause_chain_is_inspected(self):
        inner = socket.gaierror(11001, "getaddrinfo failed")
        outer = RuntimeError("wrapped network failure")
        outer.__cause__ = inner
        self.assertEqual(shadow_failure_category(outer), "provider_dns_failure")


if __name__ == "__main__":
    unittest.main()
