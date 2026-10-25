"""Loopback-only, read-only bridge from OurAI forward evidence to OrionTrader.

This server NEVER runs a model, fetches market data, edits a journal or trades.
The frozen 15m magnitude forward journal is fully verified for every request.
No direction/confidence/trading signal is exposed. Opt-in local desktop use.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from typing import Any

from scripts.run_v3_forward_shadow import DEFAULT_JOURNAL
from trading.forward_analytics import read_forward_report


HOST = "127.0.0.1"
PORT = 8765
ROUTE = "/v1/forward-evidence"

# Whitelist fields at EVERY nesting level. Forward analytics may grow new
# internal metadata; none of it should silently become part of this HTTP API.
_SYMBOL_FIELDS = (
    "predictions", "resolved", "pending", "overdue_pending",
    "champion_mae_bps", "linear_mae_bps", "null_mae_bps",
)
_LATEST_FIELDS = (
    "symbol", "predicted_at", "sample_timestamp", "target_due_at",
    "expected_magnitude_bps", "resolved",
)


def forward_evidence_snapshot(journal_path: str | Path) -> dict[str, Any]:
    """Only expose a tightly scoped view of VERIFIED read-only V3 data."""
    report = read_forward_report(journal_path)
    if not report["journal_exists"]:
        raise FileNotFoundError("forward journal unavailable")
    if not report["journal_valid"]:
        raise ValueError("forward journal validation failed")
    return {
        "schema_version": 1,
        "source": "ourai_verified_forward_journal",
        "experiment": "frozen_v3_magnitude_15m",
        "mode": "RESEARCH_SHADOW_ONLY",
        "trading_enabled": False,
        "horizon_minutes": 15,
        "direction_available": False,
        "journal_valid": True,
        "evaluated_at_utc": report["evaluated_at_utc"],
        "prediction_count": report["prediction_count"],
        "resolution_count": report["resolution_count"],
        "pending_count": report["pending_count"],
        "overdue_pending_count": report["overdue_pending_count"],
        "champion_mae_bps": report["champion_mae_bps"],
        "linear_mae_bps": report["linear_mae_bps"],
        "null_mae_bps": report["null_mae_bps"],
        "symbols": {
            symbol: {field: row[field] for field in _SYMBOL_FIELDS}
            for symbol, row in report["symbols"].items()
        },
        "latest_by_symbol": {
            symbol: {field: row[field] for field in _LATEST_FIELDS}
            for symbol, row in report["latest_by_symbol"].items()
        },
        "limitations": (
            "Absolute 15m price-movement magnitude only; no UP/DOWN odds, "
            "trade signal, production promotion, or proof of trading edge."
        ),
    }


def make_handler(journal_path: str | Path):
    class ForwardEvidenceHandler(BaseHTTPRequestHandler):
        def _json(self, code: int, payload: dict[str, Any]) -> None:
            encoded = json.dumps(payload, allow_nan=False, separators=(",", ":")).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def do_GET(self) -> None:
            # Does not allow remote exposure or browser CORS. HTTP Host
            # must also identify local loopback, to reject DNS-rebinding hosts.
            host_header = self.headers.get("Host", "").split(":")[0].lower()
            if host_header not in {"127.0.0.1", "localhost"}:
                self._json(403, {"status": "forbidden_host"})
                return
            # CORS only prevents a browser from reading a response; it does
            # not prevent it from issuing a GET that verifies the full journal.
            # This endpoint is for the Orion backend, not browsers. Reject
            # *all* Fetch Metadata site values, including "none" (address-bar
            # navigation) and "same-origin", before touching the journal.
            if (
                self.headers.get("Origin") is not None
                or self.headers.get("Sec-Fetch-Site") is not None
            ):
                self._json(403, {"status": "forbidden_browser_origin"})
                return
            if self.path != ROUTE:
                self._json(404, {"status": "not_found"})
                return
            try:
                self._json(200, forward_evidence_snapshot(journal_path))
            except (OSError, ValueError, TypeError, KeyError, AssertionError):
                # Never return raw file contents, local paths or exception
                # messages across the HTTP bridge.
                self._json(503, {"status": "forward_evidence_unavailable"})

        def log_message(self, format_string: str, *args: object) -> None:
            # Avoid logging request headers or any confidential local paths.
            return

    return ForwardEvidenceHandler


def main() -> int:
    parser = argparse.ArgumentParser(description="Serve verified OurAI V3 forward metrics to local OrionTrader.")
    parser.add_argument("--journal", default=DEFAULT_JOURNAL)
    parser.add_argument("--port", type=int, default=PORT)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port must be in 1..65535")
    handler = make_handler(args.journal)
    server = ThreadingHTTPServer((HOST, args.port), handler)
    print(f"OurAI read-only bridge: http://{HOST}:{args.port}{ROUTE}")
    print("Only local connections; no trading, no model updates, no raw journal export.")
    try:
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
