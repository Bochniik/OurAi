# OurAI → OrionTrader, integration phase 1: read-only forward evidence

**Purpose:** Let a local OrionTrader backend display VERIFIED, frozen OurAI V3
15-minute absolute magnitude predictions and resolved error summaries.
This is NOT an UP/DOWN engine or an order signal. Do not replace Orion's
scanner, decision policy, or execution model.

## OurAI on Windows (terminal A)

From the `OurAI` repository with its virtual environment activated:

```powershell
git checkout main
git pull origin main
python -m scripts.serve_orion_forward_bridge --port 8765
```

Leave the server running. It reads the existing, append-only journal from
`runs/first-real-15m-v3/forward_shadow_v2/events.jsonl` in the current
OurAI working directory. Override `--journal` only for deliberate local
testing. It never writes that journal, retrains, executes trading, requests
Alpaca bars, or reads the sealed holdout.

Loopback endpoint:

```text
GET http://127.0.0.1:8765/v1/forward-evidence
```

It returns an explicit schema with magnitude-only metadata, resolved MAE for
champion/linear/null, pending/overdue counts, per-symbol summaries and latest
logged magnitude predictions with timestamps. Missing/corrupted journal returns
HTTP 503 without leaking local paths or secrets; unknown routes return 404.

The server binds ONLY to IPv4 127.0.0.1, has no CORS opt-in, checks Host header,
and should **not** be reverse-proxied or exposed to a LAN/public network.
Run OurAI and the OrionTrader backend on the same trusted Windows machine.
This local read-only service does not authenticate individual users, so the
consuming OrionTrader endpoint must remain explicitly opt-in and should not be
exposed to additional accounts on a multi-user deployment.

## Operational semantics

- `journal_valid=true`: OurAI has checked the hash chain, causal
  timestamps, prediction/resolution consistency, and recorded errors.
- `direction_available=false`: a predicted 15-minute absolute return
  does NOT imply positive or negative price movement.
- `prediction_count` includes unresolved entries. MAE uses resolved
  predictions ONLY, with the same stored frozen V3, linear and null models.
- Predictions can remain pending after a PC sleep or if a specific IEX
  target bar is absent. Do not silently fill them with fabricated closes.
- A single trading day is NOT proof of profitability, stable skill or
  independent statistical significance. Autopromotion and trading are disabled.
- Latest-by-symbol timestamps must be displayed and treated as stale when
  appropriate, especially outside regular US trading hours.
- This bridge does not change the frozen V3 forward protocol or the underlying
  journal files.

## Tests

```powershell
python -m unittest discover -s tests -p "test_orion_forward_bridge.py" -v
```

OrionTrader's backend consumes this endpoint in a separate opt-in phase.
