# OurAI Forward Reliability — preflight v1

The frozen v3 magnitude predictor **does not predict UP/DOWN and never trades**.
This read-only preflight checks the existing v2 bundle, journal, Alpaca IEX
access and a small temporary append+fsync benchmark. It does not edit the
champion, frozen protocol or real append-only prediction journal.

## Run from Windows PowerShell

Run in the OurAI folder with the project virtual environment active.

**1. Offline check (can run on Sunday and needs no Alpaca credentials):**

```powershell
python -m scripts.check_v3_forward_readiness --offline
```

Expect `LOCAL_OK_MARKET_NOT_CHECKED`: model + journal checks succeeded,
but this is **not** evidence that the data feed works.

**2. Online check (requires local Alpaca environment variables):**

```powershell
python -m scripts.check_v3_forward_readiness
```

It checks the configured seven tickers using the frozen **IEX, 1Min, regular**
feed and never prints or persists credentials. Environment variable names are
`ALPACA_API_KEY_ID` and `ALPACA_API_SECRET_KEY` (the `APCA_...` aliases
also work). Configure them locally; **never paste the values into chat or PRs**.

On a weekend or before NY market opening it queries a small window near the
previous weekday's market close. A successful `HISTORICAL_DATA_OK_LIVE_UNTESTED`
confirms only historical IEX access for the symbols, **not** that Monday's
live feed is reliable, that SIP data are available, or that the exchange is open.
Holidays are not included in the weekday-only probe selector.

During a normal NY regular session after ~09:33 ET, run it again:

```powershell
python -m scripts.check_v3_forward_readiness --output runs\first-real-15m-v3\forward_preflight.json
```

Only `LIVE_PROBE_OK` demonstrates that all frozen symbols have recently
observed IEX bars. Even this does not validate future accuracy or the
availability of every individual 1-minute bar.

**3. If checks succeed, start the original shadow service:**

```powershell
python -m scripts.run_v3_forward_shadow --mode serve --poll-seconds 20
```

Check recording without changing the journal:

```powershell
python -m scripts.run_v3_forward_shadow --mode status
```

The frozen v2 forward start is 2026-10-19 13:30 UTC. The collector must
be running during market hours on the local PC for truly forward predictions.
Do **not** backfill missed predictions or change the frozen forward start.

## Error interpretation

| Result | Meaning |
| --- | --- |
| `MISSING_ALPACA_CREDENTIALS` | Local Alpaca credentials not visible in current PowerShell process |
| `PROBE_ERROR` + `HTTP_401` / `HTTP_403` | Authentication or feed entitlement is likely missing |
| `NO_HISTORICAL_BARS` or `NO_USABLE_BARS` | No regular-session bars for that probe window; check time/feed/market holiday |
| `LIVE_COVERAGE_INCOMPLETE_OR_STALE` | At least one symbol missing or has a bar too old to declare live readiness |
| `HISTORICAL_DATA_OK_LIVE_UNTESTED` | Historical probe is successful, live not yet checked |
| `LOCAL_OK_MARKET_NOT_CHECKED` | Offline: artifact/journal tests only |
| `LIVE_PROBE_OK` | IEX bars observed recently for all seven symbols; not a guarantee of trading success |

The report shows per-symbol latest source-bar left-edge timestamp, missing /
stale tickers and provider round-trip latency. The benchmark now compares
100 **synthetic single-event appends** against a **single batch of 100 synthetic
events**, each written to its own removable temporary JSONL journal. It reports
`single_total_ms` and `batch_total_ms` on the current machine as well as
legacy per-event latency. Pass `--benchmark-events 0` to skip the benchmark.

**Updated journal writer:** forward cycles now prepare prediction records and
subsequent resolution records in memory and append each group as an ordered
batch, verifying the complete preexisting SHA-256 chain once per batch and
flushing/syncing once per batch. This preserves the exact existing on-disk
schema, hash contents and event order; old journals remain readable. Duplicate
event IDs, invalid event fields and corrupted history are rejected before
appending. It avoids O(N²) within each prediction batch, but full journal
verification still costs O(N) per batch or cycle.

**Important limits:** use **one** shadow runner per journal. This revision does
not implement cross-process file locking or transactional multi-event writes.
A disk error/power interruption can leave an incomplete final JSONL line;
the reader fails closed and never silently truncates, backfills, or rewrites
that evidence. Back up the journal to preserve evidence; manually investigate
a damaged file rather than deleting it. The synthetic benchmark is not a
guarantee of performance during live market-data fetching and resolution.
The exit status is nonzero for local check failures or insufficient online
coverage. The machine's credentials are never copied into GitHub CI, which
exercises only fully synthetic unit tests.
