# OurAI Forward Reliability — preflight v1

The frozen v3 magnitude predictor **does not predict UP/DOWN and never trades**.
This read-only preflight checks the existing v2 bundle, journal, Alpaca IEX
access and a small temporary append+fsync benchmark. It does not edit the
champion, frozen protocol or real append-only prediction journal.

## Run from Windows PowerShell

Run in the OurAI folder with the project virtual environment active.

**1. Offline check (can run on Sunday and needs no Alpaca credentials):**

\`\`\`powershell
python -m scripts.check_v3_forward_readiness --offline
\`\`\`

Expect \`LOCAL_OK_MARKET_NOT_CHECKED\`: model + journal checks succeeded,
but this is **not** evidence that the data feed works.

**2. Online check (requires local Alpaca environment variables):**

\`\`\`powershell
python -m scripts.check_v3_forward_readiness
\`\`\`

It checks the configured seven tickers using the frozen **IEX, 1Min, regular**
feed and never prints or persists credentials. Environment variable names are
\`ALPACA_API_KEY_ID\` and \`ALPACA_API_SECRET_KEY\` (the \`APCA_...\` aliases
also work). Configure them locally; **never paste the values into chat or PRs**.

On a weekend or before NY market opening it queries a small window near the
previous weekday's market close. A successful \`HISTORICAL_DATA_OK_LIVE_UNTESTED\`
confirms only historical IEX access for the symbols, **not** that Monday's
live feed is reliable, that SIP data are available, or that the exchange is open.
Holidays are not included in the weekday-only probe selector.

During a normal NY regular session after ~09:33 ET, run it again:

\`\`\`powershell
python -m scripts.check_v3_forward_readiness --output runs\first-real-15m-v3\forward_preflight.json
\`\`\`

Only \`LIVE_PROBE_OK\` demonstrates that all frozen symbols have recently
observed IEX bars. Even this does not validate future accuracy or the
availability of every individual 1-minute bar.

**3. If checks succeed, start the original shadow service:**

\`\`\`powershell
python -m scripts.run_v3_forward_shadow --mode serve --poll-seconds 20
\`\`\`

Check recording without changing the journal:

\`\`\`powershell
python -m scripts.run_v3_forward_shadow --mode status
\`\`\`

The frozen v2 forward start is 2026-10-19 13:30 UTC. The collector must
be running during market hours on the local PC for truly forward predictions.
Do **not** backfill missed predictions or change the frozen forward start.

## Error interpretation

| Result | Meaning |
| --- | --- |
| \`MISSING_ALPACA_CREDENTIALS\` | Local Alpaca credentials not visible in current PowerShell process |
| \`PROBE_ERROR\` + \`HTTP_401\` / \`HTTP_403\` | Authentication or feed entitlement is likely missing |
| \`NO_HISTORICAL_BARS\` or \`NO_USABLE_BARS\` | No regular-session bars for that probe window; check time/feed/market holiday |
| \`LIVE_COVERAGE_INCOMPLETE_OR_STALE\` | At least one symbol missing or has a bar too old to declare live readiness |
| \`HISTORICAL_DATA_OK_LIVE_UNTESTED\` | Historical probe is successful, live not yet checked |
| \`LOCAL_OK_MARKET_NOT_CHECKED\` | Offline: artifact/journal tests only |
| \`LIVE_PROBE_OK\` | IEX bars observed recently for all seven symbols; not a guarantee of trading success |

The report shows per-symbol latest source-bar left-edge timestamp, observed
missing/stale tickers and provider round-trip latency. The benchmark appends
100 **synthetic** events into a removable temporary directory, measures average
and p95 append latency including fsync, and removes the test file. Pass
\`--benchmark-events 0\` to skip it.

**Known performance limitation:** the existing append-only journal verifies
the full hash chain before *each* append. Per-event work therefore increases
with the growing journal. The preflight detects baseline latency but is not
itself an optimization or a promise that a full trading session will sustain
7-symbol throughput. Journal optimization requires a separate carefully
tested change preserving tamper evidence and recovery properties.

The exit status is nonzero for local check failures or insufficient online
coverage. The machine's credentials are never copied into GitHub CI, which
exercises only fully synthetic unit tests.
