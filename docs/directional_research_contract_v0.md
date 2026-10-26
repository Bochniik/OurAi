# Directional research-only contract v0 (NOT validated)

This is a **standalone, opt-in-by-import, data-shape validator** for a possible
future OurAI direction/signed-return research output. It is **not** part of the
current OurAI forward-journal HTTP producer, Orion bridge, production engines,
or any order/execution path. No real predictions are created by this change.

The existing trusted-loopback read-only bridge remains **magnitude-only**:
`direction_available=false`, `trading_enabled=false`. Never put this
research payload in `/v1/forward-evidence` or present it as verified
forward evidence. No migration/promotion is implied.

## Contract

`research/directional_research_contract_v0.py` provides
`validate_directional_research_v0(payload)`. The function accepts a flat
dictionary with exactly the following fields (no extensions):

| Field | Requirement |
| --- | --- |
| `contract` | `ourai_directional_research_v0` |
| `schema_version` | integer `0` |
| `mode` | `RESEARCH_ONLY` |
| `validation_status` | `NOT_INDEPENDENTLY_VALIDATED` |
| `evidence_status` | `UNVERIFIED_MODEL_OUTPUT` |
| `research_only`, `trading_enabled` | literal `true`, literal `false` |
| `experiment_id` | bounded opaque research identifier, not a path |
| `symbol` | one of NVDA, AAPL, TSLA, JPM, AMD, META, MSFT |
| `horizon_minutes` | integer `15` |
| `as_of_utc`, `target_due_at_utc` | explicit UTC Z timestamps, exactly 15 minutes apart |
| `probability_conditioning` | `up_vs_down_given_nonzero_15m_return` |
| `p_up`, `p_down` | finite probabilities in [0,1], sum to 1 (tolerance 1e-6) |
| `expected_signed_change_pct` | finite signed number in **percentage points** (e.g. -0.12 means -0.12%, not -12%) |

The binary probability pair is **conditional on a nonzero return**: a flat
15-minute outcome is outside that binary event. The signed percent estimate
is a separate research output; the validator does not establish calibration,
economic coherence, accuracy, or profitability. Its output is still
**unverified** even if the shape passes. All numerical test examples are
synthetic, not forward evidence or a benchmark.

## Guardrails and validation gate

Unknown fields (including `trade_signal`, `order`, `journal_valid`,
`direction_available`, `accuracy`), bad timestamps, nonfinite values,
wrong probability sums, production labels and any claim of independent
validation fail closed. The function performs no I/O.

A future independently evaluated model would require a **separate explicit
reviewed contract and release gate**; changing a flag here cannot authorize
production trading. Keep the v3 frozen champion, append-only hash-chain
journal, lockbox, datasets, credentials and existing engines untouched.

Offline synthetic regression:
`python -m unittest tests.test_directional_research_contract_v0 -v`.
