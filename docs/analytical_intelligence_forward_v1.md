# OurAI Analytical Intelligence v1 — read-only forward journal bridge

OurAI can now answer a limited set of questions based on the **actual local v2
forward-shadow journal** for frozen v3 **15-minute absolute-return magnitude**
predictions. The journal reader runs independently of the LLM, Alpaca API,
training datasets and trading code; no background monitoring or trading occurs.

## Windows PowerShell

After the PR is merged, run from the OurAI repository:

```powershell
git pull --ff-only
python -m unittest tests.test_forward_analytics_chat -v
python -m scripts.chat_ourai --backend ollama --model qwen3:4b-instruct-2507-q4_K_M --num-ctx 1536 --max-new-tokens 240 --concise --forward-journal runs\first-real-15m-v3\forward_shadow_v2\events.jsonl
```

Try these questions:

```text
Aké sú tvoje posledné predikcie?
Aký 15-minútový pohyb očakávaš pri NVDA?
Ako presné boli tvoje posledné predikcie?
Pri ktorých akciách robíš najväčšie chyby?
Ako presné boli predikcie AMD?
```

These answers display `[source: verified local v2 forward journal; read-only, magnitude-only]`
and **do not invoke the generative model**. Without `--forward-journal`,
there is no connection to the local files. The option does not start a
collector or create a prediction. If the file is missing or empty, the
assistant reports that no observations are available. On a damaged, mixed
protocol or contradictory journal, the assistant reports an error instead
of fabricating a summary; it does not edit or repair the file.

## What is measured

The bridge fully verifies the existing append-only hash chain, checks causal
timestamps and event relationships, and recomputes recorded absolute errors
from each prediction's magnitude vs the resolved **actual absolute magnitude**.
Only a prediction with a separate, valid resolution contributes to:

- v3 champion mean absolute error (MAE), **basis points (bps)**;
- frozen linear comparison MAE, bps;
- frozen median-null comparison MAE, bps;
- counts of logged, resolved, pending and overdue-pending events per symbol.

It reports latest observed *logged* prediction with UTC timestamp and age
context, not an automatically refreshed price or guaranteed live market
observation. 100 bps = 1 percentage point of price change; the v3 forecast
does **not** say UP or DOWN. No claims about probability of a rise/fall,
trade signals, realized trading profitability or statistical significance
are made. Ranking per-symbol errors is descriptive and especially unstable
with very few resolved observations.

The v2 evaluation plan still requires a future forward observation period,
and this UI feature **does not bypass** the preregistered acceptance rules.
Historical holdout/test datasets are neither opened nor used.

## Important isolation

The existing 8-question Slovak finance benchmark still measures the **raw
Qwen model**, without the curated knowledge or forward journal. Its default
constructor never loads the journal. When `--raw-llm` is used, interactive
chat also disables the optional bridge, so results are not confused with
the LLM's own abilities.

The authoritative input is an append-only local JSONL file, **not a signed
independent market-data feed**: hash-chain verification detects inconsistent
modifications but does not independently attest price-provider truth.
A single writer should keep the journal up to date. The reader does not
write to the journal and may refuse to read it during an interrupted
concurrent append. Do not delete/repair a damaged journal without separately
preserving the evidence.
