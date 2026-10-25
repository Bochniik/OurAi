# Experimental verified evidence tool selection

OurAI's Ollama chat can now **optionally select among two read-only
verified evidence sources** using a single constrained local Qwen JSON
classification. This is an incremental tool-routing capability, **not** a
general autonomous browsing/agent platform.

## Sources

- `historical_v3`: existing v3 magnitude validation/lockbox JSON reports.
- `forward_v2`: existing append-only, integrity-checked v2 shadow journal.
- `none`: general knowledge questions go to ordinary Qwen conversation.

The model proposes only the tool and a permitted enum action. The Python
dispatcher validates both the original user's wording and the schema. All
financial values come from the established deterministic readers; the
model cannot supply a path, ticker, numerical result, trading command or
other executable argument. Forward and historical claims remain isolated.

## Local command (PowerShell)

```powershell
git pull origin main
python -m scripts.chat_ourai `
  --backend ollama `
  --model qwen3:4b-instruct-2507-q4_K_M `
  --num-ctx 1536 `
  --max-new-tokens 240 `
  --concise `
  --semantic-tool-routing `
  --historical-reports-dir runs\first-real-15m-v3 `
  --forward-journal runs\first-real-15m-v3\forward_shadow_v2\events.jsonl
```

`--semantic-tool-routing` is **separate** from the legacy
`--semantic-routing`: choose exactly one. Both are opt-in and require
the local Ollama backend. It makes one extra local Qwen request on questions
not resolved by existing deterministic routes. This has not been validated
against a live local Qwen4B benchmark: the tests mock its structured output.

Sample tests (without internet or installed model weights):

```powershell
python -m unittest discover -s tests -p "test_verified_tool_selection.py" -v
```

Try asking: `Ako sa ti darí v shadow režime?` or
`Aké máš výsledky shadow pre NVDA?`. Even if the model chooses a forward
tool, an absent journal is reported as **absent**, never replaced by
historical results. `Čo je paper trading?` remains a general question.

This mode never initiates live acquisition, executes trades, opens a
locked dataset, retunes a forecast, or promotes a model. Verified journal
performance cannot be equated with directional accuracy or profitability.


## Ambiguous multi-ticker forward questions

The opt-in forward tool selector builds a **single-ticker** verified-journal
query (or an overall query if no ticker is named). If the user's original
question names two or more **distinct symbols from the frozen universe**
(e.g. `Aké máš výsledky shadow pre NVDA a TSLA?`), it now rejects the
model-proposed forward tool choice rather than silently selecting the first
symbol. A repeated mention of the *same* ticker remains a single-ticker
request. A general shadow-performance question without a named ticker still
uses the existing overall read-only query.

This guard applies to the **optional model-proposed tool selection path**;
it does not change the deterministic forward reader or implement a multi-stock
comparison. Unsupported or unrecognized tickers are not interpreted by this
narrow check. A rejected proposal falls through to ordinary conversation,
not a fabricated per-stock statistic. Tests use synthetic JSON proposals;
no local Qwen benchmark was performed.

Offline regression:
`python -m unittest tests.test_verified_forward_multi_symbol_selection -v`.
