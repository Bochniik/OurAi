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
comparison. A follow-up guard also rejects **explicit unsupported ticker syntax** (for
example `pre IBM`, `$IBM`, `NVDA oproti IBM`, `NVDA a IBM`, or
`NVDA, IBM`) in this opt-in tool selector. The pair scanner also rejects
**mixed/lowercase unknown tickers next to a recognized frozen ticker** in stock comparisons, such as `NVDA a ibm`,
`ibm oproti NVDA`, or `NVDA, NVDA, ibm`. This is conservative:
ambiguous short words next to known tickers may also cause the optional
tool proposal to be rejected rather than risking an incorrect answer.
The pair scanner also checks
**overlapping pairs in longer lists** (such as `NVDA, NVDA, IBM`), so an
unsupported third ticker cannot be silently skipped. It does not silently discard
an unknown stock and answer for the remaining known ticker or overall
journal. Recognized metric abbreviations such as MAE and BPS are not treated
as stock names. Explicit stock cues such as `ticker ibm`, `symbol ibm`,
`akciu ibm`, or `stock ibm` now also check **lowercase/mixed-case** 2–5
letter candidates, rather than silently selecting an overall or NVDA-only
report. The broad Slovak preposition `pre` remains **uppercase-only** for
unsupported candidates to avoid mistaking ordinary lowercase words for
tickers. An unverified company name such as `akciu Alphabet` is **not** silently
mapped to a ticker by this opt-in selector; without a verified alias it
fails closed. Narrow, manually curated exceptions recognize the Slovak
forms `Nvidia`, `Nvidie`, `Nvidii`, `Nvidiu`, `Nvidiou` as **NVDA**
and `Tesla`, `Tesly`, `Tesle`, `Teslu`, `Teslou` as **TSLA**, plus
`Microsoft`, `Microsoftu`, `Microsofte`, `Microsoftom` as **MSFT**.
All three tickers are already in the frozen journal universe. These aliases apply
to the opt-in tool selector only: they neither add tickers nor change
any evidence reader. A request naming two distinct companies/tickers
(including a verified alias paired with an unsupported name in either
order) is rejected rather than silently narrowed. Overlapping alias
pairs in longer lists are also checked, including repeated names.
Other company names remain unverified, and ambiguous or unrecognized
names must not be guessed. The explicit stock-cue
scanner checks names up to 30 ASCII letters, rather than silently treating
longer names as an overall forward-journal question. This remains a bounded
fail-closed guard, **not** a general company-name-to-ticker resolver.
This intentionally bounded syntax does **not** detect every possible
unmarked or lowercase ticker reference and does not extend the supported
universe. Unknown tickers are never passed as reader arguments. A rejected proposal falls through to ordinary conversation,
not a fabricated per-stock statistic. Tests use synthetic JSON proposals;
no local Qwen benchmark was performed.

Offline regression:
`python -m unittest tests.test_verified_forward_multi_symbol_selection tests.test_verified_forward_unsupported_ticker_selection tests.test_verified_forward_nvidia_alias tests.test_verified_forward_tesla_alias tests.test_verified_forward_microsoft_alias -v`.
