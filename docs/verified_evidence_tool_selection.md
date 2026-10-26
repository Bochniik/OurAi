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
longer names as an overall forward-journal question.
The same bounded guard now recognizes **explicit Slovak company cues**
(`spoločnosť`, `spoločnosti`, `spoločnosťou`, their unaccented
spellings, and inflected `firma` forms). For example, `shadow
spoločnosti Alphabet` or `forward predikcie firmy IBM` must not
silently select an overall journal report. Existing verified Nvidia,
Tesla and Microsoft aliases, and frozen tickers, remain eligible after
these cues. This is a fail-closed lexical check, **not** a new company
resolver or permission to read additional evidence sources. This remains a bounded
fail-closed guard, **not** a general company-name-to-ticker resolver.
This intentionally bounded syntax does **not** detect every possible
unmarked or lowercase ticker reference and does not extend the supported
universe. Unknown tickers are never passed as reader arguments. A rejected proposal falls through to ordinary conversation,
not a fabricated per-stock statistic. Tests use synthetic JSON proposals;
no local Qwen benchmark was performed.

Offline regression:
`python -m unittest tests.test_verified_forward_multi_symbol_selection tests.test_verified_forward_unsupported_ticker_selection tests.test_verified_forward_nvidia_alias tests.test_verified_forward_tesla_alias tests.test_verified_forward_microsoft_alias tests.test_verified_forward_company_cues -v`.

## Slovak alternative-stock questions

The opt-in forward selector now also treats the Slovak alternatives
`alebo` and `či` as potential multi-stock requests. A question such as
`Aké máš výsledky shadow pre Nvidia alebo Alphabet?` or
`Aké máš výsledky shadow pre NVDA alebo Alphabet?` must **not** be
silently narrowed to NVDA, because Alphabet is not a verified alias or
frozen-universe ticker. The same applies with reversed ordering and
lowercase unknown symbols. Questions naming two distinct verified
companies remain unsupported by the single-stock selector.

Equivalent aliases for the **same** frozen ticker (for example
`Nvidia alebo NVDA`) remain eligible, as do metric abbreviations such
as `NVDA alebo MAE`. This is a conservative lexical guard, not
a general natural-language entity resolver; unknown symbols are never
passed to the journal reader. The historical reader and all trading
or forward-journal operations are unchanged.

Offline regression:
`python -m unittest tests.test_verified_forward_disjunction -v`.
These tests use synthetic model proposals, **not** a local Qwen run.

## Verified forward ticker-switch follow-ups

A short Slovak follow-up such as `A čo TSLA?` or `A čo pre Teslu?`
may preserve the **metric** of the immediately preceding verified forward
answer and switch only to the explicitly named, frozen-universe ticker.
For example, after a verified NVDA forward performance answer, `A čo
TSLA?` re-reads TSLA **performance** from the same verified journal;
after a verified latest-prediction answer, it re-reads the latest TSLA
record. The new answer is not copied from prior assistant text.

This requires an immediately preceding `verified_forward_journal`
assistant turn with a recorded `evidence_question`. Both direct
deterministic forward answers and opt-in selected forward answers now
record that provenance in conversation history. No historical result,
unverified model prose, older non-adjacent turn or unrecognized stock
can authorize a follow-up. Longer, ambiguous, multi-stock, educational
or trading requests do not inherit the prior ticker or metric.
The existing verified read-only journal reader remains authoritative;
no new data source, ticker, journal mutation or model permission is added.

Offline regression:
`python -m unittest tests.test_verified_forward_ticker_followups -v`.
The tests use a synthetic mocked journal reader and synthetic classifier
proposal; no local Qwen benchmark was run.

## Verified forward latest-record follow-ups

After an immediately preceding **verified single-ticker forward** answer,
a short Slovak question such as `A aká bola posledná predikcia?` or
`A čo posledná predikcia?` retains that one ticker and requests its
**latest recorded forward magnitude prediction**. This bounded pattern is
resolved **before** the ordinary deterministic forward reader, which would
otherwise interpret a ticker-free latest question as the overall journal's
latest record. The newly constructed canonical question re-reads the existing
integrity-checked journal; no previous numeric answer is reused.

This is **not** general conversational entity resolution. The immediately
preceding assistant turn must carry `source=verified_forward_journal` and
a valid `evidence_question` identifying exactly one frozen-universe ticker
(or curated alias). A mismatched, multi-stock or unsupported original user
request is rejected. Untagged model prose, historical reports, older turns,
questions naming a different ticker, lockbox comparisons and trading requests
do not inherit context. If the conditions are absent, the existing routing
behavior remains unchanged. No new tool, data source, prediction, or
historical-to-forward conversion is introduced.

Offline regression:
`python -m unittest tests.test_verified_forward_latest_followups -v`.
The tests mock the journal reader; **no local Qwen benchmark** was run.

## Verified forward performance follow-ups

After an immediately preceding **source-tagged, single-ticker forward** answer,
a short Slovak question such as `A aká bola jej presnosť?`,
`A aká bola presnosť predikcií?` or `A čo ich presnosť?`
may retain the ticker and request the **per-ticker aggregate forward
magnitude-error metrics**. The canonical question is passed to the existing
integrity-checked forward reader; no numeric values or claims are inherited
from earlier assistant prose. In particular, this is **not** directional
accuracy, profitability or an evaluation of one pending prediction.

This narrow feature shares the latest-record follow-up's provenance guard:
the immediately preceding assistant answer must be tagged
`verified_forward_journal`, carry a valid single-ticker
`evidence_question`, and not contradict the preceding user request.
Unverified/model-generated answers, historical v3 reports, ambiguous or
unsupported tickers, non-adjacent turns, historical/lockbox comparisons,
explicit stock switches and trading requests do not authorize inheritance.
The feature does not add any reader, symbol, source, or permission; it does
not modify models, journal, protocol or execution. Existing deterministic
routes remain authoritative.

Offline regression:
`python -m unittest tests.test_verified_forward_performance_followups -v`.
Tests use a synthetic mocked reader; **no local Qwen benchmark** was run.
