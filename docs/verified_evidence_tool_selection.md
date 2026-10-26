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
a short Slovak question such as `A aká bola posledná predikcia?`,
`A čo posledná predikcia?`, or `A aká bola jej posledná predikcia?`
retains that one ticker and requests its
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

## Verified forward result-explanation follow-ups

After an immediately preceding **source-tagged single-ticker forward
performance** answer, the bounded Slovak follow-ups `Čo tieto výsledky
znamenajú?`, `Čo znamenajú tieto výsledky?` and `Vysvetli mi tieto
výsledky.` re-read that ticker's existing verified journal performance
and append a fixed explanation of magnitude MAE in bps. The response
never copies earlier numerical values or invokes a model to invent a
cause, directional accuracy, or profitability.

The same immediate provenance guard as the latest/performance follow-ups
applies: the assistant source must be `verified_forward_journal`, the
`evidence_question` must identify exactly one supported ticker, and the
original user turn must not contradict it. A prior latest-record answer,
overall performance, historical/lockbox report, unsupported or multiple
tickers, an untagged model answer, and unrelated or causal questions
(`Prečo je chyba taká veľká?`) do **not** authorize this explanation.
No new evidence reader, ticker, prediction, journal write or trade
operation is introduced.

Offline regression:
`python -m unittest tests.test_verified_forward_result_explanations -v`.
Tests use a synthetic reader; no local Qwen benchmark was run.


## Verified latest-prediction explanation follow-ups

After an **immediately preceding, source-tagged, single-ticker latest
forward record**, the narrowly phrased Slovak question `Čo táto predikcia
vlastne znamená?` (also `Čo táto predikcia znamená?` or `Vysvetli mi
túto predikciu.`) re-reads the **same latest ticker record** from the
existing integrity-checked forward journal. A fixed Slovak explanation
then clarifies that magnitude is an **absolute 15-minute price change**,
1 bps is 0.01 percentage point, and the resolution status is not a
measurement of directional accuracy or a trading signal. No numbers are
copied from assistant prose.

This explanation is handled before generic deterministic forward routing:
the word `predikcia` alone could otherwise trigger an overall journal
answer. Eligibility requires the immediate assistant turn to carry
`source=verified_forward_journal`, a single supported ticker in its
`evidence_question`, an explicit latest-record cue in that question,
and a non-contradictory original user turn. Aggregate performance,
historical/lockbox results, generic ticker reports, unknown or multiple
symbols, untagged model text, causal questions and trade requests do
not authorize the follow-up. The feature is a bounded lexical explanation,
not a model-generated causal interpretation, new prediction, tool,
journal write or permission to trade.

Offline regression:
`python -m unittest tests.test_verified_forward_latest_explanations -v`.
Tests use a synthetic reader; **no local Qwen benchmark was run**.


## Ticker-switch provenance consistency

A short follow-up such as `A čo TSLA?` still uses only the
immediately preceding source-tagged verified forward answer. Before
switching, the router now checks that the stored `evidence_question`
does not name multiple or unsupported tickers, and that the original
user question does not name a conflicting, multiple, or unsupported
ticker. A source tag alone does not make contradictory evidence safe.

An overall verified forward performance report with **no ticker in
either the user question or the canonical evidence question** remains
eligible for an explicit ticker switch. Source-bound chains such as
latest NVDA → explanation → `A čo TSLA?` → accuracy continue to
re-read the verified journal on each turn; no figures are inherited.
This is a narrow provenance guard, not a general memory system or a
new permission to select tools.

Offline regression:
`python -m unittest tests.test_verified_forward_ticker_provenance -v`.
The tests use a synthetic reader; no local Qwen benchmark was run.

## Cross-source provenance guard for forward follow-ups

A stored `source=verified_forward_journal` tag cannot by itself authorize
inheriting a ticker from a preceding user request that explicitly names a
**historical, validation, retrospective, backtest, or lockbox** source.
The same guard applies if the stored `evidence_question` itself names
one of these incompatible sources. Such contradictory history fails closed
for ticker-switch, latest-record, aggregate-performance, and latest-record
explanation follow-ups. A verified forward-only sequence remains eligible;
every eligible turn still re-reads the integrity-checked journal.

This is a narrow provenance validation, not a new historical/forward tool,
model permission, or source conversion. No model, lockbox, append-only
journal, trading signal, execution, or Orion integration code is changed.

Offline regression:
`python -m unittest tests.test_verified_forward_cross_source_provenance -v`.
All history is synthetic; **no local Qwen benchmark was run**.


### Bounded tickerless provenance (26 October 2026)

A source tag and a canonical per-symbol `evidence_question` are not
sufficient to justify inheriting that symbol if the immediately preceding
user turn did not name it. For example, an overall `Ako sa ti darí v shadow
režime?` question or unrelated `Ahoj` cannot legitimately acquire NVDA
merely because the assistant history claims a verified NVDA answer.

When the previous user turn does not name a symbol, per-symbol forward
inheritance now requires one of the already supported, narrowly phrased
Slovak follow-ups (latest prediction, aggregate accuracy, or a bounded
explanation) **consistent with the recorded evidence kind**. Explicit
supported-ticker switches and overall forward reports remain available.
Every accepted follow-up re-reads the integrity-checked journal; no
numeric content is inherited from chat history. This guard does not
establish a complete cryptographic chain of provenance across truncated
chat histories, and does not authorize new tools or trading actions.

Regression tests: `python -m unittest discover -s tests -p
"test_verified_forward_tickerless_provenance.py" -v`. These are synthetic
unit tests, **not** a local Qwen benchmark.


### Named-turn forward provenance (26 October 2026)

A previous user turn that **names** NVDA (or another supported ticker) is
not by itself permission to inherit that ticker from a stored
`source=verified_forward_journal` response. In particular, stock price,
news, trading, greetings and conceptual questions such as
`Koľko stojí NVDA?` or `Čo znamená forward predikcia NVDA?` cannot
be converted into a verified forward conversation by a contradictory
stored `evidence_question`.

The immediate named user turn must itself have recognizable forward
prediction/record intent, or be one of the already supported short
single-ticker switches (`A čo NVDA?`, `A pre NVDA?`). Existing source,
ticker-count, historical-source and evidence-question checks still apply.
This is a conservative guard, not an open-ended semantic classifier;
unrecognized formulations stay outside automatic context inheritance.

Regression: `python -m unittest tests.test_verified_forward_named_turn_intent -v`.
The tests use synthetic history, not a local Qwen benchmark. No journal,
lockbox, predictive model, execution or Orion integration code changes.


## Named-turn metric provenance

The verified forward follow-up path now checks the **kind of metric requested
by the immediately preceding named-ticker user turn** against the recorded
`evidence_question`. A user asking for NVDA's latest prediction cannot
authorize a source-tagged answer whose stored evidence instead requests
aggregate NVDA MAE, and the reverse mismatch is also rejected. This applies
to both ticker-switch and same-ticker follow-ups, even when the ticker and
`verified_forward_journal` source tag agree.

Explicit, matching latest/performance requests remain eligible. The existing
bounded `A čo NVDA?` / `A pre NVDA?` switches remain eligible because they
request no metric themselves and must still have adjacent verified evidence.
Every accepted follow-up re-reads the same read-only verified forward journal;
it never inherits figures from assistant text. The change does not infer
metrics from free-form model output or alter historical readers.

Offline regression (synthetic provenance only, no Qwen benchmark):
`python -m unittest tests.test_verified_forward_named_metric_provenance -v`.


## Ambiguous mixed-metric provenance in named forward turns

A previous user request naming one supported stock can still be ambiguous
about **which verified metric** it requested. For example, `Aká je posledná
predikcia NVDA a aká je jej presnosť?` asks about both the latest record and
performance, but one recorded `evidence_question` may represent only one
reader result. A source tag and matching ticker cannot justify carrying
that partial evidence into a later ticker switch, accuracy follow-up or
explanation.

The immediate-turn provenance guard therefore declines automatic ticker
inheritance when a named user turn combines a latest-record cue
(`posledná`, `najnovšia`, `latest`) with a performance cue
(`presnosť`, `chyba`, `MAE`, `výsledky`). This intentionally
fails closed for some legitimate compound questions; it does not claim to
answer both requests, and it does not change the direct forward reader,
historical reports, or the model's general conversation.

Single-metric verified forward conversations remain eligible. Accepted
follow-ups still re-read the existing integrity-checked journal and never
reuse numeric assistant prose. Regression:
`python -m unittest tests.test_verified_forward_dual_metric_provenance -v`.
Tests use synthetic history; **no local Qwen benchmark was performed**.


## Source-bound colloquial Slovak error questions (26 October 2026)

The deterministic forward reader now recognizes a **small set of explicit
error-magnitude phrasings**: `Ako veľmi sa tvoje forward odhady mýlia?`,
`Ako ďaleko boli predikcie vo forward denníku od skutočnosti?`,
`O koľko sa tvoje forward predikcie líšili od reality?`, and
`Aká bola odchýlka forward predikcií NVDA?`. They select the **existing
aggregate magnitude MAE** reader, not the latest individual prediction,
directional accuracy or trading profitability.

A source cue (`forward`, `shadow`, `paper trading`, `denník`,
`journal`, or `záznam`) is mandatory for these newly recognized
expressions. Without one, or when the question explicitly asks about
historical/validation/backtest/lockbox results, **no forward reader is
selected**. Causal questions such as `Prečo sa tvoje forward predikcie
mýlia?` also remain outside this reader: MAE alone cannot establish
causation. Unrecognized phrasings are not claimed to be understood.

Only `ourai_language/forward_answers.py`, synthetic regressions and this
documentation change. No new reader, ticker, model weights, journal writes,
lockbox access, signals, execution or OurAI–Orion integration changes.

Regression:
`python -m unittest tests.test_forward_colloquial_error_source -v`.
Tests use a synthetic report and mocked journal reader. **No local Qwen
benchmark was performed.**

## Explicit forward directional-accuracy questions (26 October 2026)

The v3 forward journal reports predictions of **absolute 15-minute price
movement magnitude**, not UP/DOWN direction. Requests such as
`Aká bola smerová presnosť forward predikcií NVDA?`,
`Ako často si trafil smer predikcií vo forward denníku?`, and
`Koľkokrát si správne predpovedal rast/pokles NVDA v shadow režime?`
must **not** be answered with magnitude MAE as if it were a directional
hit rate. The deterministic forward path now reads the existing
integrity-checked journal and gives a fixed Slovak explanation that
directional accuracy is unavailable from this evidence. It does not
invent a hit-rate percentage, run a trading strategy, or claim that a
magnitude prediction predicts direction.

This bounded route requires both a **directional hit-rate cue** and an
**explicit forward/shadow/journal source**. Questions without a source,
historical/lockbox comparisons, and educational definitions do not
automatically select the forward reader. Ordinary magnitude MAE requests
remain unchanged. An invalid journal still fails closed rather than
returning unverified numerical evidence. This is not a general-purpose
natural-language understanding guarantee or a new analytic tool.

Regression:
`python -m unittest tests.test_forward_directional_accuracy_boundary -v`.
Tests use synthetic data and a mocked reader; **no local Qwen benchmark
was performed**.


## Ambiguous forward hit-rate wording

Natural Slovak questions such as `Ako často si sa trafil vo forward denníku?`,
`Koľkokrát boli tvoje predikcie správne v shadow režime?` and
`Aká je úspešnosť forward predikcií NVDA?` do **not** specify what
"correct" means. They now receive a bounded Slovak clarification asking
whether the user means **directional hit rate** or **absolute-magnitude
prediction error (MAE)**. No hit-rate percentage or threshold is inferred,
and no MAE value is misrepresented as a success rate.

This deterministic path requires an explicit forward/shadow/journal cue,
reads only the existing integrity-checked forward report, and fails closed
for corrupt data. Historical/lockbox wording, unanchored correctness
questions and educational definitions do not select the forward reader.
Explicit directional questions retain the separate "direction unavailable"
boundary; explicit MAE requests retain the magnitude-performance route.
No new tools, symbols, predictive models or trading behavior are introduced.

Synthetic-only regression (no local Qwen benchmark):
`python -m unittest tests.test_forward_ambiguous_accuracy_clarification -v`.

## Replies to a verified forward accuracy clarification

When a forward/shadow question asks how often predictions were "correct"
without defining correctness, the deterministic reader asks whether the user
means **directional hit rate** or **absolute-magnitude MAE**. A directly
following `Myslím MAE` or `Zaujíma ma smer` can now resolve that
clarification without guessing a metric or calling Qwen.

The reply is eligible **only** after the immediately preceding source-tagged
`verified_forward_journal` turn actually contained the deterministic
ambiguity clarification. Both the original user question and recorded
`evidence_question` must be matching forward ambiguity requests, with
identical supported ticker scope (or both overall). An absent/empty journal,
historical/lockbox answer, unsupported or multiple symbols, mismatched source,
intervening turn, or mixed/negated reply does not authorize inheritance.

The resolution constructs a new bounded question for the **existing
integrity-checked forward reader**. MAE returns observed absolute-magnitude
error; choosing direction returns the existing explicit
`direction_unavailable` explanation because v3 predicts magnitude, not
UP/DOWN. It does not invent directional hit rates, copy prior numeric values,
or touch the journal, model weights, trading/execution, or Orion integration.
These are deliberately narrow reply patterns, not general pronoun resolution.

Offline regression (synthetic evidence, no local Qwen benchmark):
`python -m unittest tests.test_forward_accuracy_clarification_replies -v`.


## Indirect Slovak replies to forward accuracy clarifications (26 October 2026)

A **direct** reply to the existing, verified ambiguity question can now use
a few natural Slovak paraphrases instead of naming `MAE` or `smer`.
For example, `Zaujíma ma skôr to, o koľko si sa pomýlil` selects
**aggregate absolute-magnitude MAE**, while
`Chcem vedieť, či si správne odhadol rast alebo pokles` selects the
existing **direction-unavailable** explanation. Commas, accents and
insignificant whitespace do not change the selection.

This is a **bounded allowlist**, not a general semantic parser. It is active
only after the immediately preceding verified forward answer actually asked
the accuracy clarification, with matching original/evidence source and
ticker scope. Negations, mixed metric requests, ticker switches, historical
references, stale context and unsupported phrasings do not inherit a tool.
The runtime constructs a canonical question and re-reads the **existing**
integrity-checked forward reader; it never copies prior numeric values.
No new reader, ticker, model, lockbox access, journal write, trading
operation or Orion integration behavior is introduced.

Regression:
`python -m unittest tests.test_forward_indirect_accuracy_clarification_replies -v`.
Tests use mocked synthetic evidence; **no local Qwen benchmark was run**.
