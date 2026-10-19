# OurAI v3 historical performance chat (read-only)

The conversational layer can now answer retrospective performance questions from
**local** v3 validation and test-lockbox **report JSON** files. This does not
load or change the frozen prediction model, access test examples, request an
Alpaca connection, or write a forward journal.

## Windows / PowerShell

After pulling the branch (or merging the PR and pulling main), run from the
OurAi repository root:

```powershell
python -m unittest tests.test_historical_performance_chat -v

python -m scripts.chat_ourai `
  --backend ollama `
  --model qwen3:4b-instruct-2507-q4_K_M `
  --num-ctx 1536 `
  --max-new-tokens 240 `
  --concise `
  --historical-reports-dir runs\first-real-15m-v3 `
  --forward-journal runs\first-real-15m-v3\forward_shadow_v2\events.jsonl
```

The `--historical-reports-dir` flag is opt-in. It looks for these **local**
files, which may be excluded from Git:

- `nonlinear_magnitude_validation_v1.json`
- `nonlinear_magnitude_test_lockbox_v1.json`

The `--forward-journal` flag stays optional and independent. No forward
file is created by the chat tool. The fixed v2 forward start configured in
`trading/forward_analytics.py` is 2026-10-19 13:30 UTC.

Try these Slovak questions:

- `Ako dopadla historická validácia v3?`
- `Akú MAE dosiahol lockbox test?`
- `Porovnaj model v3 s lineárnym baseline.`
- `Porovnaj svoje výsledky s lineárnym modelom.` (validation and lockbox MAE with absolute and relative differences)
- `Porovnaj výsledky lockbox testu s nulovým baseline.`
- `Pri ktorých akciách robíš najväčšie chyby?`
- `Ako presné boli tvoje posledné predikcie?`

The last question deliberately consults the **forward journal**, not the
historical report. If the journal is absent, the answer says it is absent.
If per-symbol retrospective metrics are not present, OurAI does not invent a
stock ranking. A missing historical report is reported explicitly; invalid
report fields, incompatible provenance, and invalid numbers block generated
answers rather than asking the LLM to guess.

The historical answers are deterministic in Python for both Transformers and
Ollama, so no local model generation or API request is needed for recognized
questions. The `--raw-llm` flag disables these evidence-based paths for
language-model benchmarking.

## Meaning and limits

MAE is the mean absolute error for **15-minute absolute return magnitude**,
reported in basis points (1 bps = 0.01 percentage points). Lower is better.
The linear and null baselines are comparisons on the same historical split.
The validation and held-out test gate outcomes represent historical criteria;
they are **not** current market forecasts, UP/DOWN accuracy, profitability,
calibration, or independent future-forward acceptance.

This feature reads report summaries only. To rank individual stocks by
historical prediction error, separate per-symbol diagnostics must be generated
under an explicit evaluation protocol. No such ranking is inferred from daily
or aggregate MAE.

## Local testing

```powershell
python -m unittest tests.test_historical_performance_chat tests.test_forward_analytics_chat tests.test_ollama_language -v
```

The repo's CI also runs Python compilation and the full unittest suite on PRs.

## Context-aware follow-ups

The local terminal chat now tags the origin of each assistant answer and preserves
a canonical evidence question. When the immediately preceding assistant answer
came from verified local historical reports, short follow-ups can inherit its
test split and baseline *without consulting the language model*. For example:

```text
You > Porovnaj svoje výsledky s lineárnym modelom.
You > A na lockboxe?
You > A oproti nulovému?
You > O koľko je to lepšie?
You > A prečo?
You > Vysvetli to jednoducho.
```

The "why" answer explicitly distinguishes measured performance from an
unproven causal explanation. A standalone general question (e.g. about Python)
still reaches local Qwen. New or latest forward questions are always routed
independently; historical numbers must never be passed off as live metrics.

Context is deliberately limited to the **immediately preceding, evidence-tagged**
turn, with a canonical question retained for consecutive follow-ups. A social
or unrelated turn breaks this chain. This is not unlimited conversation memory:
open-ended model replies use the regular bounded local chat history, and some
analytical phrasings may still need a full question.

Run the regression tests without downloading LLM weights:

```powershell
python -m unittest discover -s tests -p "test_contextual_grounded_dialogue.py" -v
```
