# OurAI Financial Knowledge v1 — controlled educational answers

OurAI now contains a small **curated Slovak finance concept library** in
`ourai_language/financial_knowledge.py`. The purpose is to prevent repeated,
known factual errors from small local language models (e.g. equating
*overfitting* with model validation, or subtracting trading losses
incorrectly). This is **not a fine-tune, model training, financial advice, or
proof of a trading edge**.

## What is supported

Conservative direct-definition matching currently covers:

- fundamental vs technical analysis;
- stock liquidity and bid-ask spread;
- low trading volume;
- probability calibration;
- classification accuracy vs actual profitability;
- overfitting / generalization;
- trend vs volatility;
- Brier score;
- slippage and realistic execution costs.

Answers are short static explanations with human-readable topic IDs and
reference URLs. They never query prices, access account information, write
predictions or invent live Orion analyses. The curated references are intended
for independent review; they do **not** signify that any external institution
certifies OurAI.

### Review references

- scikit-learn: overfitting and validation — https://scikit-learn.org/stable/modules/cross_validation.html
- scikit-learn: calibration and Brier loss — https://scikit-learn.org/stable/modules/calibration.html
- Investor.gov: bid, ask and spread — https://www.investor.gov/introduction-investing/investing-basics/glossary/ask-price

This finite library should be separately maintained and reviewed when
definitions or reference links change. No statistical factual-accuracy claim
is made for arbitrary open-ended questions.

## Behavior and safety boundary

Explicit Slovak definition/explanation questions matching supported concepts
return curated text **without invoking Ollama**, usually near-instantly. The
interactive chat prints:

```text
[source: curated financial knowledge; not a live forecast]
```

when the response used the library. It remains possible to ask an open-ended
question about any other subject; those still go to the language model.
If a specific ticker from a loaded canonical analysis snapshot is named,
the new glossary does not displace snapshot-specific decision explanations
or fill in nonexistent current market facts.

The repository's **raw LLM benchmark intentionally disables** this knowledge
path for both backends, preserving comparability with prior model-only runs.
Rows record `answer_source` and reports identify
`knowledge_routing=disabled_raw_llm`; do not present these as a benchmark of
the full application.

## Windows PowerShell examples

Update the repository, run the offline deterministic tests, and chat:

```powershell
git pull --ff-only
python -m unittest tests.test_financial_knowledge -v
python -m scripts.chat_ourai --backend ollama --model qwen3:4b-instruct-2507-q4_K_M --num-ctx 1536 --max-new-tokens 240 --concise
```

Then ask:

```text
Čo je to preučenie modelu pri predpovedaní pohybu akcií?
Čo znamená kalibrácia pravdepodobností predikčného modelu?
Prečo model so 70 % úspešnosťou nemusí byť ziskový?
Aký je rozdiel medzi volatilitou a trendom?
```

To check what Qwen itself would say without the library, use the same chat
command with `--raw-llm` appended. The canonical snapshot grounding guard
and deterministic social replies are separate and remain enabled.

The fixed 8-question language benchmark remains model-only:

```powershell
python -m scripts.benchmark_ourai_language --backend ollama --model qwen3:4b-instruct-2507-q4_K_M --num-ctx 1536 --max-new-tokens 240 --timeout-seconds 600 --concise --output runs\language\qwen4b_raw_model_after_knowledge_v1.json
```

The small glossary does not guarantee correctness of *other* generated
answers. The model should still abstain from unsupported stock-specific
facts; the v3 magnitude predictor currently forecasts the absolute size
of 15-minute changes, **not their direction**.
