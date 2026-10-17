# OurAI conversational language layer

OurAI's language layer is deliberately separate from the trading forecast engine.
The forecast engine owns probabilities, expected move, uncertainty and decision
state. The language model explains those values and answers follow-up questions.

## Base model

The first production-oriented local runtime uses `Qwen/Qwen3-0.6B` rather than
continuing to scale the historical scratch-trained `ai/model.py` prototype. The old
prototype remains in the repository for lineage/research, but it was trained on a
small local corpus and is not the reliability target for OrionTrader.

The language runtime is local: there is no paid per-request language API. On the
first run Hugging Face model files are downloaded into the normal local cache; later
runs reuse that cache.

## Grounding contract

Orion passes forecast-specific facts through `AnalysisSnapshot`. Important fields are:

- ticker and forecast horizon
- `p_up` / `p_down`
- expected move percentage
- confidence and uncertainty
- `LONG_BIAS`, `SHORT_BIAS`, `NO_TRADE`, or `UNAVAILABLE`
- market/news/context labels and reasons when supplied by upstream engines

The system prompt tells the language model that this canonical snapshot is the only
source of forecast-specific facts. A deterministic grounding guard additionally
checks generated percentage claims. If the model introduces a percentage that is not
present in the canonical snapshot, the generated prose is discarded and a safe
snapshot-only fallback is returned.

This first guard is intentionally narrow. Later milestones should ground prices,
news, timestamps and market-context claims against their own structured sources.

## Local setup

Install the CUDA-enabled PyTorch build appropriate for the machine first. The normal
OurAI development environment already does this separately because CUDA wheels are
machine-specific. Then install the optional language dependencies:

```powershell
python -m pip install -r requirements-language.txt
```

Run free-form local chat:

```powershell
python -m scripts.chat_ourai --device cuda
```

Run grounded analysis chat with a canonical JSON snapshot:

```powershell
python -m scripts.chat_ourai --device cuda --analysis-json path\to\analysis.json
```

The terminal supports `/clear` and `/exit`.

## Why not train a language model from scratch?

The existing OurAI prototype is a useful educational lineage artifact, but reliable
conversation requires broad language knowledge before Orion-specific alignment. The
shortest path is therefore:

1. start from a capable open-weight multilingual instruct model;
2. enforce a strict structured interface to the forecast engine;
3. evaluate local inference and grounding;
4. build an Orion/finance supervised fine-tuning dataset;
5. train a small adapter/LoRA only after the baseline is measured.

This also lets the forecast team and language team evolve in parallel without
coupling model internals.
