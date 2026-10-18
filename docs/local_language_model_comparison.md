# OurAI: local conversational-model comparison

OurAI's reliable snapshot explainer and guardrails are separate from the
open-ended language model. The original Hugging Face Qwen3-0.6B path remains
available. A second backend speaks to **local Ollama on 127.0.0.1:11434 only**,
not Ollama Cloud or any metered inference API. Do not use model tags ending
in `:cloud`.

## Windows installation and first run

1. Install Ollama for Windows from https://ollama.com/download/windows .
   Start Ollama if the installation did not start its local server.
2. In a new PowerShell window, run:

   ```powershell
   ollama --version
   ollama pull qwen3:1.7b
   ollama list
   ```

   The Ollama `qwen3:1.7b` package uses Q4_K_M quantization (about 1.4 GB of
   downloaded model data). Actual VRAM use **also includes KV cache and
   runtime overhead**, so close other GPU-heavy jobs during language testing.
   On the 4-GB GTX 1650, start with 1.7B before trying larger models.

3. In `C:\Users\nikop\Desktop\OurAI`, with `.venv` activated:

   ```powershell
   git pull --ff-only
   python -m scripts.chat_ourai --backend ollama --model qwen3:1.7b --analysis-json examples\analysis_snapshot.example.json
   ```

   The static NVDA example is NOT a real-time forecast. Known snapshot Q&A
   is intentionally answered deterministically in both backends; it does not
   indicate the larger language model's capabilities.

4. Run the **open-ended Slovak finance benchmark**:

   ```powershell
   python -m scripts.benchmark_ourai_language --backend ollama --model qwen3:1.7b --output runs\language\ollama_qwen3_1_7b_benchmark_v1.json
   ```

   To compare with the previously installed FP16 0.6B model, run separately
   (not simultaneously):

   ```powershell
   python -m scripts.benchmark_ourai_language --backend transformers --model Qwen/Qwen3-0.6B --device cuda --output runs\language\hf_qwen3_0_6b_benchmark_v1.json
   ```

   Both models receive the same general-chat system instructions and prompts.
   The comparison records raw responses and per-question wall time only.
   **Manually grade** factual accuracy, fluent Slovak, completeness and
   appropriately stated uncertainty; keyword matching alone is not enough.
   This small fixed benchmark is a diagnostic, NOT a formal external
   language-model benchmark or evidence of trading profitability.

## GTX 1650: controlled context / latency experiment

The local Ollama adapter previously hardcoded `num_ctx=3072`. It now exposes
`--num-ctx` in both chat and benchmark; **the default remains 3072** for
backwards compatibility. It also has an optional `--concise` response style
(aims for 3–5 short sentences; not a factuality guarantee).

On a 4 GB GTX 1650, `ollama ps` previously showed
`32%/68% CPU/GPU` for `qwen3:4b-instruct-2507-q4_K_M`. A smaller context
window *may* release enough KV/cache memory for greater GPU offload, but GPU
offload depends on available VRAM and Ollama's runtime; **no improvement is
guaranteed**. Too small a context can truncate the conversation or snapshot.

Run **sequentially** (never simultaneously), with the exact same model,
question set, generated-token limit and optional concise flag:

```powershell
git pull --ff-only
ollama stop qwen3:4b-instruct-2507-q4_K_M
python -m scripts.benchmark_ourai_language --backend ollama --model qwen3:4b-instruct-2507-q4_K_M --num-ctx 3072 --max-new-tokens 240 --timeout-seconds 600 --concise --output runs\language\qwen4b_ctx3072_concise.json
ollama ps
ollama stop qwen3:4b-instruct-2507-q4_K_M
python -m scripts.benchmark_ourai_language --backend ollama --model qwen3:4b-instruct-2507-q4_K_M --num-ctx 1536 --max-new-tokens 240 --timeout-seconds 600 --concise --output runs\language\qwen4b_ctx1536_concise.json
ollama ps
```

The reports contain total wall time per question, a mean including and
excluding the first question, number of truncated answers and (when returned
by Ollama) `load_duration`, `prompt_eval_count`, `eval_count`,
`eval_duration`, `total_duration` and `done_reason`. Ollama's durations
are **nanoseconds**. Check `ollama ps` *while the model is loaded* for
CPU/GPU split; the breakdown after a model unload cannot be inferred from
benchmark JSON. The first call can be dominated by model loading.

**Caution:** concise mode is a new prompt setting. Do not compare its raw
latency directly with an old benchmark using the longer default prompt.
For assessing only the impact of `--num-ctx`, ensure all other flags match
exactly. Manually review factual accuracy and Slovak fluency. Reduced context
does not solve knowledge errors (e.g., confusing overfitting with validation).

A short interactive check with the smaller context:

```powershell
python -m scripts.chat_ourai --backend ollama --model qwen3:4b-instruct-2507-q4_K_M --num-ctx 1536 --max-new-tokens 240 --timeout-seconds 600 --concise
```

## Troubleshooting

- `ollama` not recognized: reopen PowerShell after installing Ollama.
- Verify local server quickly: `Invoke-RestMethod http://127.0.0.1:11434/api/tags -TimeoutSec 10`. If it fails to connect, start Ollama for Windows. Only use `ollama serve` when no Ollama server is already running.
- If the first question runs for **180 seconds** before a timeout, that suggests a slow model load/generation rather than proof the server is absent. Check `ollama ps` for CPU/GPU execution; if Ollama runs on CPU, responses may be slow. Try `--max-new-tokens 128 --timeout-seconds 600` on the benchmark/chat command. Check server reachability separately via `/api/tags`.
- Model responses may require longer on the first run; give the warm-up time, but do not interpret a timed-out run as a valid benchmark.
- Model missing: `ollama pull qwen3:1.7b`.
- Model spills into CPU RAM or is slow: use `ollama ps`; close other GPU tasks.
- If 1.7B remains too weak, try `ollama pull qwen3:4b-instruct` (about
  2.5 GB model data) as a **separate later benchmark**, not as a proven
  improvement. On a 4GB GPU, KV-cache/overhead can force partial CPU offload.
- The original `python -m scripts.chat_ourai --device cuda ...` command
  still works and uses the Hugging Face backend by default.

The API adapter enforces the same `ANALYSIS_QA / GENERAL_CHAT` routing,
canonical snapshot fact rules and unsupported-percentage fallback used by
the old backend. Full factual grounding outside snapshot percentages still
requires future work; never treat open-ended generated claims as verified
market information.
