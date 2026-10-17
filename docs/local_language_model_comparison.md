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

## Troubleshooting

- `ollama` not recognized: reopen PowerShell after installing Ollama.
- Ollama server unavailable: start Ollama for Windows. If necessary, run
  `ollama serve` in a separate terminal *only if Ollama is not already running*.
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
