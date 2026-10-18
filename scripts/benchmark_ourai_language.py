"""Reproducible small manual-review benchmark for local OurAI language backends.

The report records raw answers/latencies; no fabricated automated correctness
scores. Evaluate Slovak fluency, financial factuality, completeness and honesty.
Test backends one at a time to avoid exceeding low-VRAM local systems.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter
from typing import Any

from ourai_language.ollama_runtime import DEFAULT_OLLAMA_MODEL, LocalOllamaChatRuntime
from ourai_language.runtime import DEFAULT_MODEL_ID, OurAIChatRuntime


CASES: tuple[tuple[str, str], ...] = (
    ("finance_fundamental_technical",
     "Vysvetli mi rozdiel medzi fundamentálnou a technickou analýzou."),
    ("finance_liquidity",
     "Čo znamená likvidita akcie a ako súvisí s bid-ask spreadom?"),
    ("finance_volume",
     "Prečo môže byť nízky objem problém pri day tradingu?"),
    ("finance_calibration",
     "Čo znamená kalibrácia pravdepodobností predikčného modelu?"),
    ("finance_accuracy_profit",
     "Prečo model so 70 % úspešnosťou nemusí byť ziskový?"),
    ("finance_overfitting",
     "Čo je to preučenie modelu pri predpovedaní pohybu akcií?"),
    ("finance_volatility",
     "Aký je rozdiel medzi volatilitou a trendom?"),
    ("forecast_humility",
     "Dokážeš garantovať, že NVDA zajtra porastie?"),
)


def evaluate(runtime: Any, cases: tuple[tuple[str, str], ...] = CASES) -> list[dict]:
    results: list[dict] = []
    for index, (case_id, prompt) in enumerate(cases, start=1):
        begin = perf_counter()
        error = None
        try:
            response = runtime.respond(prompt, history=[])
            answer = response.text
            answer_source = getattr(response, "answer_source", "model")
        except RuntimeError as exc:
            error = str(exc)
            answer = ""
            answer_source = None
        duration = round(perf_counter() - begin, 3)
        model_metrics = getattr(runtime, "last_response_metrics", None)
        row = {
            "case_id": case_id,
            "prompt": prompt,
            "response": answer,
            "answer_source": answer_source,
            "latency_seconds": duration,
            "error": error,
            "ollama_metrics": dict(model_metrics) if model_metrics is not None else None,
            "truncated": "[Odpoveď bola skrátená limitom dĺžky.]" in answer,
        }
        results.append(row)
        print(f"\n[{index}/{len(cases)}] {case_id} ({duration:.1f} s)")
        print("Q:", prompt)
        print("A:", answer if answer else f"<ERROR: {error}>")
        if error:
            raise RuntimeError(f"benchmark stopped at {case_id}: {error}")
    return results


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run fixed Slovak language QA benchmark; manual evaluation required."
    )
    parser.add_argument("--backend", choices=("ollama", "transformers"), default="ollama")
    parser.add_argument("--model")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--timeout-seconds", type=int, default=180, help="Local Ollama HTTP wait limit")
    parser.add_argument("--num-ctx", type=int, default=3072, help="Local Ollama context window (try 1536 vs 3072)")
    parser.add_argument("--concise", action="store_true", help="Ask Ollama for brief direct answers")
    parser.add_argument(
        "--output", default="runs/language/ollama_qwen3_1_7b_benchmark_v1.json",
    )
    args = parser.parse_args()
    model = args.model or (DEFAULT_OLLAMA_MODEL if args.backend == "ollama" else DEFAULT_MODEL_ID)
    print(f"OurAI language benchmark v1 | {args.backend} | {model}")
    if args.backend == "ollama":
        print(f"Ollama num_ctx={args.num_ctx} | concise={args.concise} | num_predict={args.max_new_tokens}")
    if args.backend == "ollama":
        runtime = LocalOllamaChatRuntime(
            model_id=model, max_new_tokens=args.max_new_tokens,
            timeout_seconds=args.timeout_seconds,
            num_ctx=args.num_ctx, concise=args.concise,
            knowledge_enabled=False,
        )
    else:
        runtime = OurAIChatRuntime(
            model_id=model, device=args.device, max_new_tokens=args.max_new_tokens,
            knowledge_enabled=False,
        )
    rows = evaluate(runtime)
    report = {
        "schema_version": 1,
        "benchmark": "ourai_language_finance_slovak_manual_v1",
        "backend": args.backend,
        "model": model,
        "max_new_tokens": args.max_new_tokens,
        "num_ctx": args.num_ctx if args.backend == "ollama" else None,
        "concise": args.concise if args.backend == "ollama" else None,
        "cases": rows,
        "quality_grades": None,
        "knowledge_routing": "disabled_raw_llm",
        "mean_wall_latency_seconds": round(sum(row["latency_seconds"] for row in rows) / len(rows), 3),
        "mean_wall_latency_excluding_first_seconds": round(sum(row["latency_seconds"] for row in rows[1:]) / (len(rows) - 1), 3),
        "truncated_responses": sum(row["truncated"] for row in rows),
        "grading_note": (
            "Raw qualitative test only. Review factuality, Slovak fluency, "
            "completeness and unsupported claims manually; latency is hardware-dependent."
        ),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("\nReport:", output)
    print("Truncated answers:", report["truncated_responses"], "/", len(rows))
    print("IMPORTANT: No automated factual accuracy score is implied.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
