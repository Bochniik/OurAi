from __future__ import annotations

import argparse
import json
from pathlib import Path

from ourai_language.contracts import AnalysisSnapshot
from ourai_language.runtime import DEFAULT_MODEL_ID, OurAIChatRuntime
from ourai_language.ollama_runtime import DEFAULT_OLLAMA_MODEL, LocalOllamaChatRuntime


def _load_analysis(path: str | None) -> AnalysisSnapshot | None:
    if path is None:
        return None
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return AnalysisSnapshot.from_mapping(payload)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Run the local grounded conversational OurAI prototype."
    )
    result.add_argument("--backend", choices=("transformers", "ollama"), default="transformers")
    result.add_argument("--model", help="Hugging Face model ID or locally pulled Ollama model tag")
    result.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    result.add_argument("--max-new-tokens", type=int, default=384)
    result.add_argument(
        "--analysis-json",
        help="Optional canonical Orion analysis snapshot used to ground analysis answers.",
    )
    return result


def main() -> int:
    args = parser().parse_args()
    analysis = _load_analysis(args.analysis_json)

    print("=== OURAI LOCAL CONVERSATIONAL MODEL ===")
    model_id = args.model or (DEFAULT_OLLAMA_MODEL if args.backend == "ollama" else DEFAULT_MODEL_ID)
    print("Backend:", args.backend)
    print("Model:", model_id)
    print("Requested device:", args.device)
    print("Analysis grounding:", "ON" if analysis is not None else "OFF")
    if analysis is not None:
        print("Snapshot source:", args.analysis_json)
        print("Note: snapshot values are static inputs, not a verified live market forecast.")
    if args.backend == "ollama":
        print("Connecting to local Ollama at 127.0.0.1:11434 (no cloud API)...")
        runtime = LocalOllamaChatRuntime(
            model_id=model_id, max_new_tokens=args.max_new_tokens,
        )
    else:
        print("Loading model (the first run may download model weights)...")
        runtime = OurAIChatRuntime(
            model_id=model_id,
            device=args.device,
            max_new_tokens=args.max_new_tokens,
        )
    print("Runtime device:", runtime.device)
    print("Commands: /clear, /exit")

    history: list[dict[str, str]] = []
    while True:
        try:
            user_message = input("\nYou > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_message:
            continue
        if user_message.lower() in {"/exit", "exit", "quit"}:
            break
        if user_message.lower() == "/clear":
            history.clear()
            print("Conversation history cleared.")
            continue

        try:
            result = runtime.respond(
                user_message,
                analysis=analysis,
                history=history,
            )
        except RuntimeError as exc:
            print(f"Local model error: {exc}")
            continue
        print("\nOurAI >")
        print(result.text)
        if result.used_grounding_fallback:
            values = ", ".join(f"{value:g}%" for value in result.unsupported_percentages)
            print(f"[grounding guard replaced unsupported generated percentages: {values}]")

        history.extend(
            [
                {"role": "user", "content": user_message, "mode": result.mode},
                {"role": "assistant", "content": result.text, "mode": result.mode},
            ]
        )
        # Bound conversational history. Mode tags prevent an analysis answer from
        # contaminating later ordinary chat (and vice versa).
        history = history[-12:]

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
