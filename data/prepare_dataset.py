from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Iterator


@dataclass(frozen=True)
class Document:
    doc_id: str
    source: str
    text: str
    language: str | None = None
    license: str | None = None


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    return text.strip()


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def split_for_hash(digest: str, seed: int = 42) -> str:
    value = hashlib.sha256(f"{seed}:{digest}".encode("utf-8")).digest()
    bucket = int.from_bytes(value[:8], "big") / 2**64
    if bucket < 0.90:
        return "train"
    if bucket < 0.95:
        return "validation"
    return "test"


def iter_txt(path: Path) -> Iterator[Document]:
    text = path.read_text(encoding="utf-8")
    yield Document(doc_id=path.stem, source=path.name, text=text)


def iter_jsonl(path: Path) -> Iterator[Document]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or not isinstance(row.get("text"), str):
                raise ValueError(f"{path}:{line_number}: each row must contain string field 'text'")
            yield Document(
                doc_id=str(row.get("id") or f"{path.stem}:{line_number}"),
                source=str(row.get("source") or path.name),
                text=row["text"],
                language=row.get("language"),
                license=row.get("license"),
            )


def iter_documents(raw_dir: Path) -> Iterable[Document]:
    for path in sorted(raw_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() == ".txt":
            yield from iter_txt(path)
        elif path.suffix.lower() == ".jsonl":
            yield from iter_jsonl(path)


def prepare_dataset(raw_dir: Path, output_dir: Path, min_chars: int = 200, seed: int = 42) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    split_paths = {
        "train": output_dir / "train.jsonl",
        "validation": output_dir / "validation.jsonl",
        "test": output_dir / "test.jsonl",
    }

    handles = {name: path.open("w", encoding="utf-8") for name, path in split_paths.items()}
    seen_hashes: set[str] = set()
    stats = {
        "input_documents": 0,
        "kept_documents": 0,
        "duplicates_removed": 0,
        "too_short_removed": 0,
        "characters": 0,
        "splits": {name: {"documents": 0, "characters": 0} for name in split_paths},
        "seed": seed,
        "min_chars": min_chars,
    }

    try:
        for document in iter_documents(raw_dir):
            stats["input_documents"] += 1
            text = normalize_text(document.text)
            if len(text) < min_chars:
                stats["too_short_removed"] += 1
                continue

            digest = content_hash(text)
            if digest in seen_hashes:
                stats["duplicates_removed"] += 1
                continue
            seen_hashes.add(digest)

            split = split_for_hash(digest, seed=seed)
            payload = asdict(document)
            payload["text"] = text
            payload["content_hash"] = digest
            handles[split].write(json.dumps(payload, ensure_ascii=False) + "\n")

            chars = len(text)
            stats["kept_documents"] += 1
            stats["characters"] += chars
            stats["splits"][split]["documents"] += 1
            stats["splits"][split]["characters"] += chars
    finally:
        for handle in handles.values():
            handle.close()

    (output_dir / "stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare OurAI Dataset v0.2")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/v0.2"))
    parser.add_argument("--min-chars", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.raw_dir.exists():
        raise SystemExit(f"Raw data directory does not exist: {args.raw_dir}")
    stats = prepare_dataset(args.raw_dir, args.output_dir, args.min_chars, args.seed)
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
