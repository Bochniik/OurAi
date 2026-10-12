# Dataset v0.2

OurAI's v0.2 data pipeline separates raw source material from reproducible processed splits.

## Local layout

```text
data/
├── raw/                 # local source documents, ignored by Git
├── processed/v0.2/      # generated train/validation/test files, ignored by Git
├── build_bootstrap_corpus.py
└── prepare_dataset.py
```

## Supported raw formats

### `.txt`

Each UTF-8 text file is treated as one document.

### `.jsonl`

Each non-empty line is a JSON object with a required `text` field. Optional metadata:

```json
{"id":"doc-1","source":"example","language":"sk","license":"CC0-1.0","text":"..."}
```

## Preparation

Run from the repository root:

```powershell
.\.venv\Scripts\python.exe data\prepare_dataset.py
```

Optional arguments:

```powershell
.\.venv\Scripts\python.exe data\prepare_dataset.py --raw-dir data\raw --output-dir data\processed\v0.2 --min-chars 200 --seed 42
```

The pipeline:

1. normalizes Unicode/newlines
2. filters very short documents
3. removes exact normalized duplicates using SHA-256
4. assigns each document deterministically to train (90%), validation (5%), or test (5%) **before tokenization**
5. writes `train.jsonl`, `validation.jsonl`, `test.jsonl`, and `stats.json`

The hash-based split is stable: adding new documents does not reshuffle existing documents between splits.

## Data licensing

Only use text that we are legally permitted to train on. Keep source/license metadata when available. Raw datasets are ignored by Git so large or restricted source material is not accidentally published with the codebase.

## Why document-level splitting matters

The v0.1 bootstrap trainer split a single concatenated token stream near the end. With repetitive synthetic text, highly similar material could therefore appear in both training and validation data. Dataset v0.2 splits whole deduplicated documents first, reducing this leakage and making validation metrics more meaningful.
