# OurAI

OurAI is an experimental decoder-only Transformer language model built from scratch in PyTorch. The project currently contains a custom model architecture, a Byte-Level BPE tokenizer pipeline, a bootstrap corpus generator, a CUDA training loop, checkpointing, and interactive text generation.

## Current status: v0.1

The first end-to-end milestone is complete:

- custom decoder-only Transformer
- causal self-attention using PyTorch scaled dot-product attention
- learned token and positional embeddings
- tied input/output token weights
- Byte-Level BPE tokenizer
- mixed-precision CUDA training
- gradient accumulation and gradient clipping
- validation-loss tracking and checkpoint saving
- interactive autoregressive generation

The current bootstrap corpus is intentionally small and synthetic. It is useful for proving that the training pipeline works, but it is **not** suitable for judging real language understanding or reasoning. The next milestone is Dataset/Training v0.2 with document-level train/validation/test splits, deduplication, real text sources, and reproducible evaluation.

## Model v0.1

Default architecture:

- decoder-only Transformer
- 6 Transformer blocks
- `d_model = 320`
- 5 attention heads
- context capacity: 512 tokens
- dropout: 0.1
- vocabulary learned by the tokenizer (bootstrap run produced 3,671 tokens)

The exact parameter count depends on tokenizer vocabulary size. The first bootstrap training run used approximately 8.7M trainable parameters.

## Repository layout

```text
OurAi/
├── ai/
│   ├── config.py
│   └── model.py
├── data/
│   └── build_bootstrap_corpus.py
├── tokenizer/
│   ├── train_tokenizer.py
│   ├── test_tokenizer.py
│   └── tokenizer.json
├── training/
│   └── train.py
├── generate.py
├── test_model.py
├── requirements.txt
└── README.md
```

Generated corpora and model checkpoints should not be committed in future changes.

## Setup

Create a virtual environment:

```powershell
python -m venv .venv
```

Install the lightweight dependencies:

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Install PyTorch separately so the build matches your hardware. The first validated CUDA setup used PyTorch 2.14 with CUDA 12.6:

```powershell
.\.venv\Scripts\python.exe -m pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cu126
```

Verify CUDA:

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

## Bootstrap run

Build the small synthetic corpus:

```powershell
.\.venv\Scripts\python.exe data\build_bootstrap_corpus.py
```

Train the tokenizer:

```powershell
.\.venv\Scripts\python.exe tokenizer\train_tokenizer.py
```

Test tokenizer round-tripping:

```powershell
.\.venv\Scripts\python.exe tokenizer\test_tokenizer.py
```

Test the Transformer itself:

```powershell
.\.venv\Scripts\python.exe test_model.py
```

Train the model:

```powershell
.\.venv\Scripts\python.exe -m training.train
```

Generate text after a checkpoint exists:

```powershell
.\.venv\Scripts\python.exe generate.py
```

## Development direction

The next priorities are:

1. replace the synthetic corpus with a reproducible real-text dataset pipeline
2. split by document before tokenization to prevent train/validation leakage
3. deduplicate documents and report dataset statistics
4. add held-out test evaluation and perplexity
5. add resumable training and learning-rate scheduling
6. later add instruction/chat formatting and assistant fine-tuning
7. only after data/evaluation quality is established, scale the model size

## Important note

A low loss on the current bootstrap corpus mainly demonstrates that the model can learn repeated patterns in that corpus. It does not by itself demonstrate broad language understanding, reasoning, or factual reliability.
