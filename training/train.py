from pathlib import Path
import time
import random

import torch
from tokenizers import Tokenizer

from ai.config import ModelConfig
from ai.model import OurAI


# ============================================================
# CONFIG
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

TOKENIZER_FILE = ROOT / "tokenizer" / "tokenizer.json"
CORPUS_DIR = ROOT / "data" / "corpus"
CHECKPOINT_DIR = ROOT / "checkpoints"

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42

CONTEXT_LENGTH = 256
BATCH_SIZE = 8
GRADIENT_ACCUMULATION_STEPS = 4

MAX_STEPS = 3000
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.1

EVAL_INTERVAL = 200
EVAL_STEPS = 30
SAVE_INTERVAL = 500

TRAIN_SPLIT = 0.95


# ============================================================
# SETUP
# ============================================================

random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = "cuda" if torch.cuda.is_available() else "cpu"

print("=" * 70)
print("OUR AI v0.1 — FIRST REAL TRAINING")
print("=" * 70)

print("Device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

print()


# ============================================================
# TOKENIZER
# ============================================================

print("Loading tokenizer...")

tokenizer = Tokenizer.from_file(
    str(TOKENIZER_FILE)
)

vocab_size = tokenizer.get_vocab_size()

print("Vocabulary size:", vocab_size)


# ============================================================
# DATA
# ============================================================

print("\nLoading corpus...")

corpus_files = sorted(
    CORPUS_DIR.glob("*.txt")
)

if not corpus_files:
    raise RuntimeError(
        f"No corpus files found in {CORPUS_DIR}"
    )

texts = []

for file in corpus_files:
    print(" -", file.name)

    text = file.read_text(
        encoding="utf-8"
    )

    texts.append(text)

full_text = "\n".join(texts)

print(
    f"\nCorpus characters: {len(full_text):,}"
)

print("Tokenizing corpus...")

encoded = tokenizer.encode(full_text)

all_tokens = torch.tensor(
    encoded.ids,
    dtype=torch.long,
)

print(
    f"Total tokens: {len(all_tokens):,}"
)

split_index = int(
    len(all_tokens) * TRAIN_SPLIT
)

train_tokens = all_tokens[:split_index]
val_tokens = all_tokens[split_index:]

print(
    f"Training tokens:   {len(train_tokens):,}"
)

print(
    f"Validation tokens: {len(val_tokens):,}"
)

if len(val_tokens) <= CONTEXT_LENGTH + 1:
    raise RuntimeError(
        "Validation dataset is too small."
    )


# ============================================================
# BATCH GENERATOR
# ============================================================

def get_batch(data):
    max_start = (
        len(data)
        - CONTEXT_LENGTH
        - 1
    )

    starts = torch.randint(
        0,
        max_start,
        (BATCH_SIZE,),
    )

    x = torch.stack([
        data[
            i:
            i + CONTEXT_LENGTH
        ]
        for i in starts
    ])

    y = torch.stack([
        data[
            i + 1:
            i + CONTEXT_LENGTH + 1
        ]
        for i in starts
    ])

    return (
        x.to(device),
        y.to(device),
    )


# ============================================================
# MODEL
# ============================================================

config = ModelConfig(
    vocab_size=vocab_size,
    context_length=512,
    n_layers=6,
    n_heads=5,
    d_model=320,
    dropout=0.1,
)

model = OurAI(config).to(device)

num_params = sum(
    p.numel()
    for p in model.parameters()
)

print(
    f"\nModel parameters: {num_params:,}"
)

print(
    f"Training context: {CONTEXT_LENGTH}"
)

print(
    f"Batch size: {BATCH_SIZE}"
)

print(
    "Gradient accumulation:",
    GRADIENT_ACCUMULATION_STEPS
)

effective_batch = (
    BATCH_SIZE
    * GRADIENT_ACCUMULATION_STEPS
)

print(
    f"Effective batch size: "
    f"{effective_batch}"
)


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
)

use_amp = device == "cuda"

scaler = torch.amp.GradScaler(
    "cuda",
    enabled=use_amp,
)


# ============================================================
# EVALUATION
# ============================================================

@torch.no_grad()
def estimate_loss():
    model.eval()

    losses = []

    for _ in range(EVAL_STEPS):
        x, y = get_batch(
            val_tokens
        )

        with torch.autocast(
            device_type=device,
            dtype=torch.float16,
            enabled=use_amp,
        ):
            _, loss = model(
                x,
                y,
            )

        losses.append(
            loss.item()
        )

    model.train()

    return (
        sum(losses)
        / len(losses)
    )


# ============================================================
# SAVE
# ============================================================

def save_checkpoint(
    filename,
    step,
    val_loss=None,
):
    path = CHECKPOINT_DIR / filename

    torch.save(
        {
            "step": step,
            "model_state_dict":
                model.state_dict(),
            "optimizer_state_dict":
                optimizer.state_dict(),
            "config":
                config.__dict__,
            "vocab_size":
                vocab_size,
            "val_loss":
                val_loss,
        },
        path,
    )

    print(
        f"\nCheckpoint saved: {path}"
    )


# ============================================================
# TRAINING
# ============================================================

print("\nStarting training...\n")

model.train()

best_val_loss = float("inf")

start_time = time.time()

for step in range(1, MAX_STEPS + 1):

    optimizer.zero_grad(
        set_to_none=True
    )

    total_loss = 0.0

    for _ in range(
        GRADIENT_ACCUMULATION_STEPS
    ):
        x, y = get_batch(
            train_tokens
        )

        with torch.autocast(
            device_type=device,
            dtype=torch.float16,
            enabled=use_amp,
        ):
            _, loss = model(
                x,
                y,
            )

            loss = (
                loss
                / GRADIENT_ACCUMULATION_STEPS
            )

        scaler.scale(
            loss
        ).backward()

        total_loss += (
            loss.item()
        )

    scaler.unscale_(
        optimizer
    )

    torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=1.0,
    )

    scaler.step(
        optimizer
    )

    scaler.update()

    if (
        step == 1
        or step % 20 == 0
    ):
        elapsed = (
            time.time()
            - start_time
        )

        steps_per_second = (
            step / elapsed
        )

        print(
            f"step {step:5d} | "
            f"train loss "
            f"{total_loss:.4f} | "
            f"{steps_per_second:.2f} "
            f"steps/s"
        )

    if (
        step % EVAL_INTERVAL == 0
    ):
        val_loss = estimate_loss()

        print(
            f"\n>>> step {step} | "
            f"validation loss: "
            f"{val_loss:.4f}"
        )

        if (
            val_loss
            < best_val_loss
        ):
            best_val_loss = (
                val_loss
            )

            save_checkpoint(
                "ai_v0.1_best.pt",
                step,
                val_loss,
            )

    if (
        step % SAVE_INTERVAL == 0
    ):
        save_checkpoint(
            f"ai_v0.1_step_"
            f"{step}.pt",
            step,
        )


# ============================================================
# FINISH
# ============================================================

final_val_loss = estimate_loss()

save_checkpoint(
    "ai_v0.1_final.pt",
    MAX_STEPS,
    final_val_loss,
)

total_time = (
    time.time()
    - start_time
)

print("\n" + "=" * 70)

print(
    "TRAINING COMPLETE"
)

print(
    f"Final validation loss: "
    f"{final_val_loss:.4f}"
)

print(
    f"Best validation loss: "
    f"{best_val_loss:.4f}"
)

print(
    f"Training time: "
    f"{total_time / 60:.1f} minutes"
)

if torch.cuda.is_available():
    peak_memory = (
        torch.cuda.max_memory_allocated()
        / 1024 ** 2
    )

    print(
        f"Peak GPU memory: "
        f"{peak_memory:.1f} MB"
    )

print("=" * 70)