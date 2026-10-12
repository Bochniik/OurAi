from pathlib import Path

import torch
from tokenizers import Tokenizer

from ai.config import ModelConfig
from ai.model import OurAI


ROOT = Path(__file__).resolve().parent

TOKENIZER_FILE = ROOT / "tokenizer" / "tokenizer.json"
CHECKPOINT_FILE = ROOT / "checkpoints" / "ai_v0.1_best.pt"

MAX_NEW_TOKENS = 100
TEMPERATURE = 0.8
TOP_K = 40


device = "cuda" if torch.cuda.is_available() else "cpu"


print("=" * 70)
print("OUR AI v0.1 — GENERATION")
print("=" * 70)

print("Device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# ============================================================
# TOKENIZER
# ============================================================

tokenizer = Tokenizer.from_file(
    str(TOKENIZER_FILE)
)

print(
    "Vocabulary size:",
    tokenizer.get_vocab_size()
)


# ============================================================
# LOAD CHECKPOINT
# ============================================================

print("\nLoading checkpoint...")

checkpoint = torch.load(
    CHECKPOINT_FILE,
    map_location=device,
    weights_only=True,
)

config = ModelConfig(
    **checkpoint["config"]
)

model = OurAI(config).to(device)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

print(
    "Checkpoint step:",
    checkpoint["step"]
)

print(
    "Validation loss:",
    checkpoint.get("val_loss")
)


# ============================================================
# GENERATION
# ============================================================

@torch.no_grad()
def generate(
    prompt,
    max_new_tokens=100,
    temperature=0.8,
    top_k=40,
):

    encoded = tokenizer.encode(prompt)

    input_ids = torch.tensor(
        [encoded.ids],
        dtype=torch.long,
        device=device,
    )

    for _ in range(max_new_tokens):

        # Model má maximálny kontext 512 tokenov.
        context = input_ids[
            :,
            -config.context_length:
        ]

        logits, _ = model(context)

        # Potrebujeme iba predikciu
        # poslednej pozície.
        logits = logits[:, -1, :]

        logits = logits / temperature

        if top_k is not None:
            values, _ = torch.topk(
                logits,
                min(
                    top_k,
                    logits.size(-1)
                ),
            )

            cutoff = values[:, [-1]]

            logits = torch.where(
                logits < cutoff,
                torch.full_like(
                    logits,
                    float("-inf")
                ),
                logits,
            )

        probabilities = torch.softmax(
            logits,
            dim=-1,
        )

        next_token = torch.multinomial(
            probabilities,
            num_samples=1,
        )

        input_ids = torch.cat(
            [
                input_ids,
                next_token
            ],
            dim=1,
        )

    generated_ids = (
        input_ids[0]
        .detach()
        .cpu()
        .tolist()
    )

    return tokenizer.decode(
        generated_ids
    )


# ============================================================
# INTERACTIVE MODE
# ============================================================

print("\n" + "=" * 70)

print(
    "Our AI is ready."
)

print(
    "Write a beginning of a sentence."
)

print(
    "Type 'exit' to quit."
)

print("=" * 70)


while True:

    prompt = input(
        "\nYou > "
    ).strip()

    if prompt.lower() in {
        "exit",
        "quit"
    }:
        break

    if not prompt:
        continue

    result = generate(
        prompt,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        top_k=TOP_K,
    )

    print("\nOurAI >")
    print(result)