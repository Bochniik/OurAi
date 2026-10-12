from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder


VOCAB_SIZE = 8000

SPECIAL_TOKENS = [
    "<pad>",
    "<bos>",
    "<eos>",
    "<unk>",
]


ROOT = Path(__file__).resolve().parent.parent
CORPUS_DIR = ROOT / "data" / "corpus"
OUTPUT_FILE = ROOT / "tokenizer" / "tokenizer.json"


def find_corpus_files():
    files = sorted(CORPUS_DIR.glob("*.txt"))

    if not files:
        raise RuntimeError(
            f"No .txt corpus files found in: {CORPUS_DIR}"
        )

    return [str(file) for file in files]


def train_tokenizer():
    files = find_corpus_files()

    print("=" * 60)
    print("OUR AI TOKENIZER")
    print("=" * 60)

    print(f"Corpus files: {len(files)}")

    for file in files:
        print(" -", file)

    tokenizer = Tokenizer(
        BPE(
            unk_token="<unk>"
        )
    )

    tokenizer.pre_tokenizer = ByteLevel(
        add_prefix_space=False
    )

    trainer = BpeTrainer(
        vocab_size=VOCAB_SIZE,
        min_frequency=2,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=ByteLevel.alphabet(),
        show_progress=True,
    )

    print("\nTraining tokenizer...")

    tokenizer.train(
        files=files,
        trainer=trainer,
    )

    tokenizer.decoder = ByteLevelDecoder()

    tokenizer.save(str(OUTPUT_FILE))

    print("\nTokenizer trained successfully.")
    print("Vocabulary size:", tokenizer.get_vocab_size())
    print("Saved to:", OUTPUT_FILE)


if __name__ == "__main__":
    train_tokenizer()