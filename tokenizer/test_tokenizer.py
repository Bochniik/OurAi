from pathlib import Path

from tokenizers import Tokenizer


ROOT = Path(__file__).resolve().parent.parent

TOKENIZER_FILE = (
    ROOT
    / "tokenizer"
    / "tokenizer.json"
)


tokenizer = Tokenizer.from_file(
    str(TOKENIZER_FILE)
)


tests = [
    "Artificial intelligence learns from data.",
    "Neurónová sieť sa učí z dát.",
    "Pravdepodobnosť rastu akcie je 72.5 %.",
    "NVDA price increased above the VWAP.",
    "def calculate_probability(x): return x * 0.5",
]


print("=" * 70)
print("OUR AI TOKENIZER TEST")
print("=" * 70)

print(
    "Vocabulary size:",
    tokenizer.get_vocab_size()
)

for text in tests:

    encoded = tokenizer.encode(text)

    print("\nTEXT:")
    print(text)

    print("\nTOKENS:")
    print(encoded.tokens)

    print("\nTOKEN IDS:")
    print(encoded.ids)

    decoded = tokenizer.decode(
        encoded.ids
    )

    print("\nDECODED:")
    print(decoded)

    print("-" * 70)