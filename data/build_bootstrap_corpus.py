from pathlib import Path
import random


ROOT = Path(__file__).resolve().parent.parent
CORPUS_DIR = ROOT / "data" / "corpus"

random.seed(42)

CORPUS_DIR.mkdir(parents=True, exist_ok=True)


english_paragraphs = [
    """
Artificial intelligence is the study and development of systems that can
process information, recognize patterns, learn from data, and produce useful
outputs. Machine learning is one branch of artificial intelligence.
""",
    """
A neural network contains many numerical parameters. During training, those
parameters are adjusted in order to reduce an error function. A model that
learns useful patterns can later make predictions on previously unseen data.
""",
    """
Science uses observation, measurement, experiments, and mathematical models
to describe the natural world. A good scientific explanation should make
predictions that can be tested.
""",
    """
Computers represent information using numbers. Text, images, audio, and other
forms of data can all be converted into numerical representations that
algorithms can process.
""",
    """
Financial markets connect buyers and sellers of financial assets. Prices
change as market participants react to information, liquidity, expectations,
risk, and supply and demand.
""",
]

slovak_paragraphs = [
    """
Umelá inteligencia je oblasť informatiky, ktorá skúma systémy schopné
spracúvať informácie, rozpoznávať vzory, učiť sa z dát a vytvárať užitočné
výstupy.
""",
    """
Neurónová sieť obsahuje veľké množstvo číselných parametrov. Počas tréningu
sa tieto parametre postupne upravujú tak, aby model robil menšiu chybu.
""",
    """
Počítač nerozumie textu rovnakým spôsobom ako človek. Text preto musíme
previesť na číselnú reprezentáciu, ktorú môže neurónová sieť spracovať.
""",
    """
Finančný trh je prostredie, v ktorom účastníci nakupujú a predávajú aktíva.
Cena sa mení v závislosti od ponuky, dopytu, likvidity, informácií a
očakávaní investorov.
""",
]

czech_paragraphs = [
    """
Umělá inteligence využívá matematické modely pro zpracování dat, hledání
vzorů a vytváření předpovědí.
""",
    """
Neuronová síť se během trénování učí tím, že upravuje své parametry podle
chyby mezi předpovědí a správným výsledkem.
""",
]

finance_terms = [
    "price",
    "volume",
    "volatility",
    "liquidity",
    "momentum",
    "trend",
    "market",
    "stock",
    "equity",
    "return",
    "risk",
    "probability",
    "prediction",
    "signal",
    "portfolio",
    "NASDAQ",
    "NYSE",
    "VWAP",
    "EMA",
    "SMA",
    "RSI",
]

subjects = [
    "the model",
    "the neural network",
    "the algorithm",
    "the market",
    "the system",
    "the investor",
    "the training process",
]

verbs = [
    "analyzes",
    "estimates",
    "measures",
    "learns from",
    "processes",
    "compares",
    "predicts",
]

objects = [
    "market data",
    "historical observations",
    "price movements",
    "new information",
    "numerical features",
    "probabilities",
    "patterns in data",
]

slovak_subjects = [
    "model",
    "neurónová sieť",
    "algoritmus",
    "systém",
    "trh",
]

slovak_verbs = [
    "analyzuje",
    "spracúva",
    "porovnáva",
    "odhaduje",
    "predikuje",
    "vyhodnocuje",
]

slovak_objects = [
    "trhové dáta",
    "historické údaje",
    "cenový pohyb",
    "pravdepodobnosť",
    "nové informácie",
    "číselné vstupy",
]


def generate_english_sentences(count=12000):
    lines = []

    for _ in range(count):
        subject = random.choice(subjects)
        verb = random.choice(verbs)
        obj = random.choice(objects)

        lines.append(
            f"{subject.capitalize()} {verb} {obj}."
        )

    return lines


def generate_slovak_sentences(count=6000):
    lines = []

    for _ in range(count):
        subject = random.choice(slovak_subjects)
        verb = random.choice(slovak_verbs)
        obj = random.choice(slovak_objects)

        lines.append(
            f"{subject.capitalize()} {verb} {obj}."
        )

    return lines


def generate_finance_sentences(count=6000):
    lines = []

    templates = [
        "The {a} can influence the {b}.",
        "A change in {a} may affect {b}.",
        "The model uses {a} together with {b}.",
        "Market participants observe {a} and {b}.",
        "The relationship between {a} and {b} can change over time.",
        "Higher {a} does not always imply higher {b}.",
    ]

    for _ in range(count):
        a, b = random.sample(finance_terms, 2)
        template = random.choice(templates)

        lines.append(template.format(a=a, b=b))

    return lines


def generate_code_examples(count=2500):
    examples = []

    names = [
        "price",
        "volume",
        "probability",
        "prediction",
        "loss",
        "signal",
        "value",
        "result",
    ]

    for i in range(count):
        name = random.choice(names)

        examples.append(
            f"""
def calculate_{name}_{i}(x, y):
    result = x + y
    return result

{name}_{i} = calculate_{name}_{i}(10, 5)
print({name}_{i})
"""
        )

    return examples


def save_file(name, parts):
    path = CORPUS_DIR / name

    text = "\n".join(parts)

    path.write_text(
        text,
        encoding="utf-8"
    )

    size_mb = path.stat().st_size / (1024 ** 2)

    print(f"{name}: {size_mb:.2f} MB")


def main():
    print("=" * 60)
    print("BUILDING OUR AI BOOTSTRAP CORPUS")
    print("=" * 60)

    save_file(
        "general_english.txt",
        english_paragraphs * 250
        + generate_english_sentences()
    )

    save_file(
        "slovak_czech.txt",
        slovak_paragraphs * 250
        + czech_paragraphs * 150
        + generate_slovak_sentences()
    )

    save_file(
        "finance.txt",
        generate_finance_sentences()
    )

    save_file(
        "code.txt",
        generate_code_examples()
    )

    print("\nCorpus created successfully.")
    print("Location:", CORPUS_DIR)


if __name__ == "__main__":
    main()