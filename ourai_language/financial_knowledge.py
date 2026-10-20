"""Narrow, reviewed Slovak financial concept explanations for OurAI.

This is a *curated answer library*, NOT retrieval of live prices, personal
advice, a model capability gain, or validated forward trading performance.
A conservative matcher only handles explicit educational concepts; all other
requests remain with the LLM or canonical market-snapshot explainer.

References were inspected during authoring and are informational, not a
claim that a third party has certified OurAI's exact wording.
"""
from __future__ import annotations

from dataclasses import dataclass
import re
import unicodedata

from .contracts import AnalysisSnapshot


@dataclass(frozen=True)
class FinancialKnowledgeAnswer:
    topic: str
    text: str
    references: tuple[str, ...]


# Short direct answers by design, with no invented current market facts.
_LIBRARY: dict[str, FinancialKnowledgeAnswer] = {
    "simple_moving_average": FinancialKnowledgeAnswer(
        topic="simple_moving_average",
        text=(
            "SMA znamená Simple Moving Average (jednoduchý kĺzavý priemer). "
            "Pri SMA(20) ide o aritmetický priemer posledných 20 hodnôt "
            "zvolenej ceny, spravidla uzatváracej. Pri každej novej sviečke "
            "najstaršia cena vypadne a pridá sa nová. SMA pomáha "
            "vyhladzovať krátkodobé výkyvy, ale reaguje oneskorene "
            "a sama osebe nepredpovedá budúci smer ceny."
        ),
        references=("https://www.investopedia.com/terms/s/sma.asp",),
    ),
    "fundamental_technical": FinancialKnowledgeAnswer(
        topic="fundamental_technical",
        text=(
            "Fundamentálna analýza skúma podnik, jeho hospodárenie, zadlženie, "
            "konkurenciu a odhad vnútornej hodnoty akcie. Technická analýza "
            "pracuje najmä s historickými cenami, objemom a vzorcami na grafe. "
            "Ani jeden prístup sám osebe nezaručuje budúci vývoj ceny."
        ),
        references=("https://www.investor.gov/introduction-investing",),
    ),
    "liquidity_spread": FinancialKnowledgeAnswer(
        topic="liquidity_spread",
        text=(
            "Likvidita vyjadruje, ako ľahko možno akciu kúpiť alebo predať "
            "bez výrazného posunutia jej ceny. Bid je najvyššia ponúkaná "
            "nákupná cena, ask najnižšia ponúkaná predajná cena; ich rozdiel "
            "je bid-ask spread. Likvidnejšie akcie mávajú spravidla užší "
            "spread a nižšie náklady na okamžité vykonanie obchodu, nie je "
            "to však zaručené."
        ),
        references=("https://www.investor.gov/introduction-investing/investing-basics/glossary/ask-price",),
    ),
    "low_volume": FinancialKnowledgeAnswer(
        topic="low_volume",
        text=(
            "Pri nízkom objeme môže byť menej dostupných protistrán a "
            "menšia hĺbka objednávok. To môže znamenať širší spread, "
            "väčší slippage a prudšie cenové skoky pri jednotlivých obchodoch. "
            "Samotný nízky objem však nedokazuje smer budúceho pohybu."
        ),
        references=("https://www.investor.gov/introduction-investing/investing-basics/glossary/ask-price",),
    ),
    "calibration": FinancialKnowledgeAnswer(
        topic="calibration",
        text=(
            "Kalibrácia pravdepodobností znamená, že predikované percentá "
            "zodpovedajú dlhodobej početnosti udalostí v porovnateľných situáciách. "
            "Ak model opakovane uvádza približne 70 % pravdepodobnosť rastu, "
            "rast by mal nastať asi v 70 % takýchto prípadov. "
            "Kalibráciu treba overovať na dátach oddelených od trénovania; "
            "sama osebe nezaručuje ziskovosť."
        ),
        references=("https://scikit-learn.org/stable/modules/calibration.html",),
    ),
    "accuracy_profit": FinancialKnowledgeAnswer(
        topic="accuracy_profit",
        text=(
            "Sedemdesiatpercentná úspešnosť sama neurčuje ziskovosť. "
            "Rozhoduje aj priemerný zisk, priemerná strata, spread, "
            "slippage a poplatky. Pri 70 výhrach po 1 € a 30 stratách "
            "po 3 € je výsledok 70 − 90 = −20 € ešte pred nákladmi. "
            "Presný výsledok konkrétnej stratégie treba overiť oddelene."
        ),
        references=("https://www.investor.gov/introduction-investing",),
    ),
    "overfitting": FinancialKnowledgeAnswer(
        topic="overfitting",
        text=(
            "Preučenie (overfitting) nastáva, keď sa model príliš prispôsobí "
            "tréningovým dátam vrátane náhodného šumu. Na tréningových dátach "
            "môže dosahovať výborné výsledky, no na nových, dovtedy "
            "nevidených dátach sa výrazne zhorší. Pri akciách ho odhaľujeme "
            "chronologickou validáciou a nezávislým forward testovaním, "
            "nie opakovaným ladením podľa uzamknutého testu."
        ),
        references=("https://scikit-learn.org/stable/modules/cross_validation.html",),
    ),
    "volatility_trend": FinancialKnowledgeAnswer(
        topic="volatility_trend",
        text=(
            "Volatilita opisuje mieru kolísania výnosov alebo cien, "
            "nie smer pohybu. Trend opisuje prevažujúci smer vývoja "
            "ceny v určitom období – rastúci, klesajúci alebo bočný. "
            "Akcia môže zároveň silno kolísať a mať rastúci trend."
        ),
        references=("https://www.investor.gov/introduction-investing",),
    ),
    "brier_score": FinancialKnowledgeAnswer(
        topic="brier_score",
        text=(
            "Brier score meria chybu pravdepodobnostných predikcií. "
            "Pri dvojtriednej udalosti je to priemer štvorcov rozdielov "
            "medzi odhadnutou pravdepodobnosťou a skutočným výsledkom "
            "(0 alebo 1). Nižšia hodnota je lepšia; samotný Brier score "
            "však nepreukazuje ziskovosť stratégie ani dokonalú kalibráciu."
        ),
        references=("https://scikit-learn.org/stable/modules/calibration.html",),
    ),
    "slippage": FinancialKnowledgeAnswer(
        topic="slippage",
        text=(
            "Slippage je rozdiel medzi očakávanou a skutočne dosiahnutou "
            "cenou vykonaného obchodu. Môže sa zväčšovať pri vysokej "
            "volatilite, nízkej likvidite alebo veľkých príkazoch. "
            "Do hodnotenia stratégie preto patrí aj spread, slippage "
            "a prípadné poplatky."
        ),
        references=("https://www.investor.gov/introduction-investing/investing-basics",),
    ),
}


def _normalize(value: str) -> str:
    folded = unicodedata.normalize("NFKD", value.casefold())
    plain = "".join(c for c in folded if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s%]", " ", plain)).strip()


def lookup_financial_knowledge(
    user_message: str,
    *,
    analysis: AnalysisSnapshot | None = None,
) -> FinancialKnowledgeAnswer | None:
    """Conservative general-definition routing; never answer specific forecasts.

    Requiring an explicit explanatory question plus subject keywords means
    vague queries and unrelated discussions still use their usual path.
    """
    normalized = _normalize(str(user_message or ""))
    if not normalized or len(normalized) > 280:
        return None
    if not any(normalized.startswith(prefix) for prefix in (
        "co je ", "co znamena ", "vysvetli ", "vysvetli mi ",
        "aky je rozdiel ", "aka je ", "preco ", "ako funguje ",
        "ako suvisi ", "co su ",
    )):
        return None

    # Don't turn ticker-specific or user's actual forecast questions into a
    # generic finance glossary. This guard applies with or without a snapshot.
    if analysis is not None:
        if re.search(rf"(?<!\w){re.escape(_normalize(analysis.ticker))}(?!\w)", normalized):
            return None
        if analysis.company_name and _normalize(analysis.company_name) in normalized:
            return None
    if any(marker in normalized for marker in (
        "tejto predikci", "tento model ", "nas model ", "orion model",
        "aktualn", "dnes ", "zajtra ", "konkretn", "na moj", "mojho portfol",
        "predikciu nvda", "akciu nvda",
    )):
        return None

    # SMA can also mean spinal muscular atrophy. Do not hijack an
    # explicitly medical question with a finance definition.
    if re.search(r"(?<!\w)sma(?!\w)", normalized) and not any(
        marker in normalized for marker in (
            "spin", "atrofi", "ochoren", "medicin", "genet",
            "motorick", "neurolog", "liecb", "diagnoz",
        )
    ):
        return _LIBRARY["simple_moving_average"]
    if "fundamentaln" in normalized and "technick" in normalized and "analyz" in normalized:
        return _LIBRARY["fundamental_technical"]
    if "preucen" in normalized or "overfitt" in normalized or "overfit" in normalized:
        return _LIBRARY["overfitting"]
    if "kalibrac" in normalized and any(x in normalized for x in ("pravdepodob", "model", "predikc")):
        return _LIBRARY["calibration"]
    if "brier" in normalized:
        return _LIBRARY["brier_score"]
    if "slippage" in normalized or "sklz" in normalized:
        return _LIBRARY["slippage"]
    if "volatil" in normalized and "trend" in normalized:
        return _LIBRARY["volatility_trend"]
    if "uspesnost" in normalized and any(x in normalized for x in ("zisk", "profit", "strat")):
        return _LIBRARY["accuracy_profit"]
    if "likvidit" in normalized and any(x in normalized for x in ("akci", "spread", "bid", "trh")):
        return _LIBRARY["liquidity_spread"]
    if "objem" in normalized and any(x in normalized for x in ("nizk", "day trading", "daytrading", "obchod")):
        return _LIBRARY["low_volume"]
    return None


def supported_topics() -> tuple[str, ...]:
    return tuple(_LIBRARY)
