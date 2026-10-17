"""Minimal deterministic social replies for a small multilingual fallback model.

Only exact, clearly social phrases are covered. Other general conversation
continues through the locally hosted language model.
"""
from __future__ import annotations

import re
import unicodedata


def _normalize(text: str) -> str:
    folded = unicodedata.normalize("NFKD", str(text).casefold().strip())
    ascii_text = "".join(character for character in folded if not unicodedata.combining(character))
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", ascii_text)).strip()


def deterministic_social_answer(user_message: str) -> str | None:
    text = _normalize(user_message)
    if text in {"ako sa mas", "ako sa ti dari", "ako sa dnes mas"}:
        return "Som pripravený pomôcť. Čo by si chcel vedieť?"
    if text in {"ahoj", "cau", "dobry den", "nazdar"}:
        return "Ahoj! S čím ti môžem pomôcť?"
    if text in {"dakujem", "diky", "vdaka"}:
        return "Rado sa stalo. S čím ti ešte môžem pomôcť?"
    if text in {"how are you", "how are you doing"}:
        return "I'm ready to help. What would you like to know?"
    if text in {"hi", "hello", "hey"}:
        return "Hello! How can I help?"
    if text in {"thank you", "thanks"}:
        return "You're welcome. How else can I help?"
    return None
