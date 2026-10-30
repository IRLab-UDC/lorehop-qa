"""Capitalised-token and capitalised-phrase extraction.

Shared by the analyses that need to know which surface forms a text introduces:
the adaptation generalisation test and the tokeniser fragmentation measurement.
"""

from __future__ import annotations

import re

TOKEN = re.compile(r"[A-Z][a-zA-Z'\-]{2,}")
STOP = {"The", "This", "These", "Those", "Their", "There", "They", "When", "Where", "While",
        "After", "Before", "During", "Because", "Although", "However", "Both", "Each", "Every",
        "Some", "Many", "Most", "Its", "It", "His", "Her", "One", "Two", "Three", "Four",
        "Five", "Six", "Seven", "Eight", "Nine", "Ten", "First", "Second", "Third", "In", "On",
        "At", "By", "For", "From", "With", "Within", "Under", "Over", "Between", "Among",
        "Following", "Later", "Since", "Until", "Unlike", "Despite", "Several", "Other"}


def nouns(text: str) -> tuple[set[str], set[str]]:
    """Capitalised tokens, and the maximal capitalised phrases they form.

    Both granularities are needed. A token like "Guild" is shared with real-world text because
    invented names embed ordinary English words, which says nothing about whether the name is
    real; the phrase "the Amber Registry Guild" is the unit the claim is about.
    """
    tokens: set[str] = set()
    phrases: set[str] = set()
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        run: list[str] = []
        for position, token in enumerate(sentence.split()):
            cleaned = token.strip(".,;:!?\"'()[]")
            capital = bool(TOKEN.fullmatch(cleaned)) and position > 0 and cleaned not in STOP
            if capital:
                tokens.add(cleaned)
                run.append(cleaned)
                continue
            if run:
                phrases.add(" ".join(run))
                run = []
        if run:
            phrases.add(" ".join(run))
    return tokens, phrases
