"""
Turn an open-vocabulary ASR transcript into one of the 20 commands.

strict: the transcript, normalized, is exactly a command; else "oov".
lenient: a forced choice among the commands, like the classifiers, so the
    comparison is fair: exact command words, then homophones, then the
    nearest command by edit distance. Only an empty transcript is "oov".

Both depend only on the transcript, so rescoring never re-runs a model.
"""

import re
from typing import Sequence

from src.eval.constants import COMMANDS, OOV

DIGIT_WORDS = dict(zip("0123456789", COMMANDS[:10]))  # "0" -> "zero", ... "9" -> "nine"

# Same sound, different spelling. "oh" is deliberately absent: it is a
# different word from "zero", not a mis-spelling of it.
HOMOPHONES = {
    "to": "two", "too": "two",
    "for": "four", "fore": "four",
    "won": "one",
    "ate": "eight",
    "know": "no",
    "write": "right", "rite": "right",
}

_DIGIT = re.compile(r"[0-9]")
_NOT_WORD = re.compile(r"[^\w\s]|_")


def normalize(text: str) -> str:
    """Lowercase, one word per ASCII digit, punctuation to spaces, single spaces."""
    text = _DIGIT.sub(lambda m: f" {DIGIT_WORDS[m.group()]} ", text.lower())
    return " ".join(_NOT_WORD.sub(" ", text).split())


def strict_pred(transcript: str) -> str:
    text = normalize(transcript)
    return text if text in COMMANDS else OOV


def lenient_pred(transcript: str) -> str:
    raw = normalize(transcript).split()
    if not raw:
        return OOV
    joined = "".join(raw)
    if joined in COMMANDS:
        return joined
    # A word said as itself beats a homophone: in "go to the left", "to" is a
    # function word, not "two"
    tokens = [HOMOPHONES.get(t, t) for t in raw]
    for candidates in (raw, tokens):
        for token in candidates:
            if token in COMMANDS:
                return token
    return _nearest(tokens + [joined])


def _nearest(candidates: Sequence[str]) -> str:
    """Command closest to any candidate; min() keeps the first, i.e. vocabulary order."""
    return min(COMMANDS, key=lambda c: min(levenshtein(c, s) for s in candidates))


def levenshtein(a: str, b: str) -> int:
    """Insertions, deletions and substitutions to turn a into b."""
    previous = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        current = [i]
        for j, y in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (x != y)))
        previous = current
    return previous[-1]


SCORERS = {"strict": strict_pred, "lenient": lenient_pred}
