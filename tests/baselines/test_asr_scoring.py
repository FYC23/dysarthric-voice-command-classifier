"""Mapping open-vocabulary ASR transcripts onto the 20 commands."""

import pytest

from src.baselines.asr.scoring import (
    HOMOPHONES, SCORERS, levenshtein, lenient_pred, normalize, strict_pred,
)
from src.eval.constants import COMMANDS, OOV


@pytest.mark.parametrize("text, expected", [
    ("Four.", "four"),
    ("  YES!! ", "yes"),
    ("4", "four"),
    ("10", "one zero"),
    ("for-ward", "for ward"),
    ("Sí.", "sí"),
    ("", ""),
    ("...", ""),
])
def test_normalize(text, expected):
    assert normalize(text) == expected


@pytest.mark.parametrize("a, b, d", [("", "abc", 3), ("six", "six", 0), ("sick", "six", 2),
                                     ("kitten", "sitting", 3)])
def test_levenshtein(a, b, d):
    assert levenshtein(a, b) == d == levenshtein(b, a)


@pytest.mark.parametrize("transcript, expected", [
    ("Four.", "four"),
    ("4", "four"),
    ("Forward!", "forward"),
    ("for ward", OOV),         # strict: not literally the word
    ("four four", OOV),
    ("for", OOV),              # strict applies no homophones
    ("", OOV),
    ("Thank you for watching.", OOV),
])
def test_strict(transcript, expected):
    assert strict_pred(transcript) == expected


@pytest.mark.parametrize("transcript, expected", [
    ("to", "two"), ("Too.", "two"), ("for", "four"), ("Won.", "one"), ("ate", "eight"),
    ("know", "no"), ("Write.", "right"), ("rite", "right"),
    ("for ward", "forward"),   # joined tokens beat the "for" -> four homophone
    ("go back", "back"),       # first command token in spoken order
    ("yes no", "yes"),
    ("tree", "three"),         # nearest by edit distance
    ("dawn", "down"),
    ("sick", "six"),           # tie six/back at distance 2 -> vocabulary order
    ("ape", "one"),            # tie one/up at 2 -> digits come first
])
def test_lenient(transcript, expected):
    assert lenient_pred(transcript) == expected


@pytest.mark.parametrize("transcript", ["", "   ", "?!", "..."])
def test_lenient_empty_is_oov(transcript):
    assert lenient_pred(transcript) == OOV


def test_oh_is_never_zero():
    # "oh" is a different word, not a spelling of zero (decided with the user)
    assert "oh" not in HOMOPHONES
    assert lenient_pred("Oh.") != "zero"
    assert strict_pred("Oh.") == OOV


def test_hallucinated_sentence_is_a_deterministic_command():
    pred = lenient_pred("Thank you for watching.")
    assert pred in COMMANDS
    assert pred == lenient_pred("Thank you for watching.")


def test_non_ascii_never_crashes():
    for text in ("Sí.", "Ça va", "二", "٣", "Ñandú"):
        assert strict_pred(text) in COMMANDS + (OOV,)
        assert lenient_pred(text) in COMMANDS + (OOV,)


def test_every_command_maps_to_itself_both_ways():
    for word in COMMANDS:
        assert strict_pred(word.capitalize() + ".") == word
        assert lenient_pred(word) == word


def test_scorers_registry():
    assert SCORERS == {"strict": strict_pred, "lenient": lenient_pred}
