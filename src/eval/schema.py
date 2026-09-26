"""
The prediction format every model reports through, and the checks run on it.

A Run is one model at one seed: a table with one row per scored clip, plus
the training speakers of each leave-one-speaker-out fold. A Run validates on
construction, so any Run that exists is leak-free and countable.
"""

import hashlib
from dataclasses import dataclass, field
from typing import Iterable, Mapping

import pandas as pd

from src.eval.constants import (
    COMMANDS, CONTROL_SPEAKERS, DYSARTHRIC_SPEAKERS, MICS, NON_COMMAND_PREDS,
)

PRED_COLUMNS = ("speaker_id", "session", "utterance_id", "mic", "label", "pred")
# One recording is (speaker, session, utterance) on one mic
CLIP_KEY = ("speaker_id", "session", "utterance_id", "mic")
# Runs are comparable only if they scored the same clips against the same labels
FINGERPRINT_COLUMNS = CLIP_KEY + ("label",)
KNOWN_SPEAKERS = frozenset(DYSARTHRIC_SPEAKERS) | frozenset(CONTROL_SPEAKERS)
VALID_PREDS = frozenset(COMMANDS) | frozenset(NON_COMMAND_PREDS)


class EvalValidationError(ValueError):
    """A run that would produce a misleading number."""


@dataclass(frozen=True)
class Run:
    """
    One model at one seed.

    predictions: one row per clip with PRED_COLUMNS (extra columns are kept,
        e.g. per-class scores for later threshold tuning).
    fold_train_speakers: held-out speaker -> speakers its fold trained on.
        Required for trained models; must be empty for zero-shot models.
    zero_shot: the model never trained on TORGO (e.g. off-the-shelf ASR), so
        control speakers may be scored too.
    """
    model: str
    seed: int
    predictions: pd.DataFrame
    zero_shot: bool = False
    fold_train_speakers: Mapping[str, frozenset] = field(default_factory=dict)

    def __post_init__(self):
        # Per-fold frames concatenated without ignore_index repeat index labels,
        # which breaks pandas reindexing downstream; keep a fresh 0..n-1 index.
        object.__setattr__(self, "predictions", self.predictions.reset_index(drop=True))
        validate_run(self)


def validate_run(run: Run) -> None:
    """Raise EvalValidationError if the run could produce a misleading number."""
    preds = run.predictions
    _check_columns(preds)
    _check_values(preds)
    _check_no_duplicates(preds)
    _check_speakers(preds, run.zero_shot)
    _check_folds(preds, run)


def _check_columns(preds: pd.DataFrame) -> None:
    missing = [c for c in PRED_COLUMNS if c not in preds.columns]
    if missing:
        raise EvalValidationError(f"predictions are missing columns: {missing}")
    null_cols = [c for c in PRED_COLUMNS if preds[c].isna().any()]
    if null_cols:
        raise EvalValidationError(f"predictions have missing values in: {null_cols}")


def _check_values(preds: pd.DataFrame) -> None:
    for column, allowed in (("label", set(COMMANDS)), ("pred", VALID_PREDS), ("mic", set(MICS))):
        bad = sorted(set(preds[column]) - allowed)
        if bad:
            raise EvalValidationError(f"invalid {column} values: {bad}")


def _check_no_duplicates(preds: pd.DataFrame) -> None:
    dupes = preds[preds.duplicated(list(CLIP_KEY), keep=False)]
    if not dupes.empty:
        first = dupes.iloc[0][list(CLIP_KEY)].to_dict()
        raise EvalValidationError(
            f"{len(dupes)} rows score the same clip more than once, e.g. {first}")


def _check_speakers(preds: pd.DataFrame, zero_shot: bool) -> None:
    speakers = set(preds["speaker_id"])
    unknown = sorted(speakers - set(DYSARTHRIC_SPEAKERS) - set(CONTROL_SPEAKERS))
    if unknown:
        raise EvalValidationError(f"unknown speakers: {unknown}")
    missing = sorted(set(DYSARTHRIC_SPEAKERS) - speakers)
    if missing:
        raise EvalValidationError(f"dysarthric speakers with no predictions: {missing}")
    controls = sorted(speakers & set(CONTROL_SPEAKERS))
    if controls and not zero_shot:
        raise EvalValidationError(
            f"control speakers {controls} scored for a trained model; every fold "
            "trains on all controls in Phase A, so these scores would leak")


def _check_folds(preds: pd.DataFrame, run: Run) -> None:
    folds = run.fold_train_speakers
    if run.zero_shot:
        if folds:
            raise EvalValidationError("a zero-shot run cannot have training folds")
        return
    for speaker, train in folds.items():
        # A bare string "M01" would iterate as {"M", "0", "1"} and hide a leak
        if not isinstance(train, (set, frozenset)) or not train <= KNOWN_SPEAKERS:
            raise EvalValidationError(
                f"fold for {speaker} must be a set of known speaker ids, got {train!r}")
    for speaker in sorted(set(preds["speaker_id"])):
        if speaker not in folds:
            raise EvalValidationError(f"no training fold recorded for test speaker {speaker}")
        if speaker in folds[speaker]:
            raise EvalValidationError(
                f"held-out speaker {speaker} is in its own fold's training speakers")


def clip_fingerprint(preds: pd.DataFrame) -> str:
    """Hash of the clips scored and their labels (not the predictions), order-independent."""
    rows = preds[list(FINGERPRINT_COLUMNS)].itertuples(index=False)
    keys = sorted("|".join(map(str, row)) for row in rows)
    return hashlib.sha256("\n".join(keys).encode()).hexdigest()


def check_same_clips(runs: Iterable[Run]) -> None:
    """Raise unless every run scored exactly the same clips, so numbers are comparable."""
    runs = list(runs)
    prints = {clip_fingerprint(r.predictions) for r in runs}
    if len(prints) > 1:
        names = sorted({f"{r.model}/seed{r.seed}" for r in runs})
        raise EvalValidationError(
            f"runs were not scored on the same clips with the same labels: {names}")
