"""Builders for prediction tables used across the eval tests."""

import pandas as pd

from src.eval.constants import ARRAY_MIC, DYSARTHRIC_SPEAKERS

ALL_DYSARTHRIC = DYSARTHRIC_SPEAKERS  # F01 F03 F04 M01 M02 M03 M04 M05


def pred_row(speaker, utt, label, pred, mic=ARRAY_MIC, session="Session1"):
    return {"speaker_id": speaker, "session": session, "utterance_id": utt,
            "mic": mic, "label": label, "pred": pred}


def speaker_rows(speaker, n, n_correct, mic=ARRAY_MIC, label="yes", wrong="no",
                 start=0):
    """n clips of `label` for one speaker, the first n_correct predicted right."""
    return [
        pred_row(speaker, f"{start + i:04d}", label, label if i < n_correct else wrong, mic)
        for i in range(n)
    ]


def dysarthric_preds(correct_by_speaker, n=10, mic=ARRAY_MIC):
    """All 8 dysarthric speakers with n clips each; correct counts from the dict (default n)."""
    rows = []
    for spk in ALL_DYSARTHRIC:
        rows += speaker_rows(spk, n, correct_by_speaker.get(spk, n), mic)
    return pd.DataFrame(rows)


def loso_folds(speakers=ALL_DYSARTHRIC):
    """Leak-free leave-one-speaker-out folds: each trains on the other speakers."""
    return {s: frozenset(set(speakers) - {s}) for s in speakers}
