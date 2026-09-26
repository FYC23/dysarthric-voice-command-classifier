"""
Metrics for one run on one mic.

The headline is speaker-averaged accuracy over the 8 dysarthric speakers:
each speaker counts once, however many clips they recorded, because the
question is how well the model works for a new user. Pooled clip accuracy
is reported alongside but is dominated by speakers with many clips.
"""

from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

import pandas as pd

from src.eval.constants import (
    COMMANDS, CONTROL_SPEAKERS, DYSARTHRIC_SPEAKERS, NON_COMMAND_PREDS, OOV, REJECT,
    SEVERITY, SPEECH_COMMANDS_V2_WORDS,
)
from src.eval.schema import EvalValidationError, Run

IN_PRETRAINING = tuple(w for w in COMMANDS if w in SPEECH_COMMANDS_V2_WORDS)
NOT_IN_PRETRAINING = tuple(w for w in COMMANDS if w not in SPEECH_COMMANDS_V2_WORDS)


@dataclass(frozen=True)
class RunMetrics:
    mic: str
    headline: float                 # speaker-averaged, dysarthric
    per_speaker: pd.DataFrame       # index speaker; columns n, correct, accuracy
    severity: Mapping[str, float]   # group -> mean of its speakers' accuracy
    pooled: float                   # every dysarthric clip counts once
    per_word: pd.DataFrame          # index word; columns n, accuracy (dysarthric clips)
    confusion: pd.DataFrame         # true word x predicted (commands + reject + oov)
    pretraining_overlap: Mapping[str, float]  # pooled acc, words in / not in Speech Commands
    oov_rate: float
    reject_rate: float
    control_headline: Optional[float]  # zero-shot runs only


def compute_run_metrics(run: Run, mic: str) -> RunMetrics:
    on_mic = run.predictions[run.predictions["mic"] == mic]
    dys = on_mic[on_mic["speaker_id"].isin(DYSARTHRIC_SPEAKERS)]
    per_speaker = per_speaker_accuracy(dys, DYSARTHRIC_SPEAKERS)
    empty = per_speaker.index[per_speaker["n"] == 0].tolist()
    if empty:
        raise EvalValidationError(f"dysarthric speakers with no clips on {mic}: {empty}")

    return RunMetrics(
        mic=mic,
        headline=float(per_speaker["accuracy"].mean()),
        per_speaker=per_speaker,
        severity={g: float(per_speaker.loc[list(s), "accuracy"].mean()) for g, s in SEVERITY.items()},
        pooled=_accuracy(dys),
        per_word=per_word_accuracy(dys),
        confusion=confusion_matrix(dys),
        pretraining_overlap={
            "in_speech_commands": _accuracy(dys[dys["label"].isin(IN_PRETRAINING)]),
            "not_in_speech_commands": _accuracy(dys[dys["label"].isin(NOT_IN_PRETRAINING)]),
        },
        oov_rate=float((dys["pred"] == OOV).mean()),
        reject_rate=float((dys["pred"] == REJECT).mean()),
        control_headline=_control_headline(on_mic) if run.zero_shot else None,
    )


def per_speaker_accuracy(preds: pd.DataFrame, speakers: Sequence[str]) -> pd.DataFrame:
    """One row per speaker in `speakers` (in that order); n = 0 and NaN accuracy if absent."""
    correct = preds["pred"] == preds["label"]
    table = pd.DataFrame({
        "n": preds.groupby("speaker_id").size(),
        "correct": correct.groupby(preds["speaker_id"]).sum(),
    }).reindex(list(speakers)).fillna(0).astype(int)
    return table.assign(accuracy=table["correct"] / table["n"].where(table["n"] > 0))


def per_word_accuracy(preds: pd.DataFrame) -> pd.DataFrame:
    """Pooled accuracy per true word, for words that have clips, in vocabulary order."""
    correct = (preds["pred"] == preds["label"]).groupby(preds["label"])
    table = pd.DataFrame({"n": correct.size(), "accuracy": correct.mean()})
    return table.reindex([w for w in COMMANDS if w in table.index])


def confusion_matrix(preds: pd.DataFrame) -> pd.DataFrame:
    """Counts of true word (rows, all commands) by prediction (commands, reject, oov)."""
    counts = pd.crosstab(preds["label"], preds["pred"])
    return counts.reindex(index=list(COMMANDS),
                          columns=list(COMMANDS) + list(NON_COMMAND_PREDS),
                          fill_value=0)


def _accuracy(preds: pd.DataFrame) -> float:
    return float((preds["pred"] == preds["label"]).mean()) if len(preds) else float("nan")


def _control_headline(on_mic: pd.DataFrame) -> Optional[float]:
    controls = on_mic[on_mic["speaker_id"].isin(CONTROL_SPEAKERS)]
    if controls.empty:
        return None
    present = sorted(set(controls["speaker_id"]))
    return float(per_speaker_accuracy(controls, present)["accuracy"].mean())
