"""Tests for paired per-speaker comparisons between two models."""

import pandas as pd
import pytest

from src.eval.compare import compare
from src.eval.schema import EvalValidationError, Run
from tests.eval.factories import ALL_DYSARTHRIC, dysarthric_preds, loso_folds, speaker_rows


def seeds(model, correct):
    return [Run(model=model, seed=i, predictions=dysarthric_preds(correct),
                fold_train_speakers=loso_folds()) for i in range(3)]


BASELINE = {s: 5 for s in ALL_DYSARTHRIC}
# vs baseline: six speakers +0.2, M03 tied, M05 -0.1
CANDIDATE = {"F01": 7, "F03": 7, "F04": 7, "M01": 7, "M02": 7, "M03": 5, "M04": 7, "M05": 4}


def test_per_speaker_difference_is_candidate_minus_baseline():
    c = compare(seeds("base", BASELINE), seeds("cand", CANDIDATE))
    assert c.per_speaker_diff["F01"] == pytest.approx(0.2)
    assert c.per_speaker_diff["M05"] == pytest.approx(-0.1)


def test_counts_speakers_better_worse_and_tied():
    c = compare(seeds("base", BASELINE), seeds("cand", CANDIDATE))
    assert (c.n_better, c.n_worse, c.n_tied) == (6, 1, 1)


def test_mean_difference_and_its_interval():
    c = compare(seeds("base", BASELINE), seeds("cand", CANDIDATE))
    assert c.mean_diff == pytest.approx((6 * 0.2 - 0.1) / 8)
    assert c.ci_low < c.mean_diff < c.ci_high
    assert (c.baseline, c.candidate) == ("base", "cand")


def test_models_scored_on_different_clips_cannot_be_compared():
    short = [Run(model="cand", seed=i, predictions=dysarthric_preds(CANDIDATE).iloc[1:],
                 fold_train_speakers=loso_folds()) for i in range(3)]
    with pytest.raises(EvalValidationError, match="same clips"):
        compare(seeds("base", BASELINE), short)


def test_zero_shot_baseline_scored_on_controls_too_can_still_be_compared():
    controls = pd.DataFrame(speaker_rows("FC01", 10, 10))
    whisper = [Run(model="whisper", seed=0, zero_shot=True,
                   predictions=pd.concat([dysarthric_preds(BASELINE), controls],
                                         ignore_index=True))]
    c = compare(whisper, seeds("cand", CANDIDATE))
    assert (c.n_better, c.n_worse, c.n_tied) == (6, 1, 1)
