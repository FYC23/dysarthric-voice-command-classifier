"""Tests for combining seeds into one model summary."""

import warnings

import numpy as np
import pytest

from src.eval.aggregate import bootstrap_mean_ci, summarize
from src.eval.constants import ARRAY_MIC
from src.eval.schema import EvalValidationError, Run
from tests.eval.factories import ALL_DYSARTHRIC, dysarthric_preds, loso_folds


def run(correct, seed=0, model="m"):
    return Run(model=model, seed=seed, predictions=dysarthric_preds(correct),
               fold_train_speakers=loso_folds())


def uniform(k):
    """Every speaker gets k of 10 right."""
    return {s: k for s in ALL_DYSARTHRIC}


def test_headline_is_mean_over_seeds_with_sample_std():
    s = summarize([run(uniform(5), 0), run(uniform(6), 1), run(uniform(7), 2)])
    assert s.headline == pytest.approx(0.6)
    assert s.seed_std == pytest.approx(0.1)  # sample std of 0.5, 0.6, 0.7
    assert s.n_seeds == 3


def test_per_speaker_accuracy_is_averaged_over_seeds():
    a = run({**uniform(10), "M04": 2}, 0)
    b = run({**uniform(10), "M04": 4}, 1)
    c = run({**uniform(10), "M04": 6}, 2)
    s = summarize([a, b, c])
    assert s.per_speaker.loc["M04", "accuracy"] == pytest.approx(0.4)
    assert s.per_speaker.loc["M04", "n"] == 10


def test_severity_is_averaged_over_seeds():
    s = summarize([run({**uniform(10), "M05": 0}, 0), run({**uniform(10), "M05": 10}, 1),
                   run(uniform(10), 2)])
    assert s.severity["moderate-severe"] == pytest.approx(2 / 3)
    assert s.severity["severe"] == pytest.approx(1.0)


def test_ci_collapses_when_every_speaker_scores_the_same():
    s = summarize([run(uniform(7), i) for i in range(3)])
    assert (s.ci_low, s.ci_high) == pytest.approx((0.7, 0.7))


def test_ci_brackets_headline_when_speakers_differ():
    correct = dict(zip(ALL_DYSARTHRIC, [2, 4, 6, 8, 10, 3, 5, 9]))
    s = summarize([run(correct, i) for i in range(3)])
    assert s.ci_low < s.headline < s.ci_high


def test_bootstrap_of_two_values_spans_both_extremes():
    # Resampling [0, 1] twice gives means 0, 0.5, 1 with probability 1/4, 1/2, 1/4,
    # so the 2.5th and 97.5th percentiles are 0 and 1.
    assert bootstrap_mean_ci(np.array([0.0, 1.0]), n_boot=10_000, rng_seed=0) == (0.0, 1.0)


def test_bootstrap_is_reproducible_for_a_fixed_seed():
    values = np.array([0.2, 0.5, 0.9, 0.4, 0.7])
    assert bootstrap_mean_ci(values, 2000, 3) == bootstrap_mean_ci(values, 2000, 3)


def test_runs_from_different_models_are_rejected():
    with pytest.raises(EvalValidationError, match="one model"):
        summarize([run(uniform(5), 0, "a"), run(uniform(5), 1, "b")])


def test_repeated_seed_is_rejected():
    with pytest.raises(EvalValidationError, match="seed"):
        summarize([run(uniform(5), 0), run(uniform(6), 0)])


def test_runs_on_different_clips_are_rejected():
    short = Run(model="m", seed=1, predictions=dysarthric_preds({}).iloc[1:],
                fold_train_speakers=loso_folds())
    with pytest.raises(EvalValidationError, match="same clips"):
        summarize([run(uniform(5), 0), short])


def test_fewer_than_three_seeds_warns_for_trained_models():
    with pytest.warns(UserWarning, match="3 seeds"):
        summarize([run(uniform(5), 0)])


def test_single_zero_shot_run_does_not_warn_and_has_no_seed_std():
    zs = Run(model="whisper", seed=0, predictions=dysarthric_preds(uniform(4)), zero_shot=True)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        s = summarize([zs])
    assert s.headline == pytest.approx(0.4)
    assert np.isnan(s.seed_std)


def test_confusion_is_summed_over_seeds():
    s = summarize([run(uniform(10), i) for i in range(3)])
    assert s.confusion.loc["yes", "yes"] == 3 * 80


def test_summary_records_model_and_mic():
    s = summarize([run(uniform(5), i) for i in range(3)], mic=ARRAY_MIC)
    assert (s.model, s.mic) == ("m", ARRAY_MIC)
