"""Tests for run validation: the checks that stop leaks and miscounts reaching a table."""

import pandas as pd
import pytest

from src.eval.constants import ARRAY_MIC, HEAD_MIC
from src.eval.metrics import compute_run_metrics
from src.eval.schema import EvalValidationError, Run, check_same_clips, clip_fingerprint
from tests.eval.factories import (
    ALL_DYSARTHRIC, dysarthric_preds, loso_folds, pred_row, speaker_rows,
)


def trained_run(preds, folds=None, **kw):
    return Run(model="m", seed=0, predictions=preds,
               fold_train_speakers=loso_folds() if folds is None else folds, **kw)


def test_valid_leak_free_run_is_accepted():
    run = trained_run(dysarthric_preds({}))
    assert run.model == "m"


def test_missing_column_is_rejected():
    preds = dysarthric_preds({}).drop(columns=["session"])
    with pytest.raises(EvalValidationError, match="session"):
        trained_run(preds)


def test_label_outside_vocabulary_is_rejected():
    preds = dysarthric_preds({})
    preds.loc[0, "label"] = "banana"
    with pytest.raises(EvalValidationError, match="banana"):
        trained_run(preds)


@pytest.mark.parametrize("pred", ["reject", "oov", "menu"])
def test_reject_oov_and_commands_are_valid_predictions(pred):
    preds = dysarthric_preds({})
    preds.loc[0, "pred"] = pred
    trained_run(preds)


def test_prediction_outside_vocabulary_is_rejected():
    preds = dysarthric_preds({})
    preds.loc[0, "pred"] = "four."
    with pytest.raises(EvalValidationError, match="four."):
        trained_run(preds)


def test_missing_prediction_is_rejected():
    preds = dysarthric_preds({})
    preds.loc[0, "pred"] = None
    with pytest.raises(EvalValidationError, match="missing"):
        trained_run(preds)


def test_unknown_mic_is_rejected():
    preds = dysarthric_preds({})
    preds.loc[0, "mic"] = "wav_phone"
    with pytest.raises(EvalValidationError, match="wav_phone"):
        trained_run(preds)


def test_same_utterance_and_mic_scored_twice_is_rejected():
    preds = dysarthric_preds({})
    preds = pd.concat([preds, preds.iloc[[0]]], ignore_index=True)
    with pytest.raises(EvalValidationError, match="more than once"):
        trained_run(preds)


def test_same_utterance_on_both_mics_is_not_a_duplicate():
    preds = dysarthric_preds({})
    head = preds.assign(mic=HEAD_MIC)
    trained_run(pd.concat([preds, head], ignore_index=True))


def test_same_utterance_id_in_two_sessions_is_not_a_duplicate():
    # TORGO numbers prompts per session, so "0001" repeats across sessions
    preds = dysarthric_preds({})
    other = preds.assign(session="Session2")
    trained_run(pd.concat([preds, other], ignore_index=True))


def test_missing_dysarthric_speaker_is_rejected():
    preds = dysarthric_preds({})
    preds = preds[preds.speaker_id != "M05"]
    with pytest.raises(EvalValidationError, match="M05"):
        trained_run(preds)


def test_unknown_speaker_is_rejected():
    preds = pd.concat([dysarthric_preds({}),
                       pd.DataFrame([pred_row("X99", "0001", "yes", "yes")])])
    with pytest.raises(EvalValidationError, match="X99"):
        trained_run(preds)


def test_held_out_speaker_in_its_own_training_set_is_rejected():
    folds = dict(loso_folds())
    folds["M04"] = folds["M04"] | {"M04"}
    with pytest.raises(EvalValidationError, match="M04"):
        trained_run(dysarthric_preds({}), folds=folds)


def test_trained_run_without_fold_for_a_test_speaker_is_rejected():
    folds = {k: v for k, v in loso_folds().items() if k != "F01"}
    with pytest.raises(EvalValidationError, match="F01"):
        trained_run(dysarthric_preds({}), folds=folds)


def test_control_speakers_are_rejected_for_trained_models():
    # Trained models learn from all controls in Phase A, so scoring them leaks
    preds = pd.concat([dysarthric_preds({}),
                       pd.DataFrame([pred_row("FC01", "0001", "yes", "yes")])])
    folds = {**loso_folds(), "FC01": frozenset()}
    with pytest.raises(EvalValidationError, match="FC01"):
        trained_run(preds, folds=folds)


def test_zero_shot_run_may_include_controls_and_needs_no_folds():
    preds = pd.concat([dysarthric_preds({}),
                       pd.DataFrame([pred_row("FC01", "0001", "yes", "yes")])])
    run = Run(model="whisper", seed=0, predictions=preds, zero_shot=True)
    assert run.zero_shot


def test_zero_shot_run_with_training_folds_is_rejected():
    with pytest.raises(EvalValidationError, match="zero-shot"):
        Run(model="whisper", seed=0, predictions=dysarthric_preds({}),
            zero_shot=True, fold_train_speakers=loso_folds())


def test_fingerprint_ignores_row_order_and_predictions():
    a = dysarthric_preds({})
    b = dysarthric_preds({"F01": 0}).iloc[::-1]
    assert clip_fingerprint(a) == clip_fingerprint(b)


def test_fingerprint_changes_when_a_clip_is_dropped():
    a = dysarthric_preds({})
    assert clip_fingerprint(a) != clip_fingerprint(a.iloc[1:])


def test_runs_scored_on_different_clips_are_rejected():
    a = trained_run(dysarthric_preds({}))
    b = trained_run(dysarthric_preds({}).iloc[1:])
    with pytest.raises(EvalValidationError, match="same clips"):
        check_same_clips([a, b])


def test_runs_scored_on_the_same_clips_pass():
    a = trained_run(dysarthric_preds({}))
    b = trained_run(dysarthric_preds({"F01": 3}))
    check_same_clips([a, b])


def test_runs_with_different_labels_for_the_same_clips_are_rejected():
    # e.g. the manual labels file changed between a baseline and a candidate run
    a = trained_run(dysarthric_preds({}))
    relabelled = dysarthric_preds({})
    relabelled.loc[relabelled.speaker_id == "F01", "label"] = "no"
    with pytest.raises(EvalValidationError, match="same clips"):
        check_same_clips([a, trained_run(relabelled)])


@pytest.mark.parametrize("bad_value", ["M01", frozenset({"m01"}), frozenset({"X99"})])
def test_fold_training_speakers_must_be_a_set_of_known_speakers(bad_value):
    # A bare string "M01" would become {"M", "0", "1"} and hide the leak
    folds = {**loso_folds(), "M01": bad_value}
    with pytest.raises(EvalValidationError, match="M01"):
        trained_run(dysarthric_preds({}), folds=folds)


def test_concatenated_per_fold_predictions_can_be_scored():
    # Each fold's frame is indexed from 0, so a plain concat repeats index labels
    per_fold = [pd.DataFrame(speaker_rows(s, 10, 7)) for s in ALL_DYSARTHRIC]
    run = trained_run(pd.concat(per_fold))
    assert compute_run_metrics(run, ARRAY_MIC).confusion.loc["yes", "yes"] == 56
