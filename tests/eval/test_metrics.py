"""Tests for single-run metrics. Expected values are worked out by hand in comments."""

import pandas as pd
import pytest

from src.eval.constants import ARRAY_MIC, HEAD_MIC
from src.eval.metrics import compute_run_metrics
from src.eval.schema import EvalValidationError, Run
from tests.eval.factories import (
    ALL_DYSARTHRIC, dysarthric_preds, loso_folds, pred_row, speaker_rows,
)


def trained(preds):
    return Run(model="m", seed=0, predictions=preds, fold_train_speakers=loso_folds())


def uneven_preds():
    # F01: 2 clips, both right. Other 7 speakers: 10 clips, 5 right.
    rows = speaker_rows("F01", 2, 2)
    for spk in ALL_DYSARTHRIC:
        if spk != "F01":
            rows += speaker_rows(spk, 10, 5)
    return pd.DataFrame(rows)


def test_headline_weights_each_speaker_equally():
    m = compute_run_metrics(trained(uneven_preds()), ARRAY_MIC)
    # (1.0 + 7 * 0.5) / 8
    assert m.headline == pytest.approx(0.5625)


def test_pooled_accuracy_weights_each_clip_equally():
    m = compute_run_metrics(trained(uneven_preds()), ARRAY_MIC)
    # (2 + 7 * 5) / (2 + 70)
    assert m.pooled == pytest.approx(37 / 72)


def test_per_speaker_table_has_counts_and_accuracy():
    m = compute_run_metrics(trained(uneven_preds()), ARRAY_MIC)
    assert m.per_speaker.loc["F01", "n"] == 2
    assert m.per_speaker.loc["F01", "accuracy"] == pytest.approx(1.0)
    assert m.per_speaker.loc["M03", "accuracy"] == pytest.approx(0.5)
    assert list(m.per_speaker.index) == list(ALL_DYSARTHRIC)


def test_only_the_requested_mic_is_scored():
    array = dysarthric_preds({})                     # all right
    head = dysarthric_preds({s: 0 for s in ALL_DYSARTHRIC}, mic=HEAD_MIC)  # all wrong
    run = trained(pd.concat([array, head], ignore_index=True))
    assert compute_run_metrics(run, ARRAY_MIC).headline == pytest.approx(1.0)
    assert compute_run_metrics(run, HEAD_MIC).headline == pytest.approx(0.0)


def test_severity_groups_average_their_speakers():
    correct = {"F01": 10, "M01": 0, "M02": 10, "M04": 0,   # severe: 0.5
               "M05": 5,                                   # moderate-severe: 0.5
               "F03": 10, "F04": 10, "M03": 4}             # mild: (1 + 1 + 0.4) / 3
    m = compute_run_metrics(trained(dysarthric_preds(correct)), ARRAY_MIC)
    assert m.severity == pytest.approx({"severe": 0.5, "moderate-severe": 0.5, "mild": 0.8})


def test_speaker_with_no_clips_on_the_mic_is_an_error():
    preds = dysarthric_preds({})
    preds.loc[preds.speaker_id == "M05", "mic"] = HEAD_MIC
    with pytest.raises(EvalValidationError, match="M05"):
        compute_run_metrics(trained(preds), ARRAY_MIC)


def word_preds():
    # F01 says "back" x2 (1 right, 1 heard as "up") and "yes" x2 (both right).
    # Everyone else: 10 x "yes", all right.
    rows = [pred_row("F01", "0100", "back", "back"),
            pred_row("F01", "0101", "back", "up"),
            pred_row("F01", "0102", "yes", "yes"),
            pred_row("F01", "0103", "yes", "oov")]
    for spk in ALL_DYSARTHRIC:
        if spk != "F01":
            rows += speaker_rows(spk, 10, 10)
    return pd.DataFrame(rows)


def test_per_word_accuracy_pools_clips_of_that_word():
    m = compute_run_metrics(trained(word_preds()), ARRAY_MIC)
    assert m.per_word.loc["back", "n"] == 2
    assert m.per_word.loc["back", "accuracy"] == pytest.approx(0.5)
    # yes: 70 right of 70 + 1 right of 2 from F01
    assert m.per_word.loc["yes", "accuracy"] == pytest.approx(71 / 72)
    assert "menu" not in m.per_word.index  # no clips, no row


def test_confusion_counts_true_label_by_prediction_including_oov():
    m = compute_run_metrics(trained(word_preds()), ARRAY_MIC)
    assert m.confusion.loc["back", "up"] == 1
    assert m.confusion.loc["back", "back"] == 1
    assert m.confusion.loc["yes", "oov"] == 1
    assert m.confusion.loc["yes", "yes"] == 71
    assert m.confusion.to_numpy().sum() == 74


def test_words_missing_from_speech_commands_are_split_out():
    m = compute_run_metrics(trained(word_preds()), ARRAY_MIC)
    assert m.pretraining_overlap["not_in_speech_commands"] == pytest.approx(0.5)  # back: 1/2
    assert m.pretraining_overlap["in_speech_commands"] == pytest.approx(71 / 72)  # yes


def test_oov_and_reject_rates_count_non_command_predictions():
    preds = dysarthric_preds({})
    preds.loc[0:3, "pred"] = "oov"      # 4 of 80
    preds.loc[4, "pred"] = "reject"     # 1 of 80
    m = compute_run_metrics(trained(preds), ARRAY_MIC)
    assert m.oov_rate == pytest.approx(4 / 80)
    assert m.reject_rate == pytest.approx(1 / 80)


def test_zero_shot_run_reports_controls_separately():
    controls = pd.DataFrame(speaker_rows("FC01", 10, 10) + speaker_rows("MC01", 10, 0))
    run = Run(model="whisper", seed=0, zero_shot=True,
              predictions=pd.concat([dysarthric_preds({s: 2 for s in ALL_DYSARTHRIC}), controls],
                                    ignore_index=True))
    m = compute_run_metrics(run, ARRAY_MIC)
    assert m.headline == pytest.approx(0.2)          # controls not mixed in
    assert m.control_headline == pytest.approx(0.5)  # (1.0 + 0.0) / 2


def test_trained_run_has_no_control_number():
    m = compute_run_metrics(trained(dysarthric_preds({})), ARRAY_MIC)
    assert m.control_headline is None
