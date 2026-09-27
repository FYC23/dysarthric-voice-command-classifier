"""
Leave-one-speaker-out helpers shared by every trained model (the SSL
backbones and BC-ResNet): the 20-command label table, the fold split,
eval-harness rows, building and saving a run, class weights and seeding.
"""

import numpy as np
import pandas as pd
import pytest
import torch

from src.config import Config
from src.eval.io import load_run
from src.eval.schema import CLIP_KEY, EvalValidationError
from src.training import bcresnet_recipe, finetune
from src.training.device import pick_device
from src.training.loso import (
    ID2LABEL, LABEL2ID, TORGO_CLASSES, UNVALIDATED_PREDICTIONS, balanced_class_weights,
    build_loso_run, fold_predictions, save_loso_run, set_seed, split_fold, with_label_ids,
)

DYSARTHRIC = ("F01", "F03", "F04", "M01", "M02", "M03", "M04", "M05")
CONTROLS = ("FC01", "MC01")
YES_NO_ID2LABEL = {0: "no", 1: "yes"}
MODEL = "bcresnet-1"


def fold_test_df(speaker):
    """Two clips of one speaker: 'yes' on the array mic, 'no' on the head mic."""
    return pd.DataFrame({
        "file_path": [f"/x/{speaker}/a.wav", f"/x/{speaker}/b.wav"],
        "speaker_id": [speaker, speaker],
        "session": ["Session1", "Session2"],
        "utterance_id": ["0001", "0001"],
        "mic": ["wav_arrayMic", "wav_headMic"],
        "label": ["yes", "no"],
        "label_id": [1, 0],
    }, index=[40, 7])  # a filtered frame keeps its original index


def loso_inputs():
    frames = [fold_test_df(s).pipe(lambda d: d.assign(pred=d["label"]))
              [["speaker_id", "session", "utterance_id", "mic", "label", "pred"]]
              for s in DYSARTHRIC]
    folds = {s: frozenset(set(DYSARTHRIC) - {s}) for s in DYSARTHRIC}
    return frames, folds


def test_fold_predictions_pair_each_clip_with_its_predicted_word():
    rows = fold_predictions(fold_test_df("M04"), pred_ids=[1, 1], label_ids=[1, 0],
                            id2label=YES_NO_ID2LABEL)
    assert rows.to_dict("records") == [
        {"speaker_id": "M04", "session": "Session1", "utterance_id": "0001",
         "mic": "wav_arrayMic", "label": "yes", "pred": "yes"},
        {"speaker_id": "M04", "session": "Session2", "utterance_id": "0001",
         "mic": "wav_headMic", "label": "no", "pred": "yes"},
    ]


def test_fold_predictions_refuse_labels_out_of_order():
    # If the DataLoader order ever stopped matching the table, predictions would
    # be attached to the wrong clips; the returned labels expose it.
    with pytest.raises(ValueError, match="order"):
        fold_predictions(fold_test_df("M04"), pred_ids=[1, 1], label_ids=[0, 1],
                         id2label=YES_NO_ID2LABEL)


def test_loso_run_records_controls_in_every_fold_and_validates():
    frames, folds = loso_inputs()
    run = build_loso_run(frames, folds, CONTROLS, seed=3, model=MODEL)
    assert run.model == MODEL and run.seed == 3 and not run.zero_shot
    assert run.fold_train_speakers["M04"] == frozenset(set(DYSARTHRIC) - {"M04"}) | set(CONTROLS)
    assert len(run.predictions) == 16


def test_saved_loso_run_loads_back_through_the_harness(tmp_path):
    frames, folds = loso_inputs()
    out = save_loso_run(frames, folds, CONTROLS, seed=3, model=MODEL, out_dir=tmp_path / "eval")
    loaded = load_run(out)
    assert loaded.seed == 3
    assert loaded.model == MODEL


def test_invalid_loso_run_keeps_raw_predictions_before_failing(tmp_path):
    frames, folds = loso_inputs()
    folds["M04"] = folds["M04"] | {"M04"}  # leak
    with pytest.raises(EvalValidationError, match="M04"):
        save_loso_run(frames, folds, CONTROLS, seed=3, model=MODEL, out_dir=tmp_path / "eval")
    raw = pd.read_csv(tmp_path / "eval" / UNVALIDATED_PREDICTIONS)
    assert len(raw) == 16


def test_fold_test_set_scores_each_recording_once_but_training_keeps_every_attempt():
    base = {"session": "Session1", "mic": "wav_arrayMic", "label": "yes", "label_id": 1,
            "file_path": "/x.wav"}
    df = pd.DataFrame([
        {**base, "speaker_id": "F04", "utterance_id": "0067", "seg_start": 3.8, "seg_end": 4.1},
        {**base, "speaker_id": "F04", "utterance_id": "0067", "seg_start": 1.0, "seg_end": 1.6},
        {**base, "speaker_id": "M05", "utterance_id": "0093", "seg_start": 0.5, "seg_end": 0.9},
        {**base, "speaker_id": "M05", "utterance_id": "0093", "seg_start": 2.0, "seg_end": 2.4},
    ])
    train_df, test_df = split_fold(df, "F04")
    assert test_df.seg_start.tolist() == [1.0]
    assert not test_df.duplicated(list(CLIP_KEY)).any()
    assert len(train_df) == 2 and set(train_df.speaker_id) == {"M05"}


def test_balanced_class_weights_match_sklearn_and_leave_absent_classes_at_one():
    from sklearn.utils.class_weight import compute_class_weight

    labels = [0, 0, 0, 2]
    weights = balanced_class_weights(labels, num_classes=4)
    expected = compute_class_weight("balanced", classes=np.array([0, 2]), y=np.array(labels))
    assert weights.dtype == torch.float32
    assert torch.allclose(weights, torch.tensor([expected[0], 1.0, expected[1], 1.0],
                                                dtype=torch.float32))


def test_balanced_class_weights_reject_out_of_range_labels():
    with pytest.raises(ValueError, match="label"):
        balanced_class_weights([0, 5], num_classes=4)


def test_set_seed_makes_torch_and_numpy_repeat():
    set_seed(7)
    first = (torch.rand(1).item(), np.random.rand())
    set_seed(7)
    assert (torch.rand(1).item(), np.random.rand()) == first


def test_label_table_is_the_20_sorted_commands():
    assert TORGO_CLASSES == tuple(sorted(Config.TARGET_COMMANDS))
    assert LABEL2ID == {w: i for i, w in enumerate(TORGO_CLASSES)}
    assert ID2LABEL == dict(enumerate(TORGO_CLASSES))


def test_with_label_ids_uses_the_fixed_table_and_keeps_the_input():
    samples = pd.DataFrame({"label": ["zero", "back"]})
    out = with_label_ids(samples)
    assert out["label_id"].tolist() == [LABEL2ID["zero"], LABEL2ID["back"]]
    assert "label_id" not in samples.columns


def test_with_label_ids_rejects_other_words():
    with pytest.raises(ValueError, match="outside the 20 commands"):
        with_label_ids(pd.DataFrame({"label": ["yes", "hello"]}))


def test_pick_device_honours_an_explicit_name():
    assert pick_device("cpu") == torch.device("cpu")


def test_bcresnet_code_uses_the_shared_helpers():
    assert bcresnet_recipe.pick_device is pick_device
    assert finetune.TORGO_CLASSES is TORGO_CLASSES
    assert finetune._with_label_ids is with_label_ids
