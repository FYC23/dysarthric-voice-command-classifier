"""scripts/train.py: evaluation stays on real audio; missing noise fails fast."""

import importlib.util
from pathlib import Path

import pandas as pd
import pytest
import torch

TRAIN_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "train.py"


def load_train_script():
    spec = importlib.util.spec_from_file_location("train_script", TRAIN_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _DatasetBuilt(Exception):
    pass


def test_evaluate_checkpoint_never_augments(monkeypatch, tmp_path):
    train = load_train_script()
    built = {}

    def record(*args, **kwargs):
        built.update(kwargs)
        raise _DatasetBuilt

    monkeypatch.setattr(train, "TORGOCommandDataset", record)
    with pytest.raises(_DatasetBuilt):
        train.evaluate_checkpoint("unused", tmp_path / "missing.pt", pd.DataFrame(), {}, None,
                                  torch.device("cpu"), 1)
    assert built["augment"] is False
    assert built.get("noise") is None


def test_training_noise_bank_fails_fast_without_speech_commands(monkeypatch, tmp_path):
    train = load_train_script()
    monkeypatch.setattr(train.config, "NOISE_DIR", tmp_path / "missing")
    with pytest.raises(FileNotFoundError, match="download_speech_commands"):
        train.training_noise_bank()


# --- Phase C output in the eval harness format (src/eval) -------------------

DYSARTHRIC = ("F01", "F03", "F04", "M01", "M02", "M03", "M04", "M05")
CONTROLS = ("FC01", "MC01")
LABEL2ID = {"no": 0, "yes": 1}
ID2LABEL = {0: "no", 1: "yes"}


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


def test_fold_predictions_pair_each_clip_with_its_predicted_word():
    train = load_train_script()
    rows = train.fold_predictions(fold_test_df("M04"), pred_ids=[1, 1], label_ids=[1, 0],
                                  id2label=ID2LABEL)
    assert rows.to_dict("records") == [
        {"speaker_id": "M04", "session": "Session1", "utterance_id": "0001",
         "mic": "wav_arrayMic", "label": "yes", "pred": "yes"},
        {"speaker_id": "M04", "session": "Session2", "utterance_id": "0001",
         "mic": "wav_headMic", "label": "no", "pred": "yes"},
    ]


def test_fold_predictions_refuse_labels_out_of_order():
    # If the DataLoader order ever stopped matching the table, predictions would
    # be attached to the wrong clips; the returned labels expose it.
    train = load_train_script()
    with pytest.raises(ValueError, match="order"):
        train.fold_predictions(fold_test_df("M04"), pred_ids=[1, 1], label_ids=[0, 1],
                               id2label=ID2LABEL)


def loso_inputs():
    frames = [fold_test_df(s).pipe(lambda d: d.assign(pred=d["label"]))
              [["speaker_id", "session", "utterance_id", "mic", "label", "pred"]]
              for s in DYSARTHRIC]
    folds = {s: frozenset(set(DYSARTHRIC) - {s}) for s in DYSARTHRIC}
    return frames, folds


def test_loso_run_records_controls_in_every_fold_and_validates():
    train = load_train_script()
    frames, folds = loso_inputs()
    run = train.build_loso_run(frames, folds, CONTROLS, seed=3)
    assert run.seed == 3 and not run.zero_shot
    assert run.fold_train_speakers["M04"] == frozenset(set(DYSARTHRIC) - {"M04"}) | set(CONTROLS)
    assert len(run.predictions) == 16


def test_saved_loso_run_loads_back_through_the_harness(tmp_path):
    from src.eval.io import load_run

    train = load_train_script()
    frames, folds = loso_inputs()
    out = train.save_loso_run(frames, folds, CONTROLS, seed=3, out_dir=tmp_path / "eval")
    loaded = load_run(out)
    assert loaded.seed == 3
    assert loaded.model == train.RUN_NAME


def test_invalid_loso_run_keeps_raw_predictions_before_failing(tmp_path):
    # Eight GPU folds must not be lost to a validation error: keep the raw rows
    from src.eval.schema import EvalValidationError

    train = load_train_script()
    frames, folds = loso_inputs()
    folds["M04"] = folds["M04"] | {"M04"}  # leak
    with pytest.raises(EvalValidationError, match="M04"):
        train.save_loso_run(frames, folds, CONTROLS, seed=3, out_dir=tmp_path / "eval")
    raw = pd.read_csv(tmp_path / "eval" / "predictions_unvalidated.csv")
    assert len(raw) == 16


def test_each_seed_gets_its_own_run_directory(monkeypatch, tmp_path):
    train = load_train_script()
    monkeypatch.setattr(train.config, "RUNS_DIR", tmp_path)
    assert train.seed_dir(0) != train.seed_dir(1)
    assert train.seed_dir(0).parent.parent == tmp_path
