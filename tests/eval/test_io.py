"""Tests for saving and loading runs (predictions.csv + run.json)."""

import json

import pytest

from src.eval.io import load_run, save_run
from src.eval.schema import EvalValidationError, Run
from tests.eval.factories import dysarthric_preds, loso_folds


def test_round_trip_keeps_run_metadata_and_zero_padded_ids(tmp_path):
    run = Run(model="bcresnet8", seed=2, predictions=dysarthric_preds({"M04": 3}),
              fold_train_speakers=loso_folds())
    save_run(run, tmp_path / "r")
    loaded = load_run(tmp_path / "r")

    assert (loaded.model, loaded.seed, loaded.zero_shot) == ("bcresnet8", 2, False)
    assert loaded.fold_train_speakers == loso_folds()
    assert loaded.predictions["utterance_id"].iloc[0] == "0000"
    assert (loaded.predictions["pred"] == run.predictions["pred"]).all()


def test_loading_a_leaky_run_fails_validation(tmp_path):
    run = Run(model="m", seed=0, predictions=dysarthric_preds({}),
              fold_train_speakers=loso_folds())
    save_run(run, tmp_path / "r")
    meta_path = tmp_path / "r" / "run.json"
    meta = json.loads(meta_path.read_text())
    meta["fold_train_speakers"]["M01"].append("M01")
    meta_path.write_text(json.dumps(meta))

    with pytest.raises(EvalValidationError, match="M01"):
        load_run(tmp_path / "r")


def test_fold_stored_as_a_string_is_rejected_on_load(tmp_path):
    run = Run(model="m", seed=0, predictions=dysarthric_preds({}),
              fold_train_speakers=loso_folds())
    save_run(run, tmp_path / "r")
    meta_path = tmp_path / "r" / "run.json"
    meta = json.loads(meta_path.read_text())
    meta["fold_train_speakers"]["M01"] = "M01"
    meta_path.write_text(json.dumps(meta))

    with pytest.raises(EvalValidationError, match="M01"):
        load_run(tmp_path / "r")
