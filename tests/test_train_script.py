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
