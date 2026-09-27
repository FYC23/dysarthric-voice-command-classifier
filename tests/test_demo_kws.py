"""Tests for the demo's BC-ResNet keyword spotter."""

from pathlib import Path

import numpy as np
import pytest
import torch

from src.config import Config
from src.demo.kws import CHECKPOINT_CLASSES, CheckpointError, KeywordSpotter
from src.eval.constants import COMMANDS
from src.model.bcresnet import BCResNet
from src.training.loso import TORGO_CLASSES
from tests.demo_factories import save_checkpoint

DEPLOY = Path(Config.RUNS_DIR) / "bcresnet-8" / "seed0" / "deploy.pt"
SILENCE = np.zeros(Config.MAX_AUDIO_SAMPLES, dtype=np.float32)


def test_class_order_matches_training():
    assert CHECKPOINT_CLASSES == TORGO_CLASSES


def test_predict_returns_every_command_summing_to_one(tmp_path):
    kws = KeywordSpotter.from_checkpoint(save_checkpoint(tmp_path / "deploy.pt"))
    probs, latency_ms = kws.predict(SILENCE)
    assert set(probs) == set(COMMANDS)
    assert sum(probs.values()) == pytest.approx(1.0, abs=1e-5)
    assert all(np.isfinite(p) for p in probs.values())
    assert latency_ms >= 0


def test_probabilities_follow_the_checkpoint_class_order(tmp_path):
    model = BCResNet(1, len(CHECKPOINT_CLASSES))
    with torch.no_grad():
        model.head.bias.zero_()
        model.head.bias[CHECKPOINT_CLASSES.index("zero")] = 100.0
    kws = KeywordSpotter.from_checkpoint(save_checkpoint(tmp_path / "deploy.pt", model=model))
    probs, _ = kws.predict(SILENCE)
    assert max(probs, key=probs.get) == "zero"


def test_tau_is_read_from_the_checkpoint(tmp_path):
    kws = KeywordSpotter.from_checkpoint(save_checkpoint(tmp_path / "deploy.pt", tau=2.0))
    assert kws.tau == 2.0


def test_twelve_class_checkpoint_is_refused(tmp_path):
    path = save_checkpoint(tmp_path / "pretrain.pt", classes=CHECKPOINT_CLASSES[:12])
    with pytest.raises(CheckpointError, match="not the 20 commands"):
        KeywordSpotter.from_checkpoint(path)


def test_reordered_classes_are_refused(tmp_path):
    path = save_checkpoint(tmp_path / "deploy.pt", classes=COMMANDS)  # Config order, not sorted
    with pytest.raises(CheckpointError, match="not the 20 commands"):
        KeywordSpotter.from_checkpoint(path)


def test_missing_keys_are_refused(tmp_path):
    path = tmp_path / "weights.pt"
    torch.save({"model_state_dict": BCResNet(1, 20).state_dict()}, path)
    with pytest.raises(CheckpointError, match="needs keys"):
        KeywordSpotter.from_checkpoint(path)


def test_weights_of_another_width_are_refused(tmp_path):
    path = tmp_path / "deploy.pt"
    torch.save({"model_state_dict": BCResNet(1, 20).state_dict(), "tau": 8.0,
                "classes": list(CHECKPOINT_CLASSES)}, path)
    with pytest.raises(CheckpointError, match="do not fit BC-ResNet tau=8"):
        KeywordSpotter.from_checkpoint(path)


def test_unreadable_file_is_refused(tmp_path):
    path = tmp_path / "deploy.pt"
    path.write_text("not a checkpoint")
    with pytest.raises(CheckpointError, match="could not be read"):
        KeywordSpotter.from_checkpoint(path)


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        KeywordSpotter.from_checkpoint(tmp_path / "nope.pt")


def test_two_dimensional_input_is_refused(tmp_path):
    kws = KeywordSpotter.from_checkpoint(save_checkpoint(tmp_path / "deploy.pt"))
    with pytest.raises(ValueError, match="1-D"):
        kws.predict(np.zeros((2, 100), dtype=np.float32))


@pytest.mark.skipif(not DEPLOY.is_file(), reason="needs the trained BC-ResNet-8 deploy model")
def test_real_deploy_checkpoint_loads():
    kws = KeywordSpotter.from_checkpoint(DEPLOY)
    assert kws.tau == 8.0
    probs, _ = kws.predict(SILENCE)
    assert sum(probs.values()) == pytest.approx(1.0, abs=1e-5)
