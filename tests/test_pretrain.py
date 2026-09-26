"""Stage 1: Speech Commands pretraining, on a tiny fake tree."""

import importlib.util
import json
from pathlib import Path

import pytest
import torch

from src.data.speech_commands import CLASSES
from src.training.bcresnet_recipe import PRETRAIN, SgdStage, seed_dir
from src.training.pretrain import (
    BEST_CHECKPOINT, LAST_CHECKPOINT, METRICS_FILE, PretrainJob, run_pretraining,
)

TINY = SgdStage(epochs=2, peak_lr=0.05, warmup_epochs=1, batch_size=16)
SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "pretrain_bcresnet.py"


def make_job(sc_root, tmp_path, **overrides):
    base = dict(tau=1, seed=0, out_dir=tmp_path / "run", sc_root=sc_root,
                cache_dir=tmp_path / "cache", noise_dir=sc_root / "train" / "_silence_",
                device=torch.device("cpu"), stage=TINY, num_workers=0)
    return PretrainJob(**{**base, **overrides})


def load_script():
    spec = importlib.util.spec_from_file_location("pretrain_script", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_pretraining_writes_best_checkpoint_and_metrics(fake_speech_commands, tmp_path):
    metrics = run_pretraining(make_job(fake_speech_commands, tmp_path))
    out = tmp_path / "run"
    assert {BEST_CHECKPOINT, LAST_CHECKPOINT, METRICS_FILE} <= {p.name for p in out.iterdir()}
    best = torch.load(out / BEST_CHECKPOINT)
    assert best["tau"] == 1 and best["classes"] == list(CLASSES)
    assert best["epoch"] == metrics["best_epoch"] and best["val_acc"] == metrics["best_val_acc"]
    assert [h["epoch"] for h in metrics["history"]] == [1, 2]
    assert all(h["seconds"] > 0 for h in metrics["history"])
    assert 0.0 <= metrics["test_acc"] <= 1.0
    assert json.loads((out / METRICS_FILE).read_text())["test_acc"] == metrics["test_acc"]


def test_rerun_without_resume_refuses_to_overwrite(fake_speech_commands, tmp_path):
    run_pretraining(make_job(fake_speech_commands, tmp_path))
    with pytest.raises(FileExistsError, match="--resume"):
        run_pretraining(make_job(fake_speech_commands, tmp_path))


def test_resume_continues_from_the_last_epoch(fake_speech_commands, tmp_path):
    run_pretraining(make_job(fake_speech_commands, tmp_path))
    last = tmp_path / "run" / LAST_CHECKPOINT
    checkpoint = torch.load(last)
    checkpoint["epoch"] = 1  # pretend the run stopped after epoch 1
    checkpoint["history"] = checkpoint["history"][:1]
    torch.save(checkpoint, last)
    metrics = run_pretraining(make_job(fake_speech_commands, tmp_path, resume=True))
    assert [h["epoch"] for h in metrics["history"]] == [1, 2]


def test_resume_refuses_another_width(fake_speech_commands, tmp_path):
    run_pretraining(make_job(fake_speech_commands, tmp_path))
    with pytest.raises(ValueError, match="tau"):
        run_pretraining(make_job(fake_speech_commands, tmp_path, tau=2, resume=True))


def test_epoch_override_needs_its_own_out_dir():
    script = load_script()
    with pytest.raises(SystemExit):
        script.parse_args(["--tau", "1", "--seed", "0", "--epochs", "1"])
    args = script.parse_args(["--tau", "1", "--seed", "0", "--epochs", "1",
                              "--out-dir", "runs/timing"])
    assert args.epochs == 1


def test_epoch_override_keeps_warmup_inside_the_run():
    script = load_script()
    assert script.stage_for(None) == PRETRAIN
    short = script.stage_for(1)
    assert (short.epochs, short.warmup_epochs, short.peak_lr) == (1, 1, PRETRAIN.peak_lr)


def test_default_output_is_the_seed_directory(tmp_path):
    script = load_script()
    args = script.parse_args(["--tau", "8", "--seed", "2"])
    assert script.out_dir_for(args, tmp_path) == seed_dir(tmp_path, 8, 2)
