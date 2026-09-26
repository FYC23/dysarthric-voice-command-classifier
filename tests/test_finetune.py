"""Stages 2-3: TORGO control stage, dysarthric LOSO folds, deploy model, eval run."""

import importlib.util
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pytest
import torch

from conftest import SR, add_utterance, word_clip, write_wav
from src.config import Config
from src.data.preprocessing import scan_torgo_dataset
from src.data.speech_commands import CLASSES
from src.eval.constants import DYSARTHRIC_SPEAKERS
from src.eval.cost import count_params
from src.eval.io import load_run
from src.model.bcresnet import BCResNet
from src.training.bcresnet_recipe import SgdStage, seed_dir
from src.training.finetune import (
    CONTROLS_CHECKPOINT, DEPLOY_CHECKPOINT, EVAL_DIR, TORGO_CLASSES, FinetuneJob,
    bcresnet_cost, run_finetuning,
)

GROUPS = {"F01": "F", "F03": "F", "F04": "F", "M01": "M", "M02": "M", "M03": "M",
          "M04": "M", "M05": "M", "FC01": "FC"}
WORDS = ("yes", "no", "menu")
HEAD = SgdStage(epochs=1, peak_lr=0.01, warmup_epochs=0, batch_size=2, head_only=True,
                cosine=False, class_weighted=True, drop_last=True)
FULL = SgdStage(epochs=1, peak_lr=0.01, warmup_epochs=0, batch_size=2, class_weighted=True,
                drop_last=True)
SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "finetune_bcresnet.py"


@pytest.fixture
def small_torgo(tmp_path):
    """8 dysarthric speakers and one control, three array-mic words each."""
    root = tmp_path / "TORGO"
    for i, (speaker, group) in enumerate(GROUPS.items()):
        for j, word in enumerate(WORDS):
            add_utterance(root, group, speaker, "Session1", f"{j + 1:04d}", word,
                          {"wav_arrayMic": word_clip(seed=10 * i + j)})
    samples = scan_torgo_dataset(root, Config.TARGET_COMMANDS, Config.MIC_TYPES,
                                 Config.MIN_AUDIO_DURATION, Config.MAX_AUDIO_DURATION)
    return samples.assign(seg_start=np.nan, seg_end=np.nan)


def pretrained_checkpoint(tmp_path, tau=1):
    path = tmp_path / f"pretrain_tau{tau}.pt"
    torch.save({"model_state_dict": BCResNet(tau, len(CLASSES)).state_dict(),
                "tau": float(tau), "classes": list(CLASSES)}, path)
    return path


def make_job(tmp_path, **overrides):
    noise = tmp_path / "noise"
    write_wav(noise / "n.wav", np.random.default_rng(0).normal(0, 0.05, 3 * SR))
    base = dict(tau=1, seed=0, out_dir=tmp_path / "seed0",
                pretrained=pretrained_checkpoint(tmp_path), noise_dir=noise,
                device=torch.device("cpu"), control_stages=(HEAD, FULL),
                dysarthric_stage=FULL, num_workers=0)
    return FinetuneJob(**{**base, **overrides})


def test_torgo_classes_are_the_20_commands_whatever_the_data():
    assert TORGO_CLASSES == tuple(sorted(Config.TARGET_COMMANDS)) and len(TORGO_CLASSES) == 20


def test_finetuning_writes_every_checkpoint_and_a_valid_run(small_torgo, tmp_path):
    run = run_finetuning(make_job(tmp_path), small_torgo)
    out = tmp_path / "seed0"
    names = {p.name for p in out.iterdir()}
    assert {CONTROLS_CHECKPOINT, DEPLOY_CHECKPOINT, EVAL_DIR} <= names
    assert sum(name.startswith("fold") for name in names) == 8
    assert run.model == "bcresnet-1" and run.seed == 0
    assert set(run.predictions.speaker_id) == set(DYSARTHRIC_SPEAKERS)
    assert run.fold_train_speakers["M04"] == (set(DYSARTHRIC_SPEAKERS) - {"M04"}) | {"FC01"}
    assert load_run(out / EVAL_DIR).model == "bcresnet-1"
    deploy = torch.load(out / DEPLOY_CHECKPOINT)
    assert deploy["classes"] == list(TORGO_CLASSES) and deploy["tau"] == 1.0
    assert deploy["stages"] == [asdict(s) for s in (HEAD, FULL, FULL)]
    assert deploy["pretrained"] == str(tmp_path / "pretrain_tau1.pt")
    controls = torch.load(out / CONTROLS_CHECKPOINT)
    assert controls["stages"] == [asdict(s) for s in (HEAD, FULL)]
    fold = torch.load(out / "fold7_M04.pt")
    assert fold["stages"] == [asdict(s) for s in (HEAD, FULL, FULL)]


def test_pretrained_checkpoint_of_another_width_is_refused(small_torgo, tmp_path):
    job = replace(make_job(tmp_path), pretrained=pretrained_checkpoint(tmp_path, tau=2))
    with pytest.raises(ValueError, match="tau"):
        run_finetuning(job, small_torgo)


def test_missing_pretrained_checkpoint_says_what_to_run(small_torgo, tmp_path):
    job = replace(make_job(tmp_path), pretrained=tmp_path / "missing.pt")
    with pytest.raises(FileNotFoundError, match="pretrain_bcresnet"):
        run_finetuning(job, small_torgo)


def test_cost_profile_is_at_the_2s_window():
    cost = bcresnet_cost(1)
    assert cost.input_seconds == 2.0
    assert cost.params == count_params(BCResNet(1, 20))


def test_script_defaults_to_this_seeds_pretraining(tmp_path):
    spec = importlib.util.spec_from_file_location("finetune_script", SCRIPT)
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    args = script.parse_args(["--tau", "3", "--seed", "1"])
    out_dir, pretrained = script.paths_for(args, tmp_path)
    assert out_dir == seed_dir(tmp_path, 3, 1)
    assert pretrained == seed_dir(tmp_path, 3, 1) / "pretrain.pt"
