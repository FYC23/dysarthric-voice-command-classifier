"""scripts/finetune_ssl.py and scripts/train_ssl_all.sh."""

import importlib.util
import os
import shutil
import subprocess
from pathlib import Path

import pandas as pd
import pytest
import torch

from src.eval.constants import DYSARTHRIC_SPEAKERS
from src.training.loso import build_loso_run
from src.training.ssl_recipe import CONTROL_FINETUNE, CONTROL_HEAD_WARMUP, DYSARTHRIC_FINETUNE

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"


def load_script():
    spec = importlib.util.spec_from_file_location("finetune_ssl", SCRIPTS / "finetune_ssl.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def job_for(argv, tmp_path):
    script = load_script()
    return script.make_job(script.parse_args(argv), tmp_path / "runs", tmp_path / "noise",
                           tmp_path / "cache", torch.device("cpu"))


def test_seed_is_required():
    with pytest.raises(SystemExit):
        load_script().parse_args(["--backbone", "hubert-base"])


def test_unknown_backbone_is_rejected():
    with pytest.raises(SystemExit):
        load_script().parse_args(["--backbone", "wavlm", "--seed", "0"])


def test_real_run_uses_the_recipe_and_the_runs_directory(tmp_path):
    job = job_for(["--backbone", "hubert-large", "--seed", "2"], tmp_path)
    assert job.backbone.hf_id == "facebook/hubert-large-ll60k"
    assert job.out_dir == tmp_path / "runs" / "hubert-large" / "seed2"
    assert job.control_stages == (CONTROL_HEAD_WARMUP, CONTROL_FINETUNE)
    assert job.dysarthric_stage == DYSARTHRIC_FINETUNE
    assert job.cache_dir == tmp_path / "cache"


def test_smoke_run_is_one_epoch_per_stage_under_runs_smoke(tmp_path):
    job = job_for(["--backbone", "distilhubert", "--seed", "0", "--smoke"], tmp_path)
    assert job.out_dir == tmp_path / "runs" / "smoke" / "distilhubert" / "seed0"
    assert job.controls_out_dir == tmp_path / "runs" / "smoke" / "distilhubert-controls" / "seed0"
    assert [s.epochs for s in job.control_stages] == [1, 1]
    assert job.dysarthric_stage.epochs == 1
    assert job.dysarthric_stage.head_lr == DYSARTHRIC_FINETUNE.head_lr


def test_array_mic_summary_averages_speakers():
    rows = []
    for i, speaker in enumerate(DYSARTHRIC_SPEAKERS):
        for mic, pred in (("wav_arrayMic", "yes" if i < 4 else "no"), ("wav_headMic", "no")):
            rows.append({"speaker_id": speaker, "session": "Session1", "utterance_id": "0001",
                         "mic": mic, "label": "yes", "pred": pred})
    folds = {s: frozenset(set(DYSARTHRIC_SPEAKERS) - {s}) for s in DYSARTHRIC_SPEAKERS}
    run = build_loso_run([pd.DataFrame(rows)], folds, ["FC01"], 0, "hubert-base")
    text = load_script().array_mic_summary(run)
    assert "hubert-base, seed 0" in text
    assert "Array mic, speaker-averaged: 0.5000" in text


def run_all_script(tmp_path, **env):
    copy = tmp_path / "repo" / "scripts" / "train_ssl_all.sh"
    copy.parent.mkdir(parents=True)
    shutil.copy(SCRIPTS / "train_ssl_all.sh", copy)
    return subprocess.run(["bash", str(copy)], env={**os.environ, "DRY_RUN": "1", **env},
                          capture_output=True, text=True, check=True).stdout


def test_all_script_skips_only_finished_seeds_smallest_first(tmp_path):
    finished = tmp_path / "repo" / "runs" / "hubert-base" / "seed0" / "eval"
    finished.mkdir(parents=True)
    (finished / "run.json").write_text("{}")
    smoke = tmp_path / "repo" / "runs" / "smoke" / "distilhubert" / "seed0" / "eval"
    smoke.mkdir(parents=True)
    (smoke / "run.json").write_text("{}")  # a smoke run never counts as finished
    out = run_all_script(tmp_path, SEEDS="0")
    assert "[skip] hubert-base seed 0" in out
    assert "--backbone hubert-base" not in out
    runs = [line for line in out.splitlines() if "finetune_ssl.py" in line]
    assert [l.split("--backbone ")[1].split()[0] for l in runs] == ["distilhubert", "hubert-large"]


def test_all_script_passes_device_and_workers(tmp_path):
    out = run_all_script(tmp_path, BACKBONES="distilhubert", SEEDS="1", DEVICE="cuda:1",
                         NUM_WORKERS="4")
    assert "--backbone distilhubert --seed 1 --device cuda:1 --num-workers 4" in out
