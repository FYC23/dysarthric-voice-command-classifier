"""SSL per-seed flow: control stage, controls-only run, LOSO folds, cost."""

import shutil
from dataclasses import asdict, replace

import numpy as np
import pytest
import torch

import src.training.ssl_finetune as ssl_finetune
from conftest import SR, write_wav
from ssl_factories import tiny_spec
from src.eval.constants import DYSARTHRIC_SPEAKERS
from src.eval.cost import COST_FILE, count_params, load_cost
from src.eval.io import load_run
from src.model.backbones import MASKING_OVERRIDES
from src.training.loso import TORGO_CLASSES
from src.training.ssl_finetune import (
    CONTROLS_CHECKPOINT, EVAL_DIR, SslJob, build_model, run_ssl_finetuning, ssl_cost,
)
from src.training.ssl_recipe import AdamWStage

HEAD = AdamWStage(epochs=1, head_lr=1e-3, encoder_lr=None, unfreeze=False, batch_size=2)
FULL = AdamWStage(epochs=1, head_lr=1e-3, encoder_lr=1e-4, unfreeze=True, batch_size=2)
DYSARTHRIC = set(DYSARTHRIC_SPEAKERS)


def make_job(tmp_path, **overrides):
    noise = tmp_path / "noise"
    write_wav(noise / "n.wav", np.random.default_rng(0).normal(0, 0.05, 3 * SR))
    base = dict(backbone=tiny_spec(tmp_path, layers=2), seed=0, runs_dir=tmp_path / "runs",
                noise_dir=noise, device=torch.device("cpu"), control_stages=(HEAD, FULL),
                dysarthric_stage=FULL, num_workers=0)
    return SslJob(**{**base, **overrides})


def test_job_directories(tmp_path):
    job = make_job(tmp_path)
    assert job.out_dir == tmp_path / "runs" / "tiny-hubert" / "seed0"
    assert job.controls_out_dir == tmp_path / "runs" / "tiny-hubert-controls" / "seed0"


def test_a_seed_writes_checkpoints_both_runs_and_cost(small_torgo, tmp_path):
    job = make_job(tmp_path)
    loso, controls = run_ssl_finetuning(job, small_torgo)
    names = {p.name for p in job.out_dir.iterdir()}
    assert {CONTROLS_CHECKPOINT, EVAL_DIR} <= names
    assert sum(n.startswith("fold") for n in names) == 8
    assert loso.model == "tiny-hubert" and loso.seed == 0
    assert set(loso.predictions.speaker_id) == DYSARTHRIC
    assert loso.fold_train_speakers["M04"] == (DYSARTHRIC - {"M04"}) | {"FC01"}
    assert load_run(job.out_dir / EVAL_DIR).model == "tiny-hubert"
    assert controls.model == "tiny-hubert-controls"
    assert set(controls.predictions.speaker_id) == DYSARTHRIC  # no control speakers scored
    assert all(t == {"FC01"} for t in controls.fold_train_speakers.values())
    assert load_run(job.controls_out_dir / EVAL_DIR).model == "tiny-hubert-controls"
    for name in ("tiny-hubert", "tiny-hubert-controls"):
        cost = load_cost(job.runs_dir / name / COST_FILE)
        assert cost.input_seconds == 2.0 and cost.params == count_params(build_model(job))


def test_checkpoints_record_what_produced_them(small_torgo, tmp_path):
    job = make_job(tmp_path)
    run_ssl_finetuning(job, small_torgo)
    controls = torch.load(job.out_dir / CONTROLS_CHECKPOINT)
    assert controls["stages"] == [asdict(s) for s in (HEAD, FULL)]
    assert controls["train_speakers"] == ["FC01"]
    fold = torch.load(job.out_dir / "fold7_M04.pt")
    assert fold["stages"] == [asdict(s) for s in (HEAD, FULL, FULL)]
    assert "M04" not in fold["train_speakers"] and "FC01" in fold["train_speakers"]
    assert fold["backbone"] == "tiny-hubert" and fold["hf_id"] == job.backbone.hf_id
    assert fold["classes"] == list(TORGO_CLASSES) and fold["seed"] == 0
    assert fold["masking"] == dict(MASKING_OVERRIDES)


def test_control_stage_and_folds_train_on_the_right_speakers(small_torgo, tmp_path,
                                                             monkeypatch):
    calls, loads = [], []
    train, load = ssl_finetune._train, ssl_finetune._load

    def spy_train(model, df, stages, *args):
        calls.append((frozenset(df["speaker_id"]), tuple(stages)))
        return train(model, df, stages, *args)

    def spy_load(job, path):
        loads.append(path.name)
        return load(job, path)

    monkeypatch.setattr(ssl_finetune, "_train", spy_train)
    monkeypatch.setattr(ssl_finetune, "_load", spy_load)
    job = make_job(tmp_path)
    run_ssl_finetuning(job, small_torgo)
    assert calls[0] == (frozenset({"FC01"}), job.control_stages)
    folds = [speakers for speakers, stages in calls[1:]]
    assert len(folds) == 8  # no model on all 8 dysarthric speakers
    for held_out, speakers in zip(sorted(DYSARTHRIC), folds):
        assert speakers == DYSARTHRIC - {held_out}
    assert loads == [CONTROLS_CHECKPOINT] * 8  # every fold starts from the control stage


def test_a_finished_seed_is_never_overwritten(small_torgo, tmp_path, monkeypatch):
    job = make_job(tmp_path)
    (job.out_dir / EVAL_DIR).mkdir(parents=True)
    (job.out_dir / EVAL_DIR / "run.json").write_text("{}")
    monkeypatch.setattr(ssl_finetune, "build_model",
                        lambda job: pytest.fail("loaded a model for a finished seed"))
    with pytest.raises(FileExistsError, match="finished"):
        run_ssl_finetuning(job, small_torgo)


def test_a_crashed_seed_runs_again(small_torgo, tmp_path):
    job = make_job(tmp_path)
    run_ssl_finetuning(job, small_torgo)
    shutil.rmtree(job.out_dir / EVAL_DIR)  # as if it crashed during the folds
    stale = job.controls_out_dir / EVAL_DIR / "run.json"
    stale.write_text(stale.read_text().replace('"tiny-hubert-controls"', '"stale"'))
    loso, controls = run_ssl_finetuning(job, small_torgo)
    assert loso.model == "tiny-hubert" and controls.model == "tiny-hubert-controls"


def test_missing_noise_fails_before_anything_is_written(small_torgo, tmp_path):
    job = make_job(tmp_path, noise_dir=tmp_path / "no-noise")
    with pytest.raises(FileNotFoundError, match="download_speech_commands"):
        run_ssl_finetuning(job, small_torgo)
    assert not (job.runs_dir).exists()


def test_cost_is_written_before_training(small_torgo, tmp_path, monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("training failed")

    monkeypatch.setattr(ssl_finetune, "run_stage", boom)
    job = make_job(tmp_path)
    with pytest.raises(RuntimeError, match="training failed"):
        run_ssl_finetuning(job, small_torgo)
    assert (job.runs_dir / "tiny-hubert" / COST_FILE).exists()
    assert (job.runs_dir / "tiny-hubert-controls" / COST_FILE).exists()


def test_cost_counts_the_whole_model_on_one_window(tmp_path):
    model = build_model(make_job(tmp_path))
    cost = ssl_cost(model)
    assert cost.params == count_params(model) and cost.macs > 0 and cost.input_seconds == 2.0


def test_evaluation_is_never_augmented(small_torgo, tmp_path, monkeypatch):
    built = []
    real = ssl_finetune.TORGOCommandDataset

    def spy(df, feature_extractor, config, **kwargs):
        built.append(kwargs)
        return real(df, feature_extractor, config, **kwargs)

    monkeypatch.setattr(ssl_finetune, "TORGOCommandDataset", spy)
    run_ssl_finetuning(make_job(tmp_path), small_torgo)
    eval_sets = [k for k in built if not k["augment"]]
    assert len(eval_sets) == 9  # controls-only run + 8 folds
    assert all(k.get("noise") is None for k in eval_sets)
