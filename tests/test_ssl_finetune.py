"""SSL per-seed flow: control stage, controls-only run, LOSO folds, cost."""

import gc
import shutil
import weakref
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

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
    state_dict_bytes,
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
    # the disk check's estimate, controls.pt plus one checkpoint per fold, is a floor; the
    # file overhead per tensor is fixed, so large next to this tiny model's tensors only
    written = sum(p.stat().st_size for p in job.out_dir.glob("*.pt"))
    estimate = 9 * state_dict_bytes(build_model(job))
    assert estimate <= written < 1.25 * estimate


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
    used = build_model(job).backbone.config._attn_implementation
    assert controls["attn_implementation"] == fold["attn_implementation"] == used
    assert used is not None
    assert fold["num_workers"] == controls["num_workers"] == 0
    assert fold["device"] == controls["device"] == "cpu"


@pytest.mark.parametrize("device, expected", [("mps", "eager"), ("cpu", None)])
def test_build_model_picks_attention_for_the_jobs_device(tmp_path, monkeypatch,
                                                         device, expected):
    asked = []
    real = ssl_finetune.load_backbone

    def spy(spec, cache_dir=None, attn_implementation=None):
        asked.append(attn_implementation)
        return real(spec, cache_dir, attn_implementation=attn_implementation)

    monkeypatch.setattr(ssl_finetune, "load_backbone", spy)
    model = build_model(make_job(tmp_path, device=torch.device(device)))  # stays on the CPU
    assert asked == [expected]
    if expected is not None:
        assert model.backbone.config._attn_implementation == expected


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


def seed_with(job, *files):
    """A seed directory holding `files` (paths relative to it)."""
    for name in files:
        (job.out_dir / name).parent.mkdir(parents=True, exist_ok=True)
        (job.out_dir / name).write_text("{}")


def refuse_to_load_a_model(monkeypatch):
    monkeypatch.setattr(ssl_finetune, "build_model",
                        lambda job: pytest.fail("loaded a model for a seed it must not train"))


@pytest.mark.parametrize("missing, message", [("M04", "dysarthric speakers"),
                                              ("FC01", "no control speakers")])
def test_a_seed_needs_all_8_dysarthric_speakers_and_a_control(small_torgo, tmp_path,
                                                              monkeypatch, missing, message):
    job = make_job(tmp_path, noise_dir=tmp_path / "no-noise")  # noise would fail if loaded
    refuse_to_load_a_model(monkeypatch)
    monkeypatch.setattr(ssl_finetune, "load_feature_extractor",
                        lambda *a: pytest.fail("loaded a feature extractor"))
    with pytest.raises(ValueError, match=message):
        run_ssl_finetuning(job, small_torgo[small_torgo["speaker_id"] != missing])
    assert not job.runs_dir.exists()


def test_each_fold_frees_the_previous_model_before_loading(small_torgo, tmp_path, monkeypatch):
    built, alive_at_load = [], []
    build, load = ssl_finetune.build_model, ssl_finetune._load

    def spy_build(job):
        model = build(job)
        built.append(weakref.ref(model))
        return model

    def spy_load(job, path):
        gc.collect()
        alive_at_load.append(sum(ref() is not None for ref in built))
        return load(job, path)

    monkeypatch.setattr(ssl_finetune, "build_model", spy_build)
    monkeypatch.setattr(ssl_finetune, "_load", spy_load)
    run_ssl_finetuning(make_job(tmp_path), small_torgo)
    assert alive_at_load == [0] * 8  # peak memory: one model


def test_a_finished_seed_is_never_overwritten(small_torgo, tmp_path, monkeypatch):
    job = make_job(tmp_path)
    seed_with(job, f"{EVAL_DIR}/run.json", CONTROLS_CHECKPOINT)
    refuse_to_load_a_model(monkeypatch)
    with pytest.raises(FileExistsError, match="finished"):
        run_ssl_finetuning(job, small_torgo)


def test_a_run_left_by_the_removed_train_script_is_refused(small_torgo, tmp_path, monkeypatch):
    # scripts/train.py wrote runs/hubert-large/seed<k>/eval/run.json but no controls.pt
    job = make_job(tmp_path)
    seed_with(job, f"{EVAL_DIR}/run.json")
    refuse_to_load_a_model(monkeypatch)
    with pytest.raises(FileExistsError, match="scripts/train.py") as error:
        run_ssl_finetuning(job, small_torgo)
    assert "finished" not in str(error.value) and str(job.out_dir) in str(error.value)
    assert sorted(p.name for p in job.out_dir.rglob("*")) == [EVAL_DIR, "run.json"]


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


def free_space(monkeypatch, free):
    """Pretend the disk has `free` bytes; returns the paths asked about."""
    asked = []

    def disk_usage(path):
        asked.append(Path(path))
        return SimpleNamespace(total=free, used=0, free=free)

    monkeypatch.setattr(ssl_finetune.shutil, "disk_usage", disk_usage)
    return asked


def fail_training(monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("training failed")

    monkeypatch.setattr(ssl_finetune, "run_stage", boom)


def test_a_seed_that_cannot_fit_on_disk_fails_before_anything_is_written(small_torgo, tmp_path,
                                                                        monkeypatch):
    job = make_job(tmp_path)
    needed = 9 * state_dict_bytes(build_model(job))  # controls.pt + 8 folds
    asked = free_space(monkeypatch, needed - 1)
    with pytest.raises(OSError, match=r"needs \d+\.\d GB .* \d+\.\d GB free"):
        run_ssl_finetuning(job, small_torgo)
    assert asked == [tmp_path]  # runs/ does not exist yet: its nearest existing parent
    assert not job.runs_dir.exists()


def test_a_seed_that_fits_on_disk_goes_on_to_train(small_torgo, tmp_path, monkeypatch):
    job = make_job(tmp_path)
    free_space(monkeypatch, 9 * state_dict_bytes(build_model(job)))
    fail_training(monkeypatch)
    with pytest.raises(RuntimeError, match="training failed"):
        run_ssl_finetuning(job, small_torgo)


def test_state_dict_bytes_counts_every_saved_tensor():
    model = torch.nn.Sequential(torch.nn.Linear(3, 2), torch.nn.BatchNorm1d(2))
    # weight 6 + bias 2 + bn weight, bias, running mean, running var 2 each: float32;
    # num_batches_tracked: one int64
    assert state_dict_bytes(model) == (6 + 2 + 4 * 2) * 4 + 8


def test_cost_is_written_before_training(small_torgo, tmp_path, monkeypatch):
    fail_training(monkeypatch)
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
