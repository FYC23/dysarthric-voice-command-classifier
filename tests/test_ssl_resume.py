"""Resuming an SSL seed at stage/fold granularity (src/training/ssl_finetune.py)."""

import shutil
from dataclasses import replace

import pandas as pd
import pytest
import torch

import src.training.ssl_checkpoints as ssl_checkpoints
import src.training.ssl_finetune as ssl_finetune
from src.eval.constants import DYSARTHRIC_SPEAKERS
from src.eval.cost import COST_FILE
from src.training.ssl_finetune import (
    CONTROLS_CHECKPOINT, EVAL_DIR, build_model, run_ssl_finetuning, state_dict_bytes,
    unit_seed,
)
from src.training.ssl_recipe import AdamWStage
from test_ssl_finetune import (
    FULL, fail_training, free_space, make_job, refuse_to_load_a_model, seed_with,
)

FOLDS = [f"fold{i}_{s}.pt" for i, s in enumerate(sorted(DYSARTHRIC_SPEAKERS), start=1)]
CHECKPOINTS = [CONTROLS_CHECKPOINT, *FOLDS]
TRAIN = ssl_finetune._train  # the real one, however often a test spies on it


def spy_training(monkeypatch, crash_on_call=None):
    """Speakers of every _train call; raises instead of making call number `crash_on_call`."""
    calls = []

    def spy(model, df, stages, *args):
        calls.append(frozenset(df["speaker_id"]))
        if len(calls) == crash_on_call:
            raise RuntimeError("crashed")
        return TRAIN(model, df, stages, *args)

    monkeypatch.setattr(ssl_finetune, "_train", spy)
    return calls


def crash_in_stage(monkeypatch, call):
    """Raise inside run_stage on its `call`-th call; returns the undo."""
    run_stage, calls = ssl_finetune.run_stage, []

    def spy(*args, **kwargs):
        calls.append(1)
        if len(calls) == call:
            raise RuntimeError("crashed")
        return run_stage(*args, **kwargs)

    monkeypatch.setattr(ssl_finetune, "run_stage", spy)
    return lambda: monkeypatch.setattr(ssl_finetune, "run_stage", run_stage)


def assert_same_seed(job, other, runs, other_runs):
    """Same predictions in both runs and bit-identical weights in every checkpoint."""
    for run, other_run in zip(runs, other_runs):
        pd.testing.assert_frame_equal(run.predictions, other_run.predictions)
        assert run.fold_train_speakers == other_run.fold_train_speakers
    for name in CHECKPOINTS:
        mine = torch.load(job.out_dir / name)["model_state_dict"]
        theirs = torch.load(other.out_dir / name)["model_state_dict"]
        assert mine.keys() == theirs.keys()
        assert all(torch.equal(mine[k], theirs[k]) for k in mine), name


def uninterrupted(small_torgo, tmp_path):
    job = make_job(tmp_path, runs_dir=tmp_path / "uninterrupted")
    return job, run_ssl_finetuning(job, small_torgo)


def test_unit_seeds_are_distinct_per_seed_and_unit():
    seeds = {unit_seed(seed, unit) for seed in range(3) for unit in range(9)}
    assert len(seeds) == 27
    assert unit_seed(0, 0) == 0 and unit_seed(2, 5) == 2005


def test_a_seed_that_crashed_in_a_fold_resumes_at_that_fold(small_torgo, tmp_path,
                                                            monkeypatch):
    job = make_job(tmp_path)
    spy_training(monkeypatch, crash_on_call=6)  # control stage, folds 1-4, then fold 5
    with pytest.raises(RuntimeError, match="crashed"):
        run_ssl_finetuning(job, small_torgo)
    assert {p.name for p in job.out_dir.glob("*.pt")} == {CONTROLS_CHECKPOINT, *FOLDS[:4]}
    calls = spy_training(monkeypatch)
    resumed = run_ssl_finetuning(replace(job, resume=True), small_torgo)
    held_out = sorted(DYSARTHRIC_SPEAKERS)[4:]
    assert calls == [frozenset(DYSARTHRIC_SPEAKERS) - {s} for s in held_out]
    other, straight = uninterrupted(small_torgo, tmp_path)
    assert_same_seed(job, other, resumed, straight)


def test_a_seed_that_crashed_in_the_control_stage_trains_everything(small_torgo, tmp_path,
                                                                     monkeypatch):
    job = make_job(tmp_path, resume=True)
    undo = crash_in_stage(monkeypatch, call=2)  # inside the control fine-tune
    with pytest.raises(RuntimeError, match="crashed"):
        run_ssl_finetuning(job, small_torgo)
    assert not list(job.out_dir.glob("*.pt"))
    undo()
    calls = spy_training(monkeypatch)
    resumed = run_ssl_finetuning(job, small_torgo)
    assert len(calls) == 9 and calls[0] == frozenset({"FC01"})
    other, straight = uninterrupted(small_torgo, tmp_path)
    assert_same_seed(job, other, resumed, straight)


@pytest.mark.parametrize("files", [[CONTROLS_CHECKPOINT], [FOLDS[2]],
                                   [CONTROLS_CHECKPOINT, *FOLDS[:3]]])
def test_a_partial_seed_is_not_overwritten_without_resume(small_torgo, tmp_path, monkeypatch,
                                                          files):
    job = make_job(tmp_path)
    seed_with(job, *files)
    refuse_to_load_a_model(monkeypatch)
    with pytest.raises(FileExistsError, match="--resume") as error:
        run_ssl_finetuning(job, small_torgo)
    assert "delete" in str(error.value) and str(job.out_dir) in str(error.value)
    assert sorted(p.name for p in job.out_dir.iterdir()) == sorted(files)
    assert not (job.runs_dir / "tiny-hubert" / COST_FILE).exists()


def finished_then_crashed(small_torgo, tmp_path, **overrides):
    """A finished seed whose LOSO run is removed, as if it crashed before writing it."""
    job = make_job(tmp_path, **overrides)
    run_ssl_finetuning(job, small_torgo)
    shutil.rmtree(job.out_dir / EVAL_DIR)
    return job


def test_a_fold_from_another_recipe_is_refused(small_torgo, tmp_path, monkeypatch):
    job = finished_then_crashed(small_torgo, tmp_path)
    refuse_to_load_a_model(monkeypatch)
    longer = AdamWStage(epochs=2, head_lr=1e-3, encoder_lr=1e-4, unfreeze=True, batch_size=2)
    with pytest.raises(ValueError, match=f"{FOLDS[0]}.*stages") as error:
        run_ssl_finetuning(make_job(tmp_path, dysarthric_stage=longer, resume=True), small_torgo)
    assert "train_speakers" not in str(error.value)
    assert (job.out_dir / FOLDS[0]).exists()


def test_a_control_checkpoint_from_other_speakers_is_refused(small_torgo, tmp_path,
                                                             monkeypatch):
    job = finished_then_crashed(small_torgo, tmp_path)
    path = job.out_dir / CONTROLS_CHECKPOINT
    torch.save({**torch.load(path), "train_speakers": ["FC01", "MC01"]}, path)
    refuse_to_load_a_model(monkeypatch)
    with pytest.raises(ValueError, match=f"{CONTROLS_CHECKPOINT}.*train_speakers"):
        run_ssl_finetuning(make_job(tmp_path, resume=True), small_torgo)


def test_a_truncated_checkpoint_is_named_and_kept(small_torgo, tmp_path, monkeypatch):
    job = finished_then_crashed(small_torgo, tmp_path)
    path = job.out_dir / FOLDS[2]
    path.write_bytes(path.read_bytes()[:1000])
    refuse_to_load_a_model(monkeypatch)
    with pytest.raises(RuntimeError, match=FOLDS[2]):
        run_ssl_finetuning(make_job(tmp_path, resume=True), small_torgo)
    assert path.stat().st_size == 1000


def test_a_checkpoint_from_another_device_and_workers_is_reused(small_torgo, tmp_path,
                                                                monkeypatch, capsys):
    job = finished_then_crashed(small_torgo, tmp_path)
    for name in CHECKPOINTS:
        path = job.out_dir / name
        torch.save({**torch.load(path), "device": "cuda:1", "num_workers": 4,
                    "attn_implementation": "sdpa"}, path)
    calls = spy_training(monkeypatch)
    loso, controls = run_ssl_finetuning(make_job(tmp_path, resume=True), small_torgo)
    assert calls == []
    assert set(loso.predictions.speaker_id) == set(DYSARTHRIC_SPEAKERS)
    assert "augmentation" in capsys.readouterr().out


def test_a_seed_whose_units_all_finished_only_predicts(small_torgo, tmp_path, monkeypatch):
    job = finished_then_crashed(small_torgo, tmp_path)
    stale = job.controls_out_dir / EVAL_DIR / "run.json"
    stale.write_text(stale.read_text().replace('"tiny-hubert-controls"', '"stale"'))
    calls = spy_training(monkeypatch)
    loso, controls = run_ssl_finetuning(make_job(tmp_path, resume=True), small_torgo)
    assert calls == []
    assert loso.model == "tiny-hubert" and controls.model == "tiny-hubert-controls"
    assert (job.runs_dir / "tiny-hubert" / COST_FILE).exists()


@pytest.mark.parametrize("resume", [False, True])
def test_a_leftover_temporary_checkpoint_is_not_done(small_torgo, tmp_path, monkeypatch,
                                                     capsys, resume):
    job = make_job(tmp_path, resume=resume)
    seed_with(job, f"{CONTROLS_CHECKPOINT}.tmp", f"{FOLDS[0]}.tmp")  # not loadable
    calls = spy_training(monkeypatch)
    run_ssl_finetuning(job, small_torgo)
    assert len(calls) == 9
    assert not list(job.out_dir.glob("*.tmp"))
    assert ("nothing to resume; starting fresh" in capsys.readouterr().out) == resume


def test_a_crash_while_saving_leaves_no_complete_looking_checkpoint(small_torgo, tmp_path,
                                                                    monkeypatch):
    def half_written(obj, path):
        path.write_bytes(b"half a checkpoint")
        raise OSError("disk full")

    monkeypatch.setattr(ssl_checkpoints.torch, "save", half_written)
    job = make_job(tmp_path)
    with pytest.raises(OSError, match="disk full"):
        run_ssl_finetuning(job, small_torgo)
    assert not (job.out_dir / CONTROLS_CHECKPOINT).exists()
    assert (job.out_dir / f"{CONTROLS_CHECKPOINT}.tmp").exists()


def test_the_disk_check_counts_only_checkpoints_still_to_write(small_torgo, tmp_path,
                                                               monkeypatch):
    job = finished_then_crashed(small_torgo, tmp_path)
    for name in FOLDS[4:]:
        (job.out_dir / name).unlink()
    needed = 4 * state_dict_bytes(build_model(job))  # folds 5-8
    resume = make_job(tmp_path, resume=True)
    free_space(monkeypatch, needed - 1)
    with pytest.raises(OSError, match="needs"):
        run_ssl_finetuning(resume, small_torgo)
    free_space(monkeypatch, needed)
    fail_training(monkeypatch)
    with pytest.raises(RuntimeError, match="training failed"):
        run_ssl_finetuning(resume, small_torgo)


def test_a_control_checkpoint_from_other_stages_is_refused(small_torgo, tmp_path, monkeypatch):
    job = finished_then_crashed(small_torgo, tmp_path)
    (job.out_dir / FOLDS[7]).unlink()
    calls = spy_training(monkeypatch)
    with pytest.raises(ValueError, match=f"{CONTROLS_CHECKPOINT}.*stages"):
        run_ssl_finetuning(make_job(tmp_path, resume=True, control_stages=(FULL,)), small_torgo)
    assert calls == []


def test_folds_without_the_controls_they_started_from_are_refused(small_torgo, tmp_path,
                                                                  monkeypatch):
    job = finished_then_crashed(small_torgo, tmp_path)
    (job.out_dir / CONTROLS_CHECKPOINT).unlink()
    refuse_to_load_a_model(monkeypatch)
    with pytest.raises(ValueError, match=f"not the {CONTROLS_CHECKPOINT}"):
        run_ssl_finetuning(make_job(tmp_path, resume=True), small_torgo)
    assert len(list(job.out_dir.glob("fold*.pt"))) == 8
