"""The fixed SSL recipe: stage settings, optimizer groups, run names."""

from pathlib import Path

import pytest
import torch

from ssl_factories import tiny_spec
from src.model.architecture import SSLCommandClassifier, set_trainable
from src.model.backbones import load_backbone
from src.training.ssl_recipe import (
    CONTROL_FINETUNE, CONTROL_HEAD_WARMUP, DYSARTHRIC_FINETUNE, WINDOW_SAMPLES, AdamWStage,
    controls_run_name, make_optimizer, make_scheduler, param_groups, run_name, seed_dir,
    stage_top_n,
)


def make_model(tmp_path, layers=6):
    return SSLCommandClassifier(load_backbone(tiny_spec(tmp_path, layers)), num_labels=20)


def test_stages_are_the_agreed_recipe():
    assert (CONTROL_HEAD_WARMUP.epochs, CONTROL_HEAD_WARMUP.head_lr,
            CONTROL_HEAD_WARMUP.encoder_lr, CONTROL_HEAD_WARMUP.unfreeze) == (5, 1e-4, None, False)
    assert (CONTROL_FINETUNE.epochs, CONTROL_FINETUNE.head_lr,
            CONTROL_FINETUNE.encoder_lr, CONTROL_FINETUNE.unfreeze) == (10, 1e-5, 1e-6, True)
    assert (DYSARTHRIC_FINETUNE.epochs, DYSARTHRIC_FINETUNE.head_lr,
            DYSARTHRIC_FINETUNE.encoder_lr, DYSARTHRIC_FINETUNE.unfreeze) == (15, 5e-5, 5e-6, True)
    for stage in (CONTROL_HEAD_WARMUP, CONTROL_FINETUNE, DYSARTHRIC_FINETUNE):
        assert (stage.batch_size, stage.weight_decay, stage.max_grad_norm,
                stage.class_weighted) == (8, 0.01, 1.0, True)
    assert WINDOW_SAMPLES == 32000


@pytest.mark.parametrize("kwargs", [
    dict(epochs=1, head_lr=1e-3, encoder_lr=None, unfreeze=True),
    dict(epochs=1, head_lr=1e-3, encoder_lr=1e-4, unfreeze=False),
    dict(epochs=0, head_lr=1e-3, encoder_lr=None, unfreeze=False),
    dict(epochs=1, head_lr=0.0, encoder_lr=None, unfreeze=False),
    dict(epochs=1, head_lr=1e-3, encoder_lr=None, unfreeze=False, batch_size=0),
])
def test_inconsistent_stages_are_rejected(kwargs):
    with pytest.raises(ValueError):
        AdamWStage(**kwargs)


def test_stage_top_n(tmp_path):
    model = make_model(tmp_path, layers=6)
    assert stage_top_n(model, CONTROL_HEAD_WARMUP) == 0
    assert stage_top_n(model, CONTROL_FINETUNE) == 4


def test_param_groups_cover_every_trainable_parameter_once(tmp_path):
    model = make_model(tmp_path)
    set_trainable(model, 4)
    groups = param_groups(model, DYSARTHRIC_FINETUNE)
    assert [g["lr"] for g in groups] == [5e-5, 5e-6]
    ids = [id(p) for g in groups for p in g["params"]]
    assert len(ids) == len(set(ids))
    assert set(ids) == {id(p) for p in model.parameters() if p.requires_grad}


def test_head_group_includes_pooling_and_layer_weights(tmp_path):
    model = make_model(tmp_path)
    set_trainable(model, 4)
    head_ids = {id(p) for p in param_groups(model, CONTROL_FINETUNE)[0]["params"]}
    for module in (model.head.layer_weights, model.head.attention_pooling, model.head.classifier):
        assert {id(p) for p in module.parameters()} <= head_ids  # the bug-1 case


def test_head_only_stage_has_one_group(tmp_path):
    model = make_model(tmp_path)
    set_trainable(model, 0)
    groups = param_groups(model, CONTROL_HEAD_WARMUP)
    assert len(groups) == 1 and groups[0]["lr"] == 1e-4


def test_head_only_stage_refuses_a_trainable_backbone(tmp_path):
    model = make_model(tmp_path)
    set_trainable(model, 4)
    with pytest.raises(ValueError, match="head-only"):
        param_groups(model, CONTROL_HEAD_WARMUP)


def test_unfreeze_stage_refuses_a_frozen_backbone(tmp_path):
    model = make_model(tmp_path)
    set_trainable(model, 0)
    with pytest.raises(ValueError, match="no trainable backbone"):
        param_groups(model, CONTROL_FINETUNE)


def test_optimizer_and_schedule(tmp_path):
    model = make_model(tmp_path)
    set_trainable(model, 4)
    optimizer = make_optimizer(model, CONTROL_FINETUNE)
    assert isinstance(optimizer, torch.optim.AdamW)
    assert all(g["weight_decay"] == 0.01 for g in optimizer.param_groups)
    scheduler = make_scheduler(optimizer, CONTROL_FINETUNE, steps_per_epoch=3)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(1e-5)  # no warmup
    for _ in range(CONTROL_FINETUNE.epochs * 3):
        optimizer.step()
        scheduler.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.0)


def test_scheduler_needs_at_least_one_step(tmp_path):
    model = make_model(tmp_path)
    set_trainable(model, 0)
    with pytest.raises(ValueError, match="smaller than one batch"):
        make_scheduler(make_optimizer(model, CONTROL_HEAD_WARMUP), CONTROL_HEAD_WARMUP, 0)


def test_run_names_and_directories():
    assert run_name("hubert-base") == "hubert-base"
    assert controls_run_name("hubert-base") == "hubert-base-controls"
    assert seed_dir(Path("/r"), "distilhubert", 2) == Path("/r/distilhubert/seed2")
