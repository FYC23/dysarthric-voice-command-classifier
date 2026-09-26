"""BC-ResNet training recipe and loop (log-Mel + SpecAugment on the batch)."""

from dataclasses import replace

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.data.dataset import collate_fn
from src.model.bcresnet import HEAD_PREFIX, BCResNet
from src.model.frontend import LogMel, spec_augment_params
from src.training.bcresnet_loop import predict, run_stage
from src.training.bcresnet_recipe import (
    CONTROL_FINETUNE, CONTROL_HEAD_WARMUP, DYSARTHRIC_FINETUNE, PRETRAIN, WINDOW_FRAMES,
    WINDOW_SAMPLES, SgdStage, make_optimizer, make_scheduler, pick_device, run_name, seed_dir,
)

CPU = torch.device("cpu")


def toy_loader(n=8, batch=4, classes=3):
    g = torch.Generator().manual_seed(0)
    audio = torch.randn(n, WINDOW_SAMPLES, generator=g) * 0.1
    labels = torch.arange(n) % classes
    return DataLoader(TensorDataset(audio, labels), batch_size=batch, shuffle=False)


class TestRecipe:
    def test_pretraining_is_the_papers(self):
        assert (PRETRAIN.epochs, PRETRAIN.peak_lr, PRETRAIN.warmup_epochs, PRETRAIN.batch_size,
                PRETRAIN.momentum, PRETRAIN.weight_decay) == (200, 0.1, 5, 100, 0.9, 1e-3)
        assert PRETRAIN.cosine and not PRETRAIN.class_weighted and not PRETRAIN.head_only

    def test_torgo_stages_are_the_fixed_settings(self):
        assert CONTROL_HEAD_WARMUP.head_only and not CONTROL_HEAD_WARMUP.cosine
        assert (CONTROL_HEAD_WARMUP.epochs, CONTROL_HEAD_WARMUP.peak_lr) == (5, 0.01)
        assert (CONTROL_FINETUNE.epochs, CONTROL_FINETUNE.peak_lr,
                CONTROL_FINETUNE.warmup_epochs) == (40, 0.01, 2)
        assert (DYSARTHRIC_FINETUNE.epochs, DYSARTHRIC_FINETUNE.peak_lr,
                DYSARTHRIC_FINETUNE.warmup_epochs) == (30, 0.003, 2)
        for stage in (CONTROL_HEAD_WARMUP, CONTROL_FINETUNE, DYSARTHRIC_FINETUNE):
            assert stage.batch_size == 32 and stage.class_weighted and stage.drop_last
            assert (stage.momentum, stage.weight_decay) == (0.9, 1e-3)

    def test_window_is_201_frames(self):
        assert WINDOW_SAMPLES == 32000 and WINDOW_FRAMES == 201

    @pytest.mark.parametrize("change", [dict(epochs=0), dict(warmup_epochs=201),
                                        dict(peak_lr=0.0), dict(batch_size=0)])
    def test_invalid_stage_raises(self, change):
        with pytest.raises(ValueError):
            replace(PRETRAIN, **change)

    def test_names_and_directories(self, tmp_path):
        assert run_name(8.0) == "bcresnet-8" and run_name(1) == "bcresnet-1"
        assert seed_dir(tmp_path, 2, 3) == tmp_path / "bcresnet-2" / "seed3"

    def test_pick_device_honours_an_explicit_choice(self):
        assert pick_device("cpu") == CPU

    def test_scheduler_needs_at_least_one_batch(self):
        opt = make_optimizer(BCResNet(1, 3), CONTROL_FINETUNE)
        with pytest.raises(ValueError, match="batch"):
            make_scheduler(opt, CONTROL_FINETUNE, steps_per_epoch=0)

    def test_head_only_optimizer_holds_only_the_head(self):
        model = BCResNet(1, 3)
        params = make_optimizer(model, CONTROL_HEAD_WARMUP).param_groups[0]["params"]
        assert {id(p) for p in params} == {id(p) for p in model.head.parameters()}


class TestLoop:
    def test_training_lowers_the_loss_on_a_toy_set(self):
        torch.manual_seed(0)
        model = BCResNet(1, 3)
        stage = SgdStage(epochs=15, peak_lr=0.05, warmup_epochs=0, batch_size=4)
        history = run_stage(model, stage, toy_loader(), LogMel(), None, CPU,
                            torch.Generator().manual_seed(0))
        assert [h["epoch"] for h in history] == list(range(1, 16))
        assert history[-1]["loss"] < history[0]["loss"]

    def test_head_warmup_trains_only_head_weights_while_batch_norm_adapts(self):
        torch.manual_seed(0)
        model = BCResNet(1, 3)
        weights_before = {k: v.detach().clone() for k, v in model.named_parameters()}
        stats_before = {k: v.clone() for k, v in model.named_buffers()
                        if k.endswith("running_mean")}
        stage = SgdStage(epochs=2, peak_lr=0.1, warmup_epochs=0, batch_size=4,
                         head_only=True, cosine=False)
        run_stage(model, stage, toy_loader(), LogMel(), spec_augment_params(2, 2.0), CPU,
                  torch.Generator().manual_seed(0))
        changed = {k for k, v in model.named_parameters()
                   if not torch.equal(weights_before[k], v)}
        assert changed == {HEAD_PREFIX + "weight", HEAD_PREFIX + "bias"}
        stats_after = dict(model.named_buffers())
        assert all(not torch.equal(stats_before[k], stats_after[k]) for k in stats_before)
        assert all(p.requires_grad for p in model.parameters())  # restored afterwards

    def test_class_weights_change_the_update(self):
        stage = SgdStage(epochs=1, peak_lr=0.05, warmup_epochs=0, batch_size=4)

        def trained(weights):
            torch.manual_seed(0)
            model = BCResNet(1, 3)
            run_stage(model, stage, toy_loader(), LogMel(), None, CPU,
                      torch.Generator().manual_seed(0), class_weights=weights)
            return model.head.weight.detach().clone()

        assert not torch.equal(trained(None), trained(torch.tensor([5.0, 1.0, 1.0])))

    def test_predict_returns_labels_in_loader_order(self):
        preds, labels = predict(BCResNet(1, 3), toy_loader(), LogMel(), CPU)
        assert labels.tolist() == [0, 1, 2, 0, 1, 2, 0, 1]
        assert preds.shape == (8,) and preds.dtype == np.int64

    def test_predict_accepts_torgo_dict_batches(self):
        items = [{"input_values": torch.zeros(WINDOW_SAMPLES), "label": torch.tensor(i % 3),
                  "speaker_id": "F01", "file_path": f"/x/{i}.wav"} for i in range(5)]
        loader = DataLoader(items, batch_size=2, shuffle=False, collate_fn=collate_fn)
        _, labels = predict(BCResNet(1, 3), loader, LogMel(), CPU)
        assert labels.tolist() == [0, 1, 2, 0, 1]
