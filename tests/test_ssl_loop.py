"""SSL stage loop: what each stage trains, last epoch kept, deterministic prediction."""

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

import src.training.ssl_loop as ssl_loop
from ssl_factories import tiny_spec
from src.data.dataset import collate_fn
from src.model.architecture import SSLCommandClassifier
from src.model.backbones import load_backbone
from src.training.ssl_loop import predict, run_stage
from src.training.ssl_recipe import WINDOW_SAMPLES, AdamWStage

CPU = torch.device("cpu")
HEAD = AdamWStage(epochs=1, head_lr=1e-2, encoder_lr=None, unfreeze=False, batch_size=2)
FULL = AdamWStage(epochs=1, head_lr=1e-2, encoder_lr=1e-2, unfreeze=True, batch_size=2)


def make_model(tmp_path, layers=6):
    torch.manual_seed(0)
    return SSLCommandClassifier(load_backbone(tiny_spec(tmp_path, layers)), num_labels=20)


def loader(n=4, batch_size=2):
    g = torch.Generator().manual_seed(0)
    items = [{"input_values": torch.randn(WINDOW_SAMPLES, generator=g),
              "label": torch.tensor(i % 2), "speaker_id": "F01", "file_path": f"{i}.wav"}
             for i in range(n)]
    return DataLoader(items, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)


def snapshot(module):
    return {n: p.detach().clone() for n, p in module.named_parameters()}


def changed(before, module):
    return {n for n, p in module.named_parameters() if not torch.equal(before[n], p)}


def test_head_only_stage_leaves_the_backbone_untouched(tmp_path):
    model = make_model(tmp_path)
    backbone, head = snapshot(model.backbone), snapshot(model.head)
    history = run_stage(model, HEAD, loader(), CPU)
    assert [h["epoch"] for h in history] == [1]
    assert changed(backbone, model.backbone) == set()
    assert changed(head, model.head) >= {"layer_weights.weights", "classifier.0.weight"}
    assert any(n.startswith("attention_pooling.") for n in changed(head, model.head))


def test_unfreeze_stage_trains_only_the_top_4_layers(tmp_path):
    model = make_model(tmp_path, layers=6)
    before = snapshot(model.backbone)
    run_stage(model, FULL, loader(), CPU)
    moved = changed(before, model.backbone)
    assert not any(n.startswith(("encoder.layers.0.", "encoder.layers.1.",
                                 "feature_extractor.", "encoder.pos_conv_embed."))
                   for n in moved)
    assert any(n.startswith("encoder.layers.5.") for n in moved)
    assert any(n.startswith("encoder.layers.2.") for n in moved)


def test_the_last_epoch_is_kept(tmp_path, monkeypatch):
    model = make_model(tmp_path, layers=2)
    epochs_seen = []

    def fake_epoch(model, *args, **kwargs):
        epochs_seen.append(len(epochs_seen) + 1)
        with torch.no_grad():
            model.head.classifier[-1].bias.fill_(float(epochs_seen[-1]))
        # every epoch is worse than the last (loss up, accuracy down), so a
        # best-accuracy or lowest-loss selector would keep epoch 1
        return float(epochs_seen[-1]), 1.0 / epochs_seen[-1]

    monkeypatch.setattr(ssl_loop, "train_epoch", fake_epoch)
    three = AdamWStage(epochs=3, head_lr=1e-3, encoder_lr=None, unfreeze=False, batch_size=2)
    history = run_stage(model, three, loader(), CPU)
    assert [h["epoch"] for h in history] == [1, 2, 3]
    assert torch.all(model.head.classifier[-1].bias == 3.0)


def test_predict_is_deterministic_and_in_loader_order(tmp_path):
    model = make_model(tmp_path, layers=2)
    preds1, labels1 = predict(model, loader(n=5), CPU)
    preds2, _ = predict(model, loader(n=5), CPU)
    assert labels1.tolist() == [0, 1, 0, 1, 0]
    assert np.array_equal(preds1, preds2)  # eval mode: no time masking, no dropout
    assert preds1.dtype == np.int64


def test_an_empty_loader_is_an_error(tmp_path):
    model = make_model(tmp_path, layers=2)
    empty = DataLoader([], batch_size=2, collate_fn=collate_fn)
    with pytest.raises(ValueError, match="smaller than one batch"):
        run_stage(model, HEAD, empty, CPU)
