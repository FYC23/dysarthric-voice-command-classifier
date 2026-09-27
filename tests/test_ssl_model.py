"""SSL command classifier: weighted sum of layers, one head, explicit freezing."""

import pytest
import torch

from ssl_factories import TINY_HIDDEN, tiny_spec
from src.model.architecture import (
    CommandHead, LayerWeightedSum, SSLCommandClassifier, set_trainable,
)
from src.model.backbones import load_backbone

WINDOW = 32000


def make_model(tmp_path, layers=6, stable=False):
    return SSLCommandClassifier(load_backbone(tiny_spec(tmp_path, layers, stable)), num_labels=20)


def trainable_names(model):
    return {n for n, p in model.named_parameters() if p.requires_grad}


def test_weighted_sum_starts_as_a_plain_average_of_normalised_layers():
    states = [torch.randn(2, 5, 8) for _ in range(3)]
    out = LayerWeightedSum(3)(states)
    normed = [torch.nn.functional.layer_norm(s, (8,)) for s in states]
    assert torch.allclose(out, sum(normed) / 3, atol=1e-6)


def test_weighted_sum_ignores_the_scale_of_a_layer():
    states = [torch.randn(2, 5, 8) for _ in range(3)]
    ws = LayerWeightedSum(3)
    scaled = [states[0] * 100.0, states[1], states[2]]
    assert torch.allclose(ws(states), ws(scaled), atol=1e-4)


def test_weighted_sum_weights_receive_gradients():
    ws = LayerWeightedSum(3)
    ws([torch.randn(2, 5, 8) * (i + 1) for i in range(3)]).pow(2).sum().backward()
    assert ws.weights.grad is not None and ws.weights.grad.abs().sum() > 0


def test_weighted_sum_rejects_the_wrong_number_of_layers():
    with pytest.raises(ValueError, match="hidden states"):
        LayerWeightedSum(3)([torch.randn(1, 2, 4)] * 2)


def test_head_is_weighted_sum_then_pooling_then_mlp():
    head = CommandHead(num_states=4, hidden_size=TINY_HIDDEN, num_labels=20)
    assert head.layer_weights.weights.numel() == 4
    assert head.classifier[0].in_features == TINY_HIDDEN
    assert head.classifier[0].out_features == TINY_HIDDEN // 2
    assert head.classifier[-1].out_features == 20
    assert head([torch.randn(2, 7, TINY_HIDDEN) for _ in range(4)]).shape == (2, 20)


@pytest.mark.parametrize("stable", [False, True])
def test_classifier_uses_every_hidden_state(tmp_path, stable):
    model = make_model(tmp_path, layers=3, stable=stable)
    assert model.num_layers == 3
    assert model.head.layer_weights.weights.numel() == 4  # 3 layers + the input to the first
    out = model(torch.randn(2, WINDOW), labels=torch.tensor([0, 5]))
    assert out["logits"].shape == (2, 20)
    assert out["loss"].ndim == 0


def test_classifier_loss_uses_class_weights(tmp_path):
    model = make_model(tmp_path, layers=2).eval()
    x, y = torch.randn(2, WINDOW), torch.tensor([0, 5])
    weights = torch.ones(20)
    weights[5] = 10.0
    assert not torch.isclose(model(x, labels=y)["loss"],
                             model(x, labels=y, class_weights=weights)["loss"])


def test_no_loss_without_labels(tmp_path):
    assert set(make_model(tmp_path, layers=2)(torch.randn(1, WINDOW))) == {"logits"}


@pytest.mark.parametrize("start_trainable", [True, False])
def test_set_trainable_top_n_whatever_the_starting_state(tmp_path, start_trainable):
    model = make_model(tmp_path, layers=6)
    for p in model.parameters():
        p.requires_grad_(start_trainable)  # True is the bug-2 case: built unfrozen
    set_trainable(model, 4)
    names = trainable_names(model)
    head = {n for n, _ in model.head.named_parameters()}
    assert {n for n in names if n.startswith("head.")} == {f"head.{n}" for n in head}
    backbone = {n for n in names if n.startswith("backbone.")}
    assert backbone and all(n.startswith(tuple(f"backbone.encoder.layers.{i}." for i in (2, 3, 4, 5)))
                            for n in backbone)
    per_layer = {n for n, _ in model.backbone.encoder.layers[5].named_parameters()}
    assert {f"backbone.encoder.layers.5.{n}" for n in per_layer} <= backbone


def expected_trainable_names(model, top_n):
    """The exact set of parameter names set_trainable(model, top_n) should leave trainable."""
    layers = model.backbone.encoder.layers
    num_layers = len(layers)
    backbone = {
        f"backbone.encoder.layers.{i}.{n}"
        for i in range(num_layers - top_n, num_layers)
        for n, _ in layers[i].named_parameters()
    }
    head = {f"head.{n}" for n, _ in model.head.named_parameters()}
    return head | backbone


@pytest.mark.parametrize("start_trainable", [True, False])
@pytest.mark.parametrize("top_n", [0, 1, 6])
def test_set_trainable_exact_trainable_set(tmp_path, top_n, start_trainable):
    model = make_model(tmp_path, layers=6)
    for p in model.parameters():
        p.requires_grad_(start_trainable)  # True is the bug-2 case: built unfrozen
    set_trainable(model, top_n)
    assert trainable_names(model) == expected_trainable_names(model, top_n)


def test_set_trainable_zero_is_head_only(tmp_path):
    model = make_model(tmp_path, layers=3)
    set_trainable(model, 0)
    assert trainable_names(model) == {f"head.{n}" for n, _ in model.head.named_parameters()}


def test_set_trainable_all_layers_keeps_the_front_end_frozen(tmp_path):
    model = make_model(tmp_path, layers=2, stable=True)
    set_trainable(model, 2)
    trainable = trainable_names(model)
    for prefix in ("backbone.feature_extractor.", "backbone.feature_projection.",
                   "backbone.encoder.pos_conv_embed.", "backbone.encoder.layer_norm.",
                   "backbone.masked_spec_embed"):
        assert not any(n.startswith(prefix) for n in trainable), prefix


@pytest.mark.parametrize("top_n", [-1, 4])
def test_set_trainable_rejects_impossible_counts(tmp_path, top_n):
    with pytest.raises(ValueError, match="top_n"):
        set_trainable(make_model(tmp_path, layers=3), top_n)


def test_set_trainable_freezes_the_cnn_front_ends_grad_flag(tmp_path):
    """
    HubertFeatureEncoder.forward force-sets its output's requires_grad to True
    whenever the module's own `_requires_grad` flag is set and the model is
    training -- regardless of its parameters' requires_grad. set_trainable
    must clear that flag too, or the "frozen" CNN still gets a live
    autograd graph built through it on every training step.
    """
    model = make_model(tmp_path, layers=2)
    set_trainable(model, 0)
    model.train()
    out = model.backbone(input_values=torch.randn(2, WINDOW), output_hidden_states=True)
    assert out.hidden_states[0].requires_grad is False
    assert all(not h.requires_grad for h in out.hidden_states)
