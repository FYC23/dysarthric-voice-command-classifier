"""
BC-ResNet must reproduce the paper's cost numbers (Kim et al., 2021, Table 3),
which also checks the MAC counter against published figures.

Paper input: 40 log-mel bins, 1 s at a 10 ms hop -> 101 frames, 12 classes.

Our MACs count convolutions and matrix multiplies only (src/eval/cost.py). The
paper's #Mult also counts per-element work: it matches our MACs plus 2 ops per
batch-norm output element plus 1 per activation output (0.1%, 1.2% and 2.0%
off at tau 1, 3, 8; the test allows 3%). We keep the conv/matmul convention
because batch norm folds into the conv at export, and it counts transformer
baselines the same way.
"""

import pytest
import torch
from torch import nn

from src.eval.cost import count_macs, count_params
from src.model.bcresnet import HEAD_PREFIX, BCResNet, load_pretrained_body

ONE_SECOND = torch.zeros(1, 1, 40, 101)
TWO_SECONDS = torch.zeros(1, 1, 40, 201)

PAPER_TABLE_3 = [  # tau, #Param, #Mult
    (1, 9.2e3, 3.1e6),
    (3, 54.2e3, 16.2e6),
    (8, 321e3, 89.1e6),
]


def elementwise_ops(model: nn.Module, x: torch.Tensor) -> int:
    """2 per batch-norm output element + 1 per ReLU/SiLU output element."""
    total = 0

    def hook(module, _inputs, output):
        nonlocal total
        total += output.numel() * (2 if isinstance(module, nn.BatchNorm2d) else 1)

    handles = [m.register_forward_hook(hook) for m in model.modules()
               if isinstance(m, (nn.BatchNorm2d, nn.ReLU, nn.SiLU))]
    with torch.no_grad():
        model.eval()(x)
    for h in handles:
        h.remove()
    return total


@pytest.mark.parametrize("tau, params, _mults", PAPER_TABLE_3)
def test_params_match_paper(tau, params, _mults):
    assert count_params(BCResNet(tau=tau, num_classes=12)) == pytest.approx(params, rel=0.01)


@pytest.mark.parametrize("tau, _params, mults", PAPER_TABLE_3)
def test_macs_match_paper_once_elementwise_ops_are_added(tau, _params, mults):
    model = BCResNet(tau=tau, num_classes=12)
    total = count_macs(model, ONE_SECOND) + elementwise_ops(model, ONE_SECOND)
    assert total == pytest.approx(mults, rel=0.03)


def test_outputs_one_logit_per_class():
    model = BCResNet(tau=1, num_classes=20).eval()
    assert model(TWO_SECONDS).shape == (1, 20)


def test_macs_double_at_our_two_second_window():
    model = BCResNet(tau=8, num_classes=20)
    ratio = count_macs(model, TWO_SECONDS) / count_macs(model, ONE_SECOND)
    assert ratio == pytest.approx(201 / 101, rel=0.02)


class TestPretrainedBody:
    def test_head_is_the_final_class_conv(self):
        model = BCResNet(tau=1, num_classes=20)
        assert model.head is model.classifier[-1]
        assert isinstance(model.head, nn.Conv2d) and model.head.out_channels == 20
        head_keys = {k for k in model.state_dict() if k.startswith(HEAD_PREFIX)}
        assert head_keys == {HEAD_PREFIX + "weight", HEAD_PREFIX + "bias"}

    def test_body_loads_unchanged_and_head_keeps_its_fresh_init(self):
        torch.manual_seed(0)
        pretrained = BCResNet(tau=1, num_classes=36)
        torch.manual_seed(1)
        model = BCResNet(tau=1, num_classes=20)
        fresh_head = model.head.weight.detach().clone()
        load_pretrained_body(model, pretrained.state_dict())
        loaded = model.state_dict()
        for key, value in pretrained.state_dict().items():
            if not key.startswith(HEAD_PREFIX):
                assert torch.equal(loaded[key], value), key
        assert torch.equal(model.head.weight, fresh_head)

    def test_other_width_is_refused(self):
        with pytest.raises(ValueError, match="body"):
            load_pretrained_body(BCResNet(tau=1, num_classes=20),
                                 BCResNet(tau=2, num_classes=36).state_dict())

    def test_missing_body_weights_are_refused(self):
        state = BCResNet(tau=1, num_classes=36).state_dict()
        state.pop(next(k for k in state if not k.startswith(HEAD_PREFIX)))
        with pytest.raises(ValueError, match="missing"):
            load_pretrained_body(BCResNet(tau=1, num_classes=20), state)
