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
from src.model.bcresnet import BCResNet

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
