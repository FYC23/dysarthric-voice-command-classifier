"""Tests for parameter and MAC counting on models small enough to count by hand."""

import pytest
import torch
from torch import nn

from src.eval.cost import CostProfile, count_macs, count_params, profile_model


def test_linear_layer_params_and_macs():
    # 10 x 5 weights + 5 biases; one input needs 10 * 5 multiply-accumulates
    layer = nn.Linear(10, 5)
    assert count_params(layer) == 55
    assert count_macs(layer, torch.zeros(1, 10)) == 50


def test_conv_macs_scale_with_output_positions():
    # 4 output channels x 8 x 8 positions x (1 in-channel x 3 x 3 kernel)
    conv = nn.Conv2d(1, 4, 3, padding=1)
    assert count_macs(conv, torch.zeros(1, 1, 8, 8)) == 4 * 64 * 9


def test_depthwise_conv_counts_one_input_channel_per_filter():
    # groups=4: each of 4 filters sees 1 channel, so 4 x 64 x 9, not 4 x 64 x 36
    conv = nn.Conv2d(4, 4, 3, padding=1, groups=4, bias=False)
    assert count_macs(conv, torch.zeros(1, 4, 8, 8)) == 4 * 64 * 9


def test_batch_of_more_than_one_is_rejected():
    with pytest.raises(ValueError, match="single input"):
        count_macs(nn.Linear(10, 5), torch.zeros(2, 10))


def test_counting_leaves_the_model_in_its_original_mode():
    model = nn.Sequential(nn.Linear(4, 4), nn.Dropout(0.5))
    model.train()
    count_macs(model, torch.zeros(1, 4))
    assert model.training


def test_profile_reports_int8_size_as_one_byte_per_param():
    profile = profile_model(nn.Linear(1000, 1000), torch.zeros(1, 1000), input_seconds=2.0)
    assert profile == CostProfile(params=1_001_000, macs=1_000_000, input_seconds=2.0)
    assert profile.int8_mb == pytest.approx(1.001)


class _Attention(nn.Module):
    def forward(self, qkv):
        return nn.functional.scaled_dot_product_attention(qkv, qkv, qkv)


@pytest.mark.parametrize("device", [
    "cpu",
    pytest.param("mps", marks=pytest.mark.skipif(not torch.backends.mps.is_available(),
                                                 reason="no MPS")),
])
def test_attention_matmuls_are_counted_on_every_device(device):
    # Scores (8x4 @ 4x8) and output (8x8 @ 8x4) for 2 heads: 2 x (256 + 256).
    # CPU and MPS dispatch SDPA to kernels torch's counter does not know; a
    # device-dependent count would make the same model's cost differ by machine.
    qkv = torch.zeros(1, 2, 8, 4, device=device)
    assert count_macs(_Attention().to(device), qkv) == 2 * (8 * 4 * 8 + 8 * 8 * 4)
