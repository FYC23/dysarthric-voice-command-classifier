"""
Model cost in the convention of the BC-ResNet paper (Kim et al., 2021,
arXiv:2106.04140, Tables 2-3): #Params and multiply-accumulates for one
forward pass on a single input.

MACs come from PyTorch's FlopCounterMode, which counts matrix multiplies and
convolutions (FLOPs = 2 x MACs) and ignores cheap elementwise ops, batch norm
and the log-mel front end. Always record the input length: conv-net MACs grow
linearly with the number of time frames, so the paper's 1 s figures double
at our 2 s window.
"""

from dataclasses import dataclass
from typing import Any, Callable

import torch
from torch import nn
from torch.utils.flop_counter import FlopCounterMode, sdpa_flop_count

# torch's counter knows the CUDA attention kernels but not the ones CPU and
# MPS dispatch scaled_dot_product_attention to, which it would count as 0.
# Same formula (scores + weighted sum), so a model costs the same on any machine.
def _attention_flops(query_shape, key_shape, value_shape, *args, out_shape=None, **kwargs) -> int:
    return sdpa_flop_count(query_shape, key_shape, value_shape)


_ATTENTION_OPS = {
    torch.ops.aten._scaled_dot_product_flash_attention_for_cpu: _attention_flops,
    torch.ops.aten._scaled_dot_product_attention_math_for_mps: _attention_flops,
}


@dataclass(frozen=True)
class CostProfile:
    params: int
    macs: int
    input_seconds: float
    note: str = ""  # e.g. "encoder padded to 30 s", "mean over clips incl. decoding"

    @property
    def int8_mb(self) -> float:
        """Weights-only size at one byte per parameter (on-device size comes in step 6)."""
        return self.params / 1e6


def count_params(model: nn.Module) -> int:
    """All parameters that ship with the model (batch-norm running stats excluded)."""
    return sum(p.numel() for p in model.parameters())


def count_macs_of(fn: Callable[[], Any]) -> int:
    """MACs of whatever `fn` runs, e.g. a full ASR decode; caller sets eval mode."""
    with torch.no_grad(), FlopCounterMode(display=False, custom_mapping=_ATTENTION_OPS) as counter:
        fn()
    return counter.get_total_flops() // 2


def count_macs(model: nn.Module, example_input: torch.Tensor) -> int:
    """MACs of one eval-mode forward pass on a single input (batch size 1)."""
    if example_input.shape[0] != 1:
        raise ValueError(f"MACs are per single input; got batch of {example_input.shape[0]}")
    was_training = model.training
    model.eval()
    try:
        return count_macs_of(lambda: model(example_input))
    finally:
        model.train(was_training)


def profile_model(model: nn.Module, example_input: torch.Tensor, input_seconds: float,
                  note: str = "") -> CostProfile:
    return CostProfile(params=count_params(model), macs=count_macs(model, example_input),
                       input_seconds=input_seconds, note=note)
