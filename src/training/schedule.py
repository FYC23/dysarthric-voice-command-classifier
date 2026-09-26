"""
Per-step learning-rate schedule from the BC-ResNet paper (Kim et al. 2021,
section 4.1): linear warmup from 0 to the peak, then cosine annealing to 0.
"""

import math
from typing import Callable

import torch
from torch.optim.lr_scheduler import LambdaLR


def warmup_cosine(total_steps: int, warmup_steps: int) -> Callable[[int], float]:
    """LR multiplier for the optimizer step with index `step` (0-based)."""
    if total_steps <= 0:
        raise ValueError(f"total_steps must be positive, got {total_steps}")
    if not 0 <= warmup_steps <= total_steps:
        raise ValueError(f"warmup_steps must be in [0, {total_steps}], got {warmup_steps}")

    def factor(step: int) -> float:
        if step >= total_steps:
            return 0.0
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return factor


def warmup_cosine_scheduler(optimizer: torch.optim.Optimizer, total_steps: int,
                            warmup_steps: int) -> LambdaLR:
    """Call scheduler.step() after every optimizer.step()."""
    return LambdaLR(optimizer, warmup_cosine(total_steps, warmup_steps))
