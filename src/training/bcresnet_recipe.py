"""
The fixed BC-ResNet training settings of all three stages, and where runs go.
Stage 1 is the paper's recipe (Kim et al. 2021, section 4.1). Stages 2-3 are
fixed in advance and never tuned on LOSO results.
Design: docs/superpowers/specs/2026-09-26-bcresnet-training-design.md
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from torch.optim.lr_scheduler import LambdaLR

from src.config import Config
from src.model.bcresnet import BCResNet
from src.model.frontend import HOP_LENGTH
from src.training.schedule import warmup_cosine_scheduler

TAUS = (1, 2, 3, 8)
SAMPLE_RATE = Config.SAMPLE_RATE
WINDOW_S = Config.MAX_AUDIO_LENGTH
WINDOW_SAMPLES = Config.MAX_AUDIO_SAMPLES
WINDOW_FRAMES = WINDOW_SAMPLES // HOP_LENGTH + 1  # 201 log-Mel frames


@dataclass(frozen=True)
class SgdStage:
    """One training stage: SGD, then a per-step warmup + cosine or constant learning rate."""

    epochs: int
    peak_lr: float
    warmup_epochs: int
    batch_size: int
    momentum: float = 0.9
    weight_decay: float = 1e-3
    head_only: bool = False       # train only the class head's weights; batch norm still adapts
    cosine: bool = True           # False: constant peak_lr
    class_weighted: bool = False  # "balanced" cross-entropy weights
    drop_last: bool = False       # drop the final incomplete batch

    def __post_init__(self):
        if self.epochs <= 0 or self.batch_size <= 0:
            raise ValueError(f"epochs and batch_size must be positive: {self}")
        if not 0 <= self.warmup_epochs <= self.epochs:
            raise ValueError(f"warmup_epochs must be in [0, epochs]: {self}")
        if self.peak_lr <= 0:
            raise ValueError(f"peak_lr must be positive: {self}")


PRETRAIN = SgdStage(epochs=200, peak_lr=0.1, warmup_epochs=5, batch_size=100)
CONTROL_HEAD_WARMUP = SgdStage(epochs=5, peak_lr=0.01, warmup_epochs=0, batch_size=32,
                               head_only=True, cosine=False, class_weighted=True,
                               drop_last=True)
CONTROL_FINETUNE = SgdStage(epochs=40, peak_lr=0.01, warmup_epochs=2, batch_size=32,
                            class_weighted=True, drop_last=True)
DYSARTHRIC_FINETUNE = SgdStage(epochs=30, peak_lr=0.003, warmup_epochs=2, batch_size=32,
                               class_weighted=True, drop_last=True)


def make_optimizer(model: BCResNet, stage: SgdStage) -> torch.optim.SGD:
    params = model.head.parameters() if stage.head_only else model.parameters()
    return torch.optim.SGD(params, lr=stage.peak_lr, momentum=stage.momentum,
                           weight_decay=stage.weight_decay)


def make_scheduler(optimizer: torch.optim.Optimizer, stage: SgdStage,
                   steps_per_epoch: int) -> LambdaLR:
    if steps_per_epoch <= 0:
        raise ValueError("the training set is smaller than one batch; "
                         f"batch_size={stage.batch_size}, drop_last={stage.drop_last}")
    if not stage.cosine:
        return LambdaLR(optimizer, lambda step: 1.0)
    return warmup_cosine_scheduler(optimizer, stage.epochs * steps_per_epoch,
                                   stage.warmup_epochs * steps_per_epoch)


def run_name(tau: float) -> str:
    """The model's name in eval-harness tables, e.g. bcresnet-8."""
    return f"bcresnet-{tau:g}"


def seed_dir(runs_dir: Path, tau: float, seed: int) -> Path:
    return Path(runs_dir) / run_name(tau) / f"seed{seed}"


def pick_device(name: Optional[str] = None) -> torch.device:
    """`name` if given, else CUDA, then MPS, then CPU."""
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
