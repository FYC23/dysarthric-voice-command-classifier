"""
BC-ResNet input features and SpecAugment (Kim et al. 2021 section 4.1; the
reference code's LogMel and spec_augment in utils.py).

LogMel: 40 mel bins, 30 ms window, 10 ms hop, n_fft 512, log(mel + 1e-6).
SpecAugment: 2 frequency masks and 2 time masks per second of audio (the
paper's 2 on 1 s clips, 4 on our 2 s window:
docs/superpowers/specs/2026-09-26-bcresnet-training-design.md), no time
warping, masks set to 0 as in the reference (0 is log(1), not silence). The
time parameter is 20 frames; the frequency parameter grows with width tau,
and BC-ResNet-1 uses none.
Training batches only: never applied to validation or test features.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torchaudio

from .bcresnet import N_MELS

SAMPLE_RATE = 16000
N_FFT = 512
WIN_LENGTH = 480  # 30 ms
HOP_LENGTH = 160  # 10 ms
LOG_OFFSET = 1e-6
TIME_MASK_PARAM = 20
NUM_FREQ_MASKS = 2
TIME_MASKS_PER_SECOND = 2  # the paper's 2 on its 1 s clips; 4 on our 2 s window
FREQ_MASK_PARAM_BY_TAU = {1: 0, 1.5: 1, 2: 3, 3: 5, 6: 7, 8: 7}
SPEC_AUGMENT_MIN_TAU = 1.5


class LogMel(nn.Module):
    """(batch, samples) waveform -> (batch, 1, n_mels, frames) log-Mel."""

    def __init__(self, sample_rate: int = SAMPLE_RATE, n_mels: int = N_MELS):
        super().__init__()
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate, n_fft=N_FFT, win_length=WIN_LENGTH,
            hop_length=HOP_LENGTH, n_mels=n_mels,
        )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.dim() != 2:
            raise ValueError(f"expected a (batch, samples) waveform, got shape {tuple(waveform.shape)}")
        return torch.log(self.mel(waveform) + LOG_OFFSET).unsqueeze(1)


@dataclass(frozen=True)
class SpecAugParams:
    freq_mask_param: int
    time_mask_param: int = TIME_MASK_PARAM
    num_freq_masks: int = NUM_FREQ_MASKS
    num_time_masks: int = TIME_MASKS_PER_SECOND  # for a 1 s input


def spec_augment_params(tau: float, window_s: float = 1.0) -> Optional[SpecAugParams]:
    """
    The paper's SpecAugment for BC-ResNet-tau on a `window_s` input, or None
    (BC-ResNet-1 uses none). Time masks keep the paper's density per second.
    """
    if tau not in FREQ_MASK_PARAM_BY_TAU:
        raise ValueError(f"no SpecAugment setting for tau={tau}; "
                         f"known: {sorted(FREQ_MASK_PARAM_BY_TAU)}")
    if window_s <= 0:
        raise ValueError(f"window must be positive, got {window_s} s")
    if tau < SPEC_AUGMENT_MIN_TAU:
        return None
    return SpecAugParams(freq_mask_param=FREQ_MASK_PARAM_BY_TAU[tau],
                         num_time_masks=max(1, round(TIME_MASKS_PER_SECOND * window_s)))


def _mask_span(param: int, size: int, generator: torch.Generator) -> Tuple[int, int]:
    """(start, width), width = floor(U[0, param)) as in the reference code."""
    width = min(int(torch.rand((), generator=generator).item() * param), size)
    start = int(torch.randint(0, size - width + 1, (), generator=generator).item())
    return start, width


def spec_augment(features: torch.Tensor, params: SpecAugParams,
                 generator: torch.Generator) -> torch.Tensor:
    """A masked copy of (batch, 1, freq, time) features, with independent masks per example."""
    out = features.clone()
    n_freq, n_time = out.shape[2], out.shape[3]
    for i in range(out.shape[0]):
        for _ in range(params.num_freq_masks):
            start, width = _mask_span(params.freq_mask_param, n_freq, generator)
            out[i, :, start:start + width, :] = 0
        for _ in range(params.num_time_masks):
            start, width = _mask_span(params.time_mask_param, n_time, generator)
            out[i, :, :, start:start + width] = 0
    return out
