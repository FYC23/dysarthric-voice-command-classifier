"""
BC-ResNet (Kim et al., 2021, "Broadcasted Residual Learning for Efficient
Keyword Spotting", arXiv:2106.04140), following the paper's Table 1 and
Figure 2 and the authors' reference code
(github.com/Qualcomm-AI-research/bcresnet).

Input: (batch, 1, 40 mel bins, frames). BC-ResNet-tau scales every channel
width of BC-ResNet-1 by tau.
"""

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

N_MELS = 40
SSN_SUBBANDS = 5
DROPOUT = 0.1
STAGE_BLOCKS = (2, 2, 4, 4)
STRIDED_STAGES = (1, 2)  # stages whose first block halves the frequency axis


class SubSpectralNorm(nn.Module):
    """Batch norm computed separately for each of S frequency sub-bands."""

    def __init__(self, channels: int, subbands: int = SSN_SUBBANDS):
        super().__init__()
        self.subbands = subbands
        self.bn = nn.BatchNorm2d(channels * subbands)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, c, f, t = x.shape
        x = self.bn(x.reshape(n, c * self.subbands, f // self.subbands, t))
        return x.reshape(n, c, f, t)


class BCResBlock(nn.Module):
    """
    y = x + f2(x) + BC(f1(avgpool_freq(f2(x))))   (paper eq. 2)

    f2: 3x1 frequency-depthwise conv + SubSpectralNorm (2D features)
    f1: 1x3 temporal-depthwise conv + BN + swish, 1x1 conv, dropout (1D features)
    A transition block (channel change) starts with a 1x1 conv + BN + ReLU and
    drops the identity shortcut.
    """

    def __init__(self, in_ch: int, out_ch: int, freq_stride: int, dilation: int):
        super().__init__()
        self.transition = in_ch != out_ch
        self.expand = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        ) if self.transition else nn.Identity()
        self.f2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, (3, 1), stride=(freq_stride, 1), padding=(1, 0),
                      groups=out_ch, bias=False),
            SubSpectralNorm(out_ch),
        )
        self.f1 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, (1, 3), padding=(0, dilation), dilation=(1, dilation),
                      groups=out_ch, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 1, bias=False),
            nn.Dropout2d(DROPOUT),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x2d = self.f2(self.expand(x))
        x1d = self.f1(x2d.mean(dim=2, keepdim=True))  # broadcast back over frequency
        out = x2d + x1d
        if not self.transition:
            out = out + shortcut
        return F.relu(out)


def _stage_widths(tau: float) -> List[int]:
    """Table 1 widths for BC-ResNet-1 (16 | 8 12 16 20 | 32), scaled by tau."""
    base = 8 * tau
    return [int(base * m) for m in (2, 1, 1.5, 2, 2.5, 4)]


def _block_plan(widths: List[int]) -> List[Tuple[int, int, int, int]]:
    """(in_ch, out_ch, freq_stride, dilation) for all 12 blocks."""
    plan, in_ch = [], widths[0]
    for stage, n_blocks in enumerate(STAGE_BLOCKS):
        out_ch = widths[stage + 1]
        for i in range(n_blocks):
            stride = 2 if stage in STRIDED_STAGES and i == 0 else 1
            plan.append((in_ch, out_ch, stride, 2 ** stage))
            in_ch = out_ch
    return plan


class BCResNet(nn.Module):
    def __init__(self, tau: float = 1, num_classes: int = 12):
        super().__init__()
        w = _stage_widths(tau)
        self.head = nn.Sequential(
            nn.Conv2d(1, w[0], 5, stride=(2, 1), padding=2, bias=False),
            nn.BatchNorm2d(w[0]),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(*(BCResBlock(*p) for p in _block_plan(w)))
        self.classifier = nn.Sequential(
            # 5x5 depthwise, no frequency padding: collapses the 5 remaining bins to 1
            nn.Conv2d(w[4], w[4], 5, padding=(0, 2), groups=w[4], bias=False),
            nn.Conv2d(w[4], w[5], 1, bias=False),
            nn.BatchNorm2d(w[5]),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(w[5], num_classes, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.blocks(self.head(x))).flatten(1)
