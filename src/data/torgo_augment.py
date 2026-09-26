"""
Training-time waveform augmentation for TORGO, shared by HuBERT and BC-ResNet.
Never applied to validation or test audio.

Per clip, starting from the extracted word (src/audio.py extract_word):
  speed perturbation -> random position in the window -> background noise at a
  random SNR -> random gain.
Why each step and each setting: docs/superpowers/specs/2026-09-25-data-augmentation-design.md
"""

from dataclasses import dataclass
from fractions import Fraction
from typing import Optional, Tuple

import numpy as np
from scipy.signal import resample_poly

from ..audio import fit_to_length
from .noise import NoiseBank, mean_power, mix_at_snr

_MAX_RESAMPLE_DENOMINATOR = 100  # 1/0.9 -> 10/9, 1/1.05 -> 20/21: exact for our factors


@dataclass(frozen=True)
class TorgoAugParams:
    """TORGO recipe settings. Defaults are the spec's; tests override them."""

    # Geng et al. 2020: speed beat tempo and VTLP; four factors within +-10% beat +-15%
    speed_factors: Tuple[float, ...] = (0.9, 0.95, 1.0, 1.05, 1.1)
    noise_prob: float = 0.8                     # BC-ResNet's background-noise probability
    snr_db: Tuple[float, float] = (5.0, 20.0)   # floor kept high so weak dysarthric consonants survive
    gain_db: Tuple[float, float] = (-6.0, 6.0)  # phone-to-mouth distance
    peak_limit: float = 0.99                    # rescale (never clip) above this peak

    def __post_init__(self):
        if not self.speed_factors or any(f <= 0 for f in self.speed_factors):
            raise ValueError(f"speed_factors must be non-empty and positive, got {self.speed_factors}")
        if not 0.0 <= self.noise_prob <= 1.0:
            raise ValueError(f"noise_prob must be in [0, 1], got {self.noise_prob}")
        for name in ("snr_db", "gain_db"):
            low, high = getattr(self, name)
            if low > high:
                raise ValueError(f"{name} range is reversed: {(low, high)}")
        if not 0.0 < self.peak_limit <= 1.0:
            raise ValueError(f"peak_limit must be in (0, 1], got {self.peak_limit}")


DEFAULT_TORGO_AUG = TorgoAugParams()


def speed_perturb(audio: np.ndarray, factor: float) -> np.ndarray:
    """
    Kaldi/sox-style speed change: play `factor` times faster by resampling.
    Duration scales by 1/factor; pitch and formants scale by factor.
    """
    if factor <= 0:
        raise ValueError(f"speed factor must be positive, got {factor}")
    audio = np.asarray(audio, dtype=np.float32)
    if factor == 1.0 or len(audio) == 0:
        return audio.copy()
    ratio = Fraction(1 / factor).limit_denominator(_MAX_RESAMPLE_DENOMINATOR)
    return resample_poly(audio, ratio.numerator, ratio.denominator).astype(np.float32)


def place_in_window(word: np.ndarray, length: int, rng: np.random.Generator) -> np.ndarray:
    """
    Put the word at a uniformly random offset in a silent `length` window.

    The shift range is exactly the room the word leaves, so the word is never
    cut. A word longer than the window keeps its loudest part (fit_to_length).
    """
    n = len(word)
    if n >= length:
        return fit_to_length(word, length).astype(np.float32)
    offset = int(rng.integers(0, length - n + 1))
    out = np.zeros(length, dtype=np.float32)
    out[offset:offset + n] = word
    return out


def apply_gain(audio: np.ndarray, gain_db: float, peak_limit: float) -> np.ndarray:
    """Scale by gain_db; if the peak then exceeds peak_limit, scale down to it (no clipping)."""
    out = np.asarray(audio, dtype=np.float64) * 10 ** (gain_db / 20)
    peak = float(np.max(np.abs(out))) if len(out) else 0.0
    if peak > peak_limit:
        out = out * (peak_limit / peak)
    return out.astype(np.float32)


def augment_word(word: np.ndarray, length: int, rng: np.random.Generator,
                 noise: Optional[NoiseBank],
                 params: TorgoAugParams = DEFAULT_TORGO_AUG) -> np.ndarray:
    """One random training view of a TORGO word, as a `length`-sample float32 window."""
    if params.noise_prob > 0 and noise is None:
        raise ValueError("noise_prob > 0 needs a NoiseBank")
    factor = params.speed_factors[int(rng.integers(len(params.speed_factors)))]
    resampled = speed_perturb(word, factor)
    window = place_in_window(resampled, length, rng)
    if rng.random() < params.noise_prob:
        snr_db = rng.uniform(*params.snr_db)
        window = mix_at_snr(window, noise.sample(length, rng), snr_db,
                            ref_power=mean_power(resampled))
    return apply_gain(window, rng.uniform(*params.gain_db), params.peak_limit)
