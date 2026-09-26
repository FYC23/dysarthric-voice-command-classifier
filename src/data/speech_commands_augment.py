"""
BC-ResNet's Speech Commands augmentation (Kim et al. 2021, arXiv 2106.04140,
section 4.1), matching the authors' reference code
(github.com/Qualcomm-AI-research/bcresnet, utils.py Preprocess). Training
split only; used for the reproduction and pretraining runs, not for TORGO.

Reference-code details kept on purpose:
- the time shift and the noise are applied together, with probability 0.8;
- noise is a raw amplitude U(0, 0.1) times the noise segment, not an SNR;
- the shifted-out part is zero-filled;
- the _silence_ class is noise at amplitude U(0, 1) on an empty clip;
- the result is clamped to [-1, 1].
SpecAugment on the log-Mel is in src/model/frontend.py.
"""

from dataclasses import dataclass

import numpy as np

from .noise import NoiseBank


@dataclass(frozen=True)
class BCResNetAugParams:
    noise_prob: float = 0.8
    max_shift_s: float = 0.1
    max_noise_amp: float = 0.1
    max_silence_noise_amp: float = 1.0


DEFAULT_BCRESNET_AUG = BCResNetAugParams()


def shift_with_zeros(audio: np.ndarray, shift: int) -> np.ndarray:
    """Positive shift moves the audio earlier, negative later; the gap is zero-filled."""
    audio = np.asarray(audio, dtype=np.float32)
    n = len(audio)
    out = np.zeros(n, dtype=np.float32)
    if abs(shift) >= n:
        return out
    if shift >= 0:
        out[:n - shift] = audio[shift:]
    else:
        out[-shift:] = audio[:n + shift]
    return out


def augment_command(clip: np.ndarray, sr: int, rng: np.random.Generator, noise: NoiseBank,
                    params: BCResNetAugParams = DEFAULT_BCRESNET_AUG) -> np.ndarray:
    """One training view of a Speech Commands word clip that is already its final length."""
    if rng.random() >= params.noise_prob:
        return np.array(clip, dtype=np.float32)
    shift = int(rng.uniform(-params.max_shift_s, params.max_shift_s) * sr)
    amp = rng.uniform(0.0, params.max_noise_amp)
    mixed = shift_with_zeros(clip, shift) + amp * noise.sample(len(clip), rng)
    return np.clip(mixed, -1.0, 1.0).astype(np.float32)


def make_silence(length: int, rng: np.random.Generator, noise: NoiseBank,
                 params: BCResNetAugParams = DEFAULT_BCRESNET_AUG) -> np.ndarray:
    """A _silence_ training example: background noise at a random level."""
    amp = rng.uniform(0.0, params.max_silence_noise_amp)
    return np.clip(amp * noise.sample(length, rng), -1.0, 1.0).astype(np.float32)
