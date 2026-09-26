"""
Background noise for training augmentation: a bank of long noise recordings,
random segments from it, and mixing at a target SNR.

The bank is Speech Commands' train/_silence_ folder (5 recordings, 60-95 s,
see Config.NOISE_DIR). Only the train split is used, so no held-out audio
is ever mixed into training clips.
"""

from pathlib import Path
from typing import Sequence

import numpy as np
import soundfile as sf

_SILENT_POWER = 1e-10  # mean power below this (-100 dBFS) counts as silence


class NoiseBank:
    """Long mono noise recordings to draw random segments from."""

    def __init__(self, recordings: Sequence[np.ndarray]):
        if len(recordings) == 0:
            raise ValueError("NoiseBank needs at least one recording")
        cleaned = tuple(np.array(r, dtype=np.float32) for r in recordings)
        if any(r.ndim != 1 or len(r) == 0 for r in cleaned):
            raise ValueError("noise recordings must be non-empty mono (1-D) arrays")
        self._recordings = cleaned

    @classmethod
    def from_dir(cls, directory: Path, sr: int) -> "NoiseBank":
        """Load every .wav in `directory`; each must already be at `sr`."""
        paths = sorted(Path(directory).glob("*.wav"))
        if not paths:
            raise FileNotFoundError(
                f"no .wav noise files in {directory}; run "
                "scripts/download_speech_commands.sh (only train/_silence_ is needed)")
        recordings = []
        for path in paths:
            audio, file_sr = sf.read(path, dtype="float32", always_2d=True)
            if file_sr != sr:
                raise ValueError(f"{path}: sample rate {file_sr}, expected {sr}")
            recordings.append(audio[:, 0])
        return cls(recordings)

    def __len__(self) -> int:
        return len(self._recordings)

    def sample(self, length: int, rng: np.random.Generator) -> np.ndarray:
        """A random `length`-sample segment of a random recording (tiled if too short)."""
        if length <= 0:
            raise ValueError(f"segment length must be positive, got {length}")
        recording = self._recordings[rng.integers(len(self._recordings))]
        if len(recording) < length:
            recording = np.tile(recording, -(-length // len(recording)))
        start = int(rng.integers(0, len(recording) - length + 1))
        return recording[start:start + length].copy()


def mean_power(x: np.ndarray) -> float:
    """Mean squared amplitude; 0.0 for an empty array."""
    if len(x) == 0:
        return 0.0
    return float(np.mean(np.square(x, dtype=np.float64)))


def mix_at_snr(signal: np.ndarray, noise: np.ndarray, snr_db: float,
               ref_power: float) -> np.ndarray:
    """
    signal + noise, with the noise scaled so 10*log10(ref_power / noise power) == snr_db.

    ref_power is the speech power, measured on the word rather than the
    zero-padded window, so padding does not make the noise quieter. Silent
    noise or silent speech returns an unchanged copy (nothing to scale against).
    """
    signal = np.asarray(signal)
    noise = np.asarray(noise)
    if signal.shape != noise.shape:
        raise ValueError(f"signal shape {signal.shape} != noise shape {noise.shape}")
    noise_power = mean_power(noise)
    if ref_power < _SILENT_POWER or noise_power < _SILENT_POWER:
        return signal.astype(np.float32)
    scale = np.sqrt(ref_power / (noise_power * 10 ** (snr_db / 10)))
    return (signal + scale * noise).astype(np.float32)
