"""
Waveform preparation for model input. Any inference path must call
prepare_waveform too, so the model sees audio prepared the same way as in training.

Pipeline: remove DC offset -> trim leading/trailing silence with an energy VAD
-> fit to a fixed window (centre-pad, or keep the loudest window).

Depends only on numpy/scipy (no torch/librosa), so it can be reused anywhere.

Why a noise-floor-relative VAD: TORGO recordings have only 21-44 dB between the
noise floor and speech (per speaker), so a trim relative to the loudest frame
(librosa's top_db) keeps the whole file. Median TORGO files are 2.1 s but the
word itself is ~0.3-0.5 s, so trimming removes most of what the model sees.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy.signal import butter, sosfiltfilt

_EPS = 1e-10  # energy floor (-100 dB), keeps log10 finite on digital silence
_HIGHPASS_ORDER = 4


@dataclass(frozen=True)
class VadParams:
    """Energy-VAD settings. Defaults were checked by eye on TORGO and Speech Commands."""

    frame_s: float = 0.025          # analysis frame
    hop_s: float = 0.010            # frame step
    highpass_hz: float = 80.0       # detection only: removes DC drift, start-of-file ramps, rumble
    noise_percentile: float = 10.0  # noise floor = this percentile of frame energies
    noise_margin_db: float = 12.0   # speech frames must be this far above the noise floor...
    peak_range_db: float = 40.0     # ...and within this range of the loudest frame
    min_run_s: float = 0.05         # shorter voiced runs (clicks, lip smacks) are ignored
    pad_s: float = 0.15             # context kept either side of the detected speech


DEFAULT_VAD = VadParams()


def remove_dc(audio: np.ndarray) -> np.ndarray:
    """Return a copy with the mean removed (some Speech Commands clips sit at +0.08)."""
    audio = np.asarray(audio, dtype=np.float64)
    return audio - audio.mean() if len(audio) else audio.copy()


def _frame_energy_db(audio: np.ndarray, frame: int, hop: int) -> np.ndarray:
    n_frames = 1 + (len(audio) - frame) // hop
    idx = np.arange(frame)[None, :] + hop * np.arange(n_frames)[:, None]
    return 10 * np.log10(np.mean(audio[idx] ** 2, axis=1) + _EPS)


def _voiced_runs(voiced: np.ndarray, min_frames: int) -> list:
    """(first, last) frame indices of runs of True that are at least min_frames long."""
    edges = np.diff(np.concatenate([[0], voiced.astype(np.int8), [0]]))
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    return [(s, e - 1) for s, e in zip(starts, ends) if e - s >= min_frames]


def detect_speech(audio: np.ndarray, sr: int,
                  params: VadParams = DEFAULT_VAD) -> Optional[Tuple[int, int]]:
    """
    Return (start, end) sample indices spanning all detected speech, or None.

    Keeps everything from the first to the last voiced run, so multi-part
    utterances (common in dysarthric speech) are never split.
    """
    frame, hop = int(params.frame_s * sr), int(params.hop_s * sr)
    if len(audio) < 2 * frame:
        return None

    sos = butter(_HIGHPASS_ORDER, params.highpass_hz, btype="highpass", fs=sr, output="sos")
    filtered = sosfiltfilt(sos, remove_dc(audio))
    energy = _frame_energy_db(filtered, frame, hop)

    noise_db = np.percentile(energy, params.noise_percentile)
    threshold = max(noise_db + params.noise_margin_db, energy.max() - params.peak_range_db)
    runs = _voiced_runs(energy > threshold, max(1, int(round(params.min_run_s / params.hop_s))))
    if not runs:
        return None
    return runs[0][0] * hop, min(len(audio), runs[-1][1] * hop + frame)


def trim_silence(audio: np.ndarray, sr: int, params: VadParams = DEFAULT_VAD) -> np.ndarray:
    """Return the detected speech plus padding; an unchanged copy if no speech is found."""
    audio = np.asarray(audio)
    span = detect_speech(audio, sr, params)
    if span is None:
        return audio.copy()
    pad = int(params.pad_s * sr)
    start, end = max(0, span[0] - pad), min(len(audio), span[1] + pad)
    return audio[start:end].copy()


def fit_to_length(audio: np.ndarray, length: int) -> np.ndarray:
    """
    Return exactly `length` samples.

    Shorter audio is zero-padded on both sides (word centred, as in Speech
    Commands). Longer audio keeps the highest-energy window instead of the
    centre, so a word near either end is not cut off.
    """
    audio = np.asarray(audio)
    n = len(audio)
    if n == length:
        return audio.copy()
    if n < length:
        left = (length - n) // 2
        return np.pad(audio, (left, length - n - left), mode="constant")
    power = np.concatenate([[0.0], np.cumsum(audio.astype(np.float64) ** 2)])
    start = int(np.argmax(power[length:] - power[:-length]))
    return audio[start:start + length].copy()


def kept_length_s(audio: np.ndarray, sr: int, params: VadParams = DEFAULT_VAD) -> float:
    """Seconds left after silence trimming (the whole clip if no speech is found)."""
    return len(trim_silence(audio, sr, params)) / sr


def prepare_waveform(audio: np.ndarray, sr: int, length: int,
                     params: VadParams = DEFAULT_VAD,
                     segment: Optional[Tuple[float, float]] = None) -> np.ndarray:
    """
    DC removal -> silence trim -> fixed window. Returns a new float32 array.

    `segment` = (start_s, end_s) is a hand-labelled word location. It replaces
    the VAD: dysarthric clips often start with struggle sounds that the VAD
    (correctly) keeps but that are louder than the word itself.
    """
    audio = remove_dc(audio)
    if segment is None:
        kept = trim_silence(audio, sr, params)
    else:
        start_s, end_s = segment
        if not 0 <= start_s < end_s:
            raise ValueError(f"invalid segment {segment}: need 0 <= start < end")
        pad = params.pad_s
        kept = audio[max(0, int((start_s - pad) * sr)):int((end_s + pad) * sr)]
    return fit_to_length(kept, length).astype(np.float32)
