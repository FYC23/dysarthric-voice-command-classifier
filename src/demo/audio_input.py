"""
Gradio's microphone / upload audio -> the models' 2 s window, through the same
prepare_waveform path as evaluation (src/audio.py), so a visitor's clip is
prepared exactly like a test clip.
"""

from dataclasses import dataclass
from math import gcd

import numpy as np
from scipy.signal import resample_poly

from src.audio import detect_speech, prepare_waveform
from src.config import Config


class AudioInputError(ValueError):
    """The audio cannot be used; the message is shown to the user as is."""


@dataclass(frozen=True)
class ModelInput:
    window: np.ndarray   # float32, Config.MAX_AUDIO_SAMPLES samples at Config.SAMPLE_RATE
    speech_found: bool   # False: the VAD found no word, the window is the loudest 2 s


def to_float_mono(samples: np.ndarray) -> np.ndarray:
    """Integer PCM or float audio, mono or stereo -> new float64 mono array in [-1, 1]."""
    audio = np.asarray(samples)
    if audio.size == 0:
        raise AudioInputError("The recording is empty.")
    if np.issubdtype(audio.dtype, np.integer):
        audio = audio.astype(np.float64) / float(np.iinfo(audio.dtype).max + 1)
    else:
        audio = audio.astype(np.float64)
    if audio.ndim == 2:
        # Gradio gives (samples, channels); the channel axis is the short one either way
        audio = audio.mean(axis=int(np.argmin(audio.shape)))
    elif audio.ndim != 1:
        raise AudioInputError(f"Expected mono or stereo audio, got shape {audio.shape}.")
    if not np.isfinite(audio).all():
        raise AudioInputError("The audio contains NaN or infinite samples.")
    return audio


def _resample(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    if sample_rate == Config.SAMPLE_RATE:
        return audio
    g = gcd(sample_rate, Config.SAMPLE_RATE)
    return resample_poly(audio, Config.SAMPLE_RATE // g, sample_rate // g)


def prepare_input(sample_rate: int, samples: np.ndarray) -> ModelInput:
    """Validate, convert and window one clip. Raises AudioInputError with a user-facing message."""
    if sample_rate <= 0:
        raise AudioInputError(f"Invalid sample rate {sample_rate}.")
    audio = to_float_mono(samples)
    seconds = len(audio) / sample_rate
    if seconds < Config.MIN_AUDIO_DURATION:
        raise AudioInputError(f"The clip is {seconds:.2f} s long; record at least "
                              f"{Config.MIN_AUDIO_DURATION:g} s.")
    if seconds > Config.MAX_AUDIO_DURATION:
        raise AudioInputError(f"The clip is {seconds:.1f} s long; the limit is "
                              f"{Config.MAX_AUDIO_DURATION:g} s (one word per clip).")
    audio = _resample(audio, sample_rate)
    speech_found = detect_speech(audio, Config.SAMPLE_RATE) is not None
    window = prepare_waveform(audio, Config.SAMPLE_RATE, Config.MAX_AUDIO_SAMPLES)
    return ModelInput(window=window, speech_found=speech_found)
