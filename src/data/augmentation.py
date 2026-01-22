"""
Audio augmentation pipeline using audiomentations.
"""

import random
import numpy as np
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift, Gain


def create_augmentation_pipeline(config) -> Compose:
    """
    Create audiomentations augmentation pipeline based on config settings.
    """
    return Compose([
        AddGaussianNoise(
            min_amplitude=config.AUG_NOISE_MIN_AMP,
            max_amplitude=config.AUG_NOISE_MAX_AMP,
            p=config.AUG_NOISE_PROB
        ),
        PitchShift(
            min_semitones=config.AUG_PITCH_MIN_SEMITONES,
            max_semitones=config.AUG_PITCH_MAX_SEMITONES,
            p=config.AUG_PITCH_PROB
        ),
        TimeStretch(
            min_rate=config.AUG_TIMESTRETCH_MIN,
            max_rate=config.AUG_TIMESTRETCH_MAX,
            p=config.AUG_TIMESTRETCH_PROB
        ),
        Shift(
            min_shift=config.AUG_SHIFT_MIN,
            max_shift=config.AUG_SHIFT_MAX,
            p=config.AUG_SHIFT_PROB
        ),
        Gain(
            min_gain_db=config.AUG_GAIN_MIN_DB,
            max_gain_db=config.AUG_GAIN_MAX_DB,
            p=config.AUG_GAIN_PROB
        ),
    ])


def apply_spec_augment(waveform: np.ndarray, config) -> np.ndarray:
    """
    Apply SpecAugment-style time masking to waveform.
    """
    waveform = waveform.copy()
    length = len(waveform)
    
    # Time masking: zero out random contiguous segments
    for _ in range(config.SPEC_AUG_NUM_TIME_MASKS):
        t = random.randint(0, min(config.SPEC_AUG_TIME_MASK_PARAM, length // 10))
        t0 = random.randint(0, max(0, length - t))
        waveform[t0:t0 + t] = 0
    
    return waveform