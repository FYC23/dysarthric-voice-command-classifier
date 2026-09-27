"""Tests for turning Gradio audio into the models' 2 s window."""

import numpy as np
import pytest

from src.config import Config
from src.demo.audio_input import AudioInputError, prepare_input, to_float_mono
from tests.conftest import word_clip

SR = Config.SAMPLE_RATE
WINDOW = Config.MAX_AUDIO_SAMPLES


def test_int16_is_scaled_to_unit_range():
    out = to_float_mono(np.array([16384, -32768, 0], dtype=np.int16))
    np.testing.assert_allclose(out, [0.5, -1.0, 0.0])


def test_int32_is_scaled_to_unit_range():
    out = to_float_mono(np.array([2 ** 30], dtype=np.int32))
    np.testing.assert_allclose(out, [0.5])


def test_stereo_samples_by_channels_is_averaged():
    left = np.array([0.2, 0.4, 0.6])
    out = to_float_mono(np.stack([left, np.zeros(3)], axis=1))  # (samples, channels)
    np.testing.assert_allclose(out, left / 2)


def test_stereo_channels_by_samples_is_averaged():
    left = np.array([0.2, 0.4, 0.6, 0.8])
    out = to_float_mono(np.stack([left, left]))  # (channels, samples)
    np.testing.assert_allclose(out, left)


def test_non_finite_samples_are_rejected():
    with pytest.raises(AudioInputError, match="NaN or infinite"):
        to_float_mono(np.array([0.1, np.nan, 0.2]))


def test_empty_audio_is_rejected():
    with pytest.raises(AudioInputError, match="empty"):
        to_float_mono(np.zeros(0, dtype=np.int16))


def test_window_is_32000_float32_samples():
    result = prepare_input(SR, word_clip(1.5))
    assert result.window.dtype == np.float32
    assert result.window.shape == (WINDOW,)
    assert result.speech_found


def test_48k_stereo_int16_microphone_input_is_recognised():
    clip = word_clip(2.0, sr=48000)
    pcm = (np.clip(clip, -1, 1) * 32767).astype(np.int16)
    result = prepare_input(48000, np.stack([pcm, pcm], axis=1))
    assert result.window.shape == (WINDOW,)
    assert result.speech_found


def test_word_near_the_end_of_a_long_clip_lands_in_the_window():
    audio = np.random.default_rng(0).normal(0, 1e-3, 6 * SR)
    t = np.arange(int(0.4 * SR)) / SR
    audio[5 * SR:5 * SR + len(t)] += 0.1 * np.sin(2 * np.pi * 300 * t)
    result = prepare_input(SR, audio)
    assert np.abs(result.window).max() > 0.05


def test_digital_silence_is_flagged_not_rejected():
    result = prepare_input(SR, np.zeros(SR, dtype=np.int16))
    assert not result.speech_found
    assert result.window.shape == (WINDOW,)
    assert np.isfinite(result.window).all()


def test_too_short_clip_is_rejected():
    with pytest.raises(AudioInputError, match="at least 0.1 s"):
        prepare_input(SR, np.zeros(int(0.05 * SR)))


def test_too_long_clip_is_rejected():
    with pytest.raises(AudioInputError, match="limit is 10 s"):
        prepare_input(SR, np.zeros(11 * SR))


def test_invalid_sample_rate_is_rejected():
    with pytest.raises(AudioInputError, match="sample rate"):
        prepare_input(0, np.zeros(SR))
