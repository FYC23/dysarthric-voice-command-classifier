"""Tests for the TORGO training-time waveform augmentation."""

import numpy as np
import pytest

from src.audio import fit_to_length
from src.data.torgo_augment import (
    TorgoAugParams, apply_gain, augment_word, place_in_window, speed_perturb,
)
from src.data.noise import NoiseBank, mean_power

SR = 16000
WINDOW = 2 * SR
QUIET = TorgoAugParams(speed_factors=(1.0,), noise_prob=0.0, gain_db=(0.0, 0.0))


def sine(freq, seconds, amp=0.1):
    t = np.arange(int(seconds * SR)) / SR
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def dominant_hz(audio):
    spectrum = np.abs(np.fft.rfft(audio))
    return np.fft.rfftfreq(len(audio), 1 / SR)[np.argmax(spectrum)]


@pytest.fixture
def noise_bank():
    return NoiseBank([np.random.default_rng(0).normal(0, 0.05, 3 * SR)])


class TestSpeedPerturb:
    @pytest.mark.parametrize("factor", [0.9, 0.95, 1.05, 1.1])
    def test_duration_scales_by_inverse_factor(self, factor):
        out = speed_perturb(sine(300, 1.0), factor)
        assert abs(len(out) - SR / factor) <= 2
        assert out.dtype == np.float32

    def test_pitch_scales_by_factor(self):
        assert abs(dominant_hz(speed_perturb(sine(300, 1.0), 1.1)) - 330) < 3
        assert abs(dominant_hz(speed_perturb(sine(300, 1.0), 0.9)) - 270) < 3

    def test_factor_one_is_an_exact_copy(self):
        audio = sine(300, 0.5)
        out = speed_perturb(audio, 1.0)
        np.testing.assert_array_equal(out, audio)
        assert out is not audio

    def test_empty_audio_stays_empty(self):
        assert len(speed_perturb(np.zeros(0, dtype=np.float32), 0.9)) == 0

    def test_non_positive_factor_raises(self):
        with pytest.raises(ValueError, match="factor"):
            speed_perturb(sine(300, 0.1), 0.0)


class TestPlaceInWindow:
    def test_word_is_intact_at_some_offset(self):
        word = np.full(SR // 2, 0.1, dtype=np.float32)
        out = place_in_window(word, WINDOW, np.random.default_rng(0))
        nonzero = np.flatnonzero(out)
        assert out.shape == (WINDOW,) and out.dtype == np.float32
        np.testing.assert_array_equal(out[nonzero[0]:nonzero[-1] + 1], word)

    def test_offsets_cover_the_whole_room(self):
        word = np.full(SR // 2, 0.1, dtype=np.float32)
        room = WINDOW - len(word)
        rng = np.random.default_rng(0)
        offsets = [np.flatnonzero(place_in_window(word, WINDOW, rng))[0] for _ in range(200)]
        assert min(offsets) < 0.1 * room and max(offsets) > 0.9 * room
        assert max(offsets) <= room  # never cut

    def test_word_longer_than_window_keeps_the_loudest_part(self):
        word = np.concatenate([np.full(WINDOW, 0.01), np.full(SR // 4, 0.5)]).astype(np.float32)
        np.testing.assert_array_equal(
            place_in_window(word, WINDOW, np.random.default_rng(0)), fit_to_length(word, WINDOW))


class TestApplyGain:
    def test_plus_six_db_roughly_doubles(self):
        out = apply_gain(np.array([0.1, -0.2], dtype=np.float32), 6.0, 0.99)
        np.testing.assert_allclose(out, [0.1995, -0.399], rtol=1e-3)

    def test_rescales_instead_of_clipping(self):
        out = apply_gain(np.array([0.8, -0.4], dtype=np.float32), 6.0, 0.99)
        assert np.max(np.abs(out)) == pytest.approx(0.99)
        assert out[1] / out[0] == pytest.approx(-0.5)  # shape kept, nothing clipped


class TestAugmentWord:
    def test_output_is_a_fixed_float32_window(self, noise_bank):
        out = augment_word(sine(300, 0.6), WINDOW, np.random.default_rng(0), noise_bank)
        assert out.shape == (WINDOW,) and out.dtype == np.float32
        assert np.all(np.isfinite(out)) and np.max(np.abs(out)) <= 0.99 + 1e-6

    def test_same_seed_same_view_and_different_seeds_differ(self, noise_bank):
        word = sine(300, 0.6)
        a = augment_word(word, WINDOW, np.random.default_rng(1), noise_bank)
        b = augment_word(word, WINDOW, np.random.default_rng(1), noise_bank)
        c = augment_word(word, WINDOW, np.random.default_rng(2), noise_bank)
        np.testing.assert_array_equal(a, b)
        assert not np.array_equal(a, c)

    def test_does_not_mutate_the_word(self, noise_bank):
        word = sine(300, 0.6)
        before = word.copy()
        augment_word(word, WINDOW, np.random.default_rng(0), noise_bank)
        np.testing.assert_array_equal(word, before)

    def test_noise_fills_the_whole_window_at_snr_against_the_word(self, noise_bank):
        word = np.full(SR // 2, 0.1, dtype=np.float32)  # power 0.01, a quarter of the window
        params = TorgoAugParams(speed_factors=(1.0,), noise_prob=1.0, snr_db=(10.0, 10.0),
                                gain_db=(0.0, 0.0))
        out = augment_word(word, WINDOW, np.random.default_rng(0), noise_bank, params)
        assert np.count_nonzero(out) == WINDOW
        # word energy spread over the window (0.0025) + noise at 10 dB below 0.01 (0.001);
        # measuring SNR against the padded window would give 0.0025 + 0.00025 instead
        assert mean_power(out) == pytest.approx(0.0035, abs=2e-4)

    def test_without_noise_the_word_is_only_moved(self):
        word = np.full(SR // 2, 0.1, dtype=np.float32)
        out = augment_word(word, WINDOW, np.random.default_rng(0), None, QUIET)
        assert np.count_nonzero(out) == len(word)

    def test_noise_prob_needs_a_noise_bank(self):
        with pytest.raises(ValueError, match="NoiseBank"):
            augment_word(sine(300, 0.5), WINDOW, np.random.default_rng(0), None)

    def test_empty_word_gives_a_silent_window(self, noise_bank):
        out = augment_word(np.zeros(0, dtype=np.float32), WINDOW, np.random.default_rng(0),
                           noise_bank)
        assert out.shape == (WINDOW,) and not np.any(out)

    def test_silent_word_gets_no_noise(self, noise_bank):
        out = augment_word(np.zeros(SR // 2, dtype=np.float32), WINDOW,
                           np.random.default_rng(0), noise_bank)
        assert not np.any(out)

    def test_word_longer_than_window_after_slowdown(self, noise_bank):
        word = sine(300, 1.83)  # TORGO's longest trimmed word; 0.9x makes it 2.03 s
        params = TorgoAugParams(speed_factors=(0.9,))
        out = augment_word(word, WINDOW, np.random.default_rng(0), noise_bank, params)
        assert out.shape == (WINDOW,) and np.all(np.isfinite(out))


class TestTorgoAugParams:
    @pytest.mark.parametrize("kwargs", [
        {"speed_factors": ()},
        {"speed_factors": (0.9, -1.0)},
        {"noise_prob": 1.5},
        {"snr_db": (20.0, 5.0)},
        {"gain_db": (6.0, -6.0)},
        {"peak_limit": 0.0},
    ])
    def test_rejects_invalid_settings(self, kwargs):
        with pytest.raises(ValueError):
            TorgoAugParams(**kwargs)
