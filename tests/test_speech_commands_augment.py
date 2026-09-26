"""Tests for BC-ResNet's Speech Commands augmentation (reference-code behaviour)."""

import numpy as np
import pytest

from src.data.noise import NoiseBank
from src.data.speech_commands_augment import (
    BCResNetAugParams, augment_command, augment_word_in_window, make_silence, shift_with_zeros,
)

SR = 16000
ALWAYS = BCResNetAugParams(noise_prob=1.0)


def impulse_at(index, length=SR):
    clip = np.zeros(length, dtype=np.float32)
    clip[index] = 0.5
    return clip


@pytest.fixture
def silent_bank():
    return NoiseBank([np.zeros(3 * SR)])


@pytest.fixture
def unit_bank():
    return NoiseBank([np.ones(3 * SR)])


class TestShiftWithZeros:
    def test_positive_shift_moves_audio_earlier(self):
        np.testing.assert_array_equal(shift_with_zeros(np.array([1., 2., 3., 4.]), 1), [2, 3, 4, 0])

    def test_negative_shift_moves_audio_later(self):
        np.testing.assert_array_equal(shift_with_zeros(np.array([1., 2., 3., 4.]), -1), [0, 1, 2, 3])

    def test_zero_shift_is_a_copy(self):
        audio = np.array([1., 2.], dtype=np.float32)
        out = shift_with_zeros(audio, 0)
        np.testing.assert_array_equal(out, audio)
        assert out is not audio

    def test_shift_past_the_end_is_silence(self):
        assert not np.any(shift_with_zeros(np.ones(4), 4))


class TestAugmentCommand:
    def test_shift_stays_within_100_ms(self, silent_bank):
        rng = np.random.default_rng(0)
        moves = [np.flatnonzero(augment_command(impulse_at(8000), SR, rng, silent_bank, ALWAYS))[0]
                 - 8000 for _ in range(200)]
        assert max(abs(m) for m in moves) <= 0.1 * SR
        assert min(moves) < -0.08 * SR and max(moves) > 0.08 * SR

    def test_skipped_with_probability_one_minus_noise_prob(self, silent_bank):
        clip = impulse_at(8000)
        never = BCResNetAugParams(noise_prob=0.0)
        np.testing.assert_array_equal(
            augment_command(clip, SR, np.random.default_rng(0), silent_bank, never), clip)
        rng = np.random.default_rng(0)
        changed = sum(not np.array_equal(augment_command(clip, SR, rng, silent_bank), clip)
                      for _ in range(1000))
        assert 740 < changed < 860  # ~0.8, minus the rare zero-sample shift

    def test_noise_amplitude_is_at_most_0_1(self, unit_bank):
        rng = np.random.default_rng(0)
        for _ in range(50):
            out = augment_command(np.zeros(SR, dtype=np.float32), SR, rng, unit_bank, ALWAYS)
            assert np.max(out) <= 0.1

    def test_result_is_clamped(self, unit_bank):
        loud = np.full(SR, 0.99, dtype=np.float32)
        out = augment_command(loud, SR, np.random.default_rng(0), unit_bank, ALWAYS)
        assert np.max(out) <= 1.0 and out.dtype == np.float32

    def test_does_not_mutate_the_clip(self, unit_bank):
        clip = impulse_at(8000)
        augment_command(clip, SR, np.random.default_rng(0), unit_bank, ALWAYS)
        assert clip[8000] == 0.5 and np.count_nonzero(clip) == 1


class TestMakeSilence:
    def test_is_noise_at_amplitude_up_to_one(self, unit_bank):
        rng = np.random.default_rng(0)
        amps = [make_silence(SR, rng, unit_bank)[0] for _ in range(200)]
        assert 0.0 <= min(amps) < 0.1 and 0.9 < max(amps) <= 1.0

    def test_has_the_requested_length(self, unit_bank):
        assert make_silence(2 * SR, np.random.default_rng(0), unit_bank).shape == (2 * SR,)


WINDOW = 32000


class TestAugmentWordInWindow:
    WORD = np.linspace(0.1, 0.2, 8000, dtype=np.float32)
    SILENT_NOISE = NoiseBank([np.zeros(WINDOW, np.float32)])
    NO_NOISE = BCResNetAugParams(noise_prob=0.0)
    ALWAYS_NOISE = BCResNetAugParams(noise_prob=1.0)

    def test_word_lands_whole_somewhere_in_the_window(self):
        for seed in range(20):
            out = augment_word_in_window(self.WORD, WINDOW, np.random.default_rng(seed),
                                         self.SILENT_NOISE, self.NO_NOISE)
            start = int(np.flatnonzero(out)[0])
            assert len(out) == WINDOW and out.dtype == np.float32
            np.testing.assert_array_equal(out[start:start + len(self.WORD)], self.WORD)
            assert np.count_nonzero(out) == len(self.WORD)

    def test_placement_varies_between_draws(self):
        starts = {int(np.flatnonzero(augment_word_in_window(
            self.WORD, WINDOW, np.random.default_rng(seed), self.SILENT_NOISE, self.NO_NOISE))[0])
            for seed in range(10)}
        assert len(starts) > 1

    def test_noise_covers_the_whole_window_not_just_the_word(self):
        noise = NoiseBank([np.full(WINDOW, 0.5, np.float32)])
        out = augment_word_in_window(self.WORD, WINDOW, np.random.default_rng(0), noise,
                                     self.ALWAYS_NOISE)
        assert np.all(out != 0)

    def test_result_is_clamped(self):
        noise = NoiseBank([np.full(WINDOW, 50.0, np.float32)])
        out = augment_word_in_window(np.ones(8000, np.float32), WINDOW,
                                     np.random.default_rng(0), noise, self.ALWAYS_NOISE)
        assert out.max() <= 1.0 and out.min() >= -1.0

    def test_word_longer_than_the_window_comes_back_exactly_one_window(self):
        long_word = np.full(WINDOW + 500, 0.1, np.float32)
        out = augment_word_in_window(long_word, WINDOW, np.random.default_rng(0),
                                     self.SILENT_NOISE, self.NO_NOISE)
        assert len(out) == WINDOW

    def test_does_not_mutate_the_word(self):
        word = self.WORD.copy()
        augment_word_in_window(word, WINDOW, np.random.default_rng(0),
                               NoiseBank([np.full(WINDOW, 0.5, np.float32)]), self.ALWAYS_NOISE)
        np.testing.assert_array_equal(word, self.WORD)
