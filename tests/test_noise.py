"""Tests for the background-noise bank and SNR mixing."""

import numpy as np
import pytest
import soundfile as sf

from src.data.noise import NoiseBank, mean_power, mix_at_snr

SR = 16000


def ramp_bank(*lengths):
    """Recording i holds values in [i, i + 1), rising by 1/n per sample, so any sample can be traced."""
    return NoiseBank([i + np.arange(n, dtype=np.float32) / n for i, n in enumerate(lengths)])


class TestNoiseBank:
    def test_sample_is_a_contiguous_segment_of_one_recording(self):
        bank = ramp_bank(1000, 2000)
        rng = np.random.default_rng(0)
        for _ in range(20):
            seg = bank.sample(100, rng)
            assert seg.shape == (100,) and seg.dtype == np.float32
            rec = int(np.floor(seg[0]))
            assert np.all(np.floor(seg) == rec)
            np.testing.assert_allclose(np.diff(seg), 1 / (1000, 2000)[rec], rtol=1e-3)

    def test_sample_draws_from_every_recording(self):
        bank = ramp_bank(1000, 2000)
        rng = np.random.default_rng(0)
        assert {int(np.floor(bank.sample(10, rng)[0])) for _ in range(50)} == {0, 1}

    def test_short_recording_is_tiled(self):
        bank = NoiseBank([np.array([1.0, 2.0, 3.0])])
        seg = bank.sample(10, np.random.default_rng(0))
        assert len(seg) == 10
        assert set(np.diff(seg)) <= {1.0, -2.0}  # 1 -> 2 -> 3 -> 1 ...

    def test_sample_is_reproducible_and_a_copy(self):
        bank = ramp_bank(1000)
        first = bank.sample(50, np.random.default_rng(7))
        expected = first.copy()
        first[:] = 99.0
        np.testing.assert_array_equal(bank.sample(50, np.random.default_rng(7)), expected)

    def test_rejects_empty_bank(self):
        with pytest.raises(ValueError, match="at least one"):
            NoiseBank([])

    def test_rejects_empty_or_multichannel_recordings(self):
        with pytest.raises(ValueError, match="mono"):
            NoiseBank([np.zeros(0)])
        with pytest.raises(ValueError, match="mono"):
            NoiseBank([np.zeros((10, 2))])

    def test_rejects_non_positive_length(self):
        with pytest.raises(ValueError, match="length"):
            ramp_bank(100).sample(0, np.random.default_rng(0))

    def test_from_dir_loads_every_wav(self, tmp_path):
        rng = np.random.default_rng(0)
        for name in ("a.wav", "b.wav"):
            sf.write(tmp_path / name, rng.normal(0, 0.1, SR), SR)
        (tmp_path / "notes.txt").write_text("not audio")
        assert len(NoiseBank.from_dir(tmp_path, SR)) == 2

    def test_from_dir_missing_files_says_how_to_fix(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="download_speech_commands"):
            NoiseBank.from_dir(tmp_path / "missing", SR)

    def test_from_dir_rejects_wrong_sample_rate(self, tmp_path):
        sf.write(tmp_path / "a.wav", np.zeros(8000), 8000)
        with pytest.raises(ValueError, match="8000"):
            NoiseBank.from_dir(tmp_path, SR)


class TestMixAtSnr:
    def test_hits_the_target_snr_against_the_reference_power(self):
        rng = np.random.default_rng(0)
        signal = rng.normal(0, 0.1, SR).astype(np.float32)
        noise = rng.normal(0, 0.3, SR).astype(np.float32)
        mixed = mix_at_snr(signal, noise, snr_db=10.0, ref_power=0.01)
        added = mixed.astype(np.float64) - signal
        assert abs(10 * np.log10(0.01 / mean_power(added)) - 10.0) < 1e-3
        assert mixed.dtype == np.float32

    def test_silent_noise_leaves_signal_unchanged(self):
        signal = np.full(100, 0.1, dtype=np.float32)
        np.testing.assert_array_equal(mix_at_snr(signal, np.zeros(100), 10.0, 0.01), signal)

    def test_silent_reference_leaves_signal_unchanged(self):
        signal = np.zeros(100, dtype=np.float32)
        noise = np.ones(100, dtype=np.float32)
        np.testing.assert_array_equal(mix_at_snr(signal, noise, 10.0, 0.0), signal)

    def test_does_not_mutate_inputs(self):
        signal, noise = np.full(100, 0.1), np.full(100, 0.2)
        mix_at_snr(signal, noise, 10.0, 0.01)
        assert np.all(signal == 0.1) and np.all(noise == 0.2)

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="shape"):
            mix_at_snr(np.zeros(10), np.zeros(11), 10.0, 0.01)

    def test_mean_power_of_empty_is_zero(self):
        assert mean_power(np.zeros(0)) == 0.0
