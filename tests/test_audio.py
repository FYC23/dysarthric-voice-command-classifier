"""Tests for the shared waveform preparation (DC removal, VAD trim, fixed window)."""

import numpy as np
import pytest

from src.audio import (
    VadParams, detect_speech, extract_word, fit_to_length, prepare_waveform, remove_dc, trim_silence,
)

SR = 16000


def noise(seconds, std=1e-3, seed=0):
    return np.random.default_rng(seed).normal(0, std, int(seconds * SR))


def with_burst(audio, start_s, dur_s, amp=0.1, freq=300.0):
    out = audio.copy()
    t = np.arange(int(dur_s * SR)) / SR
    s = int(start_s * SR)
    out[s:s + len(t)] += amp * np.sin(2 * np.pi * freq * t)
    return out


def assert_span(span, start_s, end_s, tol_s=0.06):
    assert span is not None
    start, end = span
    assert abs(start / SR - start_s) <= tol_s, f"start {start / SR:.3f}s vs {start_s}s"
    assert abs(end / SR - end_s) <= tol_s, f"end {end / SR:.3f}s vs {end_s}s"


class TestRemoveDc:
    def test_removes_constant_offset(self):
        audio = noise(1.0) + 0.08
        assert abs(remove_dc(audio).mean()) < 1e-3

    def test_does_not_mutate_input(self):
        audio = noise(1.0) + 0.08
        before = audio.copy()
        remove_dc(audio)
        np.testing.assert_array_equal(audio, before)


class TestDetectSpeech:
    def test_finds_burst_in_quiet_noise(self):
        audio = with_burst(noise(3.0), 1.0, 0.4)
        assert_span(detect_speech(audio, SR), 1.0, 1.4)

    def test_dc_offset_does_not_break_detection(self):
        # Speech Commands has clips sitting at +0.04..0.08; the old top_db trim failed on these
        audio = with_burst(noise(3.0), 1.0, 0.4) + 0.08
        assert_span(detect_speech(audio, SR), 1.0, 1.4)

    def test_low_dynamic_range_recording(self):
        # ~20 dB between speech and noise floor, like M05's array mic
        audio = with_burst(noise(3.0, std=7e-3), 1.0, 0.4, amp=0.1)
        assert_span(detect_speech(audio, SR), 1.0, 1.4)

    def test_ignores_short_click(self):
        # lip smack / mouse click: loud, zero-mean, broadband, 10 ms
        audio = with_burst(noise(3.0), 1.5, 0.4)
        click = np.random.default_rng(1).normal(0, 0.5, int(0.01 * SR))
        audio[int(0.3 * SR):int(0.3 * SR) + len(click)] += click
        assert_span(detect_speech(audio, SR), 1.5, 1.9)

    def test_ignores_low_frequency_ramp_at_file_start(self):
        # Some TORGO sessions start with a decaying offset (recorder settling)
        audio = with_burst(noise(3.0), 1.5, 0.4)
        audio[:SR] += 0.2 * np.exp(-np.arange(SR) / (0.1 * SR))
        assert_span(detect_speech(audio, SR), 1.5, 1.9)

    def test_keeps_every_segment_of_a_multi_part_utterance(self):
        # Dysarthric clips often have two voiced parts (e.g. a repeated attempt)
        audio = with_burst(with_burst(noise(3.0), 0.5, 0.3), 1.6, 0.3)
        assert_span(detect_speech(audio, SR), 0.5, 1.9)

    @pytest.mark.parametrize("audio", [noise(2.0), np.zeros(SR * 2), np.zeros(100)],
                             ids=["noise-only", "digital-silence", "shorter-than-frame"])
    def test_returns_none_without_speech(self, audio):
        assert detect_speech(audio, SR) is None


class TestTrimSilence:
    def test_trims_to_speech_plus_padding(self):
        params = VadParams(pad_s=0.15)
        audio = with_burst(noise(3.0), 1.0, 0.4)
        trimmed = trim_silence(audio, SR, params)
        assert abs(len(trimmed) / SR - (0.4 + 2 * 0.15)) <= 0.1

    def test_padding_is_clipped_at_file_edges(self):
        audio = with_burst(noise(1.0), 0.02, 0.4)
        trimmed = trim_silence(audio, SR, VadParams(pad_s=0.15))
        assert len(trimmed) <= len(audio)

    def test_returns_unchanged_copy_when_no_speech(self):
        audio = noise(2.0)
        trimmed = trim_silence(audio, SR)
        np.testing.assert_array_equal(trimmed, audio)
        assert trimmed is not audio


class TestFitToLength:
    def test_pads_short_audio_centred(self):
        out = fit_to_length(np.ones(100), 300)
        assert len(out) == 300
        np.testing.assert_array_equal(out[:100], 0)
        np.testing.assert_array_equal(out[100:200], 1)
        np.testing.assert_array_equal(out[200:], 0)

    def test_keeps_loudest_window_of_long_audio(self):
        audio = with_burst(noise(3.0), 2.5, 0.3)
        out = fit_to_length(audio, SR)
        assert len(out) == SR
        # the whole burst (energy 0.1^2/2 * 0.3 s) must be inside the window
        assert np.sum(out ** 2) >= 0.95 * np.sum(audio[int(2.5 * SR):int(2.8 * SR)] ** 2)

    def test_exact_length_is_returned_as_copy(self):
        audio = noise(1.0)
        out = fit_to_length(audio, len(audio))
        np.testing.assert_array_equal(out, audio)
        assert out is not audio


class TestPrepareWaveform:
    @pytest.mark.parametrize("seconds", [0.5, 2.0, 6.0])
    def test_output_has_exact_length_and_float32(self, seconds):
        audio = with_burst(noise(seconds), seconds / 3, min(0.3, seconds / 3))
        out = prepare_waveform(audio, SR, 24000)
        assert out.shape == (24000,)
        assert out.dtype == np.float32

    def test_word_survives_when_file_is_mostly_silence(self):
        # 5 s file, word near the end: the old centre crop would have cut it off
        audio = with_burst(noise(5.0), 4.3, 0.4)
        out = prepare_waveform(audio, SR, 24000)
        burst_energy = np.sum(audio[int(4.3 * SR):int(4.7 * SR)] ** 2)
        assert np.sum(out.astype(np.float64) ** 2) >= 0.95 * burst_energy

    def test_does_not_mutate_input(self):
        audio = with_burst(noise(2.0), 0.8, 0.4) + 0.05
        before = audio.copy()
        prepare_waveform(audio, SR, 24000)
        np.testing.assert_array_equal(audio, before)

    def test_labelled_segment_replaces_vad(self):
        # loud struggle sound first, quieter word later: the label must win
        audio = with_burst(with_burst(noise(4.0), 0.5, 1.0, amp=0.5), 3.0, 0.4, amp=0.1)
        out = prepare_waveform(audio, SR, 32000, segment=(2.95, 3.45)).astype(np.float64)
        word = np.sum(audio[int(3.0 * SR):int(3.4 * SR)] ** 2)
        assert len(out) == 32000
        assert np.sum(out ** 2) == pytest.approx(word, rel=0.1)  # word in, struggle out

    def test_labelled_segment_gets_padding(self):
        audio = with_burst(noise(4.0), 3.0, 0.4)
        out = prepare_waveform(audio, SR, 32000, segment=(3.0, 3.4), params=VadParams(pad_s=0.15))
        # 0.4 s word + 2 x 0.15 s padding, centred in the 2 s window: non-zero span ~0.7 s
        nonzero = np.flatnonzero(out)
        assert abs((nonzero[-1] - nonzero[0]) / SR - 0.7) < 0.02


class TestExtractWord:
    def test_trims_silence_and_keeps_the_word(self):
        audio = with_burst(noise(2.0), 0.8, 0.4)
        word = extract_word(audio, SR)
        assert word.dtype == np.float32
        # 0.4 s burst plus the VAD's 0.15 s context either side
        assert abs(len(word) / SR - 0.7) < 0.06

    def test_prepare_waveform_is_extract_word_then_fit(self):
        audio = with_burst(noise(2.0), 0.3, 0.4)
        expected = fit_to_length(extract_word(audio, SR), 2 * SR)
        np.testing.assert_array_equal(prepare_waveform(audio, SR, 2 * SR), expected)

    def test_segment_replaces_the_vad(self):
        audio = with_burst(noise(2.0), 0.8, 0.4)
        word = extract_word(audio, SR, segment=(1.0, 1.1))
        pad = VadParams().pad_s
        assert len(word) == int((1.1 + pad) * SR) - int((1.0 - pad) * SR)

    def test_invalid_segment_raises(self):
        with pytest.raises(ValueError, match="invalid segment"):
            extract_word(noise(1.0), SR, segment=(0.5, 0.2))

    def test_segment_past_the_end_gives_an_empty_word(self):
        word = extract_word(noise(1.0), SR, segment=(3.0, 3.5))
        assert len(word) == 0 and word.dtype == np.float32

    def test_does_not_mutate_input(self):
        audio = with_burst(noise(2.0), 0.8, 0.4) + 0.05
        before = audio.copy()
        extract_word(audio, SR)
        np.testing.assert_array_equal(audio, before)
