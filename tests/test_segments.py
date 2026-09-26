"""Tests for hand-labelled segment handling (long clips cropped to the labelled word)."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from conftest import word_clip, write_wav
from src.data.segments import apply_segment_labels, load_segment_labels, measure_kept_lengths

ROOT = Path("/data/TORGO")


def label_file(tmp_path, clips, **overrides):
    doc = {"format": "torgo-segment-labels", "version": 1, "window_s": 2.0, "clips": clips, **overrides}
    path = tmp_path / "labels.json"
    path.write_text(json.dumps(doc))
    return path


def row(rel, kept_s, label="left", speaker="M02", mic="wav_arrayMic"):
    return {"file_path": str(ROOT / rel), "speaker_id": speaker, "session": "Session1",
            "utterance_id": Path(rel).stem, "mic": mic, "label": label, "gender": "M",
            "is_dysarthric": True, "kept_s": kept_s}


LONG = "M/M02/Session1/wav_arrayMic/0072.wav"
SHORT = "M/M02/Session1/wav_arrayMic/0001.wav"


class TestLoadSegmentLabels:
    def test_missing_file_means_no_labels(self, tmp_path):
        assert load_segment_labels(tmp_path / "nope.json") == {}

    def test_reads_clips_keyed_by_relative_path(self, tmp_path):
        path = label_file(tmp_path, {LONG: {"status": "done", "segments": [
            {"start": 2.9, "end": 3.5, "type": "word"}]}})
        labels = load_segment_labels(path)
        assert labels[LONG]["status"] == "done"
        assert labels[LONG]["segments"][0]["type"] == "word"

    @pytest.mark.parametrize("overrides", [{"format": "something-else"}, {"version": 2}])
    def test_rejects_wrong_format_or_version(self, tmp_path, overrides):
        with pytest.raises(ValueError, match="format|version"):
            load_segment_labels(label_file(tmp_path, {}, **overrides))

    @pytest.mark.parametrize("seg", [
        {"start": 3.5, "end": 2.9, "type": "word"},
        {"start": -1, "end": 2.9, "type": "word"},
        {"start": 1.0, "end": 2.0, "type": "cough"},
    ], ids=["end-before-start", "negative", "unknown-type"])
    def test_rejects_invalid_segments(self, tmp_path, seg):
        with pytest.raises(ValueError, match=LONG):
            load_segment_labels(label_file(tmp_path, {LONG: {"status": "done", "segments": [seg]}}))

    def test_rejects_unknown_status(self, tmp_path):
        with pytest.raises(ValueError, match="status"):
            load_segment_labels(label_file(tmp_path, {LONG: {"status": "maybe", "segments": []}}))


class TestMeasureKeptLengths:
    def test_adds_kept_length_without_mutating_input(self, tmp_path):
        path = tmp_path / "clip.wav"
        write_wav(path, word_clip(seconds=2.0))  # 0.4 s burst in quiet noise
        df = pd.DataFrame([{"file_path": str(path)}])
        out = measure_kept_lengths(df)
        assert "kept_s" not in df.columns
        assert abs(out.kept_s.iloc[0] - (0.4 + 2 * 0.15)) < 0.1

    def test_no_speech_keeps_full_duration(self, tmp_path):
        path = tmp_path / "silence.wav"
        write_wav(path, np.random.default_rng(0).normal(0, 1e-3, 16000))
        out = measure_kept_lengths(pd.DataFrame([{"file_path": str(path)}]))
        assert out.kept_s.iloc[0] == pytest.approx(1.0, abs=1e-3)


class TestApplySegmentLabels:
    def test_unlabeled_clip_that_fits_is_kept_whole(self):
        result = apply_segment_labels(pd.DataFrame([row(SHORT, 0.8)]), {}, ROOT, 2.0)
        assert len(result.samples) == 1
        assert np.isnan(result.samples.seg_start.iloc[0])

    def test_unlabeled_clip_over_window_is_dropped_and_reported(self):
        result = apply_segment_labels(pd.DataFrame([row(LONG, 4.2)]), {}, ROOT, 2.0)
        assert len(result.samples) == 0
        assert result.dropped.reason.tolist() == ["unlabeled_over_window"]

    def test_labeled_clip_becomes_one_sample_per_word_segment(self):
        labels = {LONG: {"status": "done", "segments": [
            {"start": 0.0, "end": 2.1, "type": "non_speech"},
            {"start": 2.2, "end": 2.5, "type": "partial"},
            {"start": 2.9, "end": 3.5, "type": "word"},
            {"start": 3.8, "end": 4.1, "type": "word"},
        ]}}
        result = apply_segment_labels(pd.DataFrame([row(LONG, 4.2)]), labels, ROOT, 2.0)
        assert result.samples[["seg_start", "seg_end"]].values.tolist() == [[2.9, 3.5], [3.8, 4.1]]
        assert set(result.samples.label) == {"left"}
        assert result.non_speech[["seg_start", "seg_end"]].values.tolist() == [[0.0, 2.1]]
        assert result.non_speech.speaker_id.tolist() == ["M02"]

    def test_labels_also_apply_to_clips_that_fit(self):
        labels = {SHORT: {"status": "done", "segments": [{"start": 0.3, "end": 0.9, "type": "word"}]}}
        result = apply_segment_labels(pd.DataFrame([row(SHORT, 1.8)]), labels, ROOT, 2.0)
        assert result.samples[["seg_start", "seg_end"]].values.tolist() == [[0.3, 0.9]]

    def test_no_word_clip_is_dropped(self):
        labels = {LONG: {"status": "no_word", "segments": [{"start": 0.0, "end": 1.0, "type": "non_speech"}]}}
        result = apply_segment_labels(pd.DataFrame([row(LONG, 4.2)]), labels, ROOT, 2.0)
        assert len(result.samples) == 0
        assert result.dropped.reason.tolist() == ["no_word"]
        assert len(result.non_speech) == 1  # its non-speech is still usable

    def test_unfinished_labels_are_ignored(self):
        labels = {LONG: {"status": "todo", "segments": [{"start": 2.9, "end": 3.5, "type": "word"}]}}
        result = apply_segment_labels(pd.DataFrame([row(LONG, 4.2)]), labels, ROOT, 2.0)
        assert result.dropped.reason.tolist() == ["unlabeled_over_window"]

    def test_done_clip_without_word_segment_is_dropped(self):
        labels = {LONG: {"status": "done", "segments": [{"start": 0.0, "end": 1.0, "type": "non_speech"}]}}
        result = apply_segment_labels(pd.DataFrame([row(LONG, 4.2)]), labels, ROOT, 2.0)
        assert result.dropped.reason.tolist() == ["no_word_segment"]

    def test_does_not_mutate_input(self):
        df = pd.DataFrame([row(SHORT, 0.8), row(LONG, 4.2)])
        before = df.copy()
        apply_segment_labels(df, {}, ROOT, 2.0)
        pd.testing.assert_frame_equal(df, before)


# --- Evaluation clips and the shared loader ---------------------------------

from types import SimpleNamespace

from src.data.segments import evaluation_clips, load_torgo_samples
from src.eval.schema import CLIP_KEY


class TestEvaluationClips:
    def two_attempts(self):
        labels = {LONG: {"status": "done", "segments": [
            {"start": 3.8, "end": 4.1, "type": "word"},   # listed out of order on purpose
            {"start": 2.9, "end": 3.5, "type": "word"},
        ]}}
        rows = pd.DataFrame([row(SHORT, 0.8), row(LONG, 4.2)])
        return apply_segment_labels(rows, labels, ROOT, 2.0).samples

    def test_recording_with_two_attempts_keeps_the_first(self):
        samples = self.two_attempts()
        assert len(samples) == 3  # training keeps both attempts
        clips = evaluation_clips(samples)
        assert len(clips) == 2
        assert not clips.duplicated(list(CLIP_KEY)).any()
        long_row = clips[clips.utterance_id == "0072"].iloc[0]
        assert (long_row.seg_start, long_row.seg_end) == (2.9, 3.5)

    def test_other_rows_and_order_are_untouched(self):
        samples = self.two_attempts()
        clips = evaluation_clips(samples)
        assert clips.utterance_id.tolist() == ["0001", "0072"]
        assert np.isnan(clips.seg_start.iloc[0])

    def test_input_is_not_mutated(self):
        samples = self.two_attempts()
        before = samples.copy()
        evaluation_clips(samples)
        pd.testing.assert_frame_equal(samples, before)


def test_load_torgo_samples_scans_and_applies_labels(fake_torgo, tmp_path):
    cfg = SimpleNamespace(
        TORGO_ROOT=fake_torgo, TARGET_COMMANDS=["one", "two", "yes", "no", "up", "down",
                                                "left", "right"],
        MIC_TYPES=("wav_arrayMic", "wav_headMic"), MIN_AUDIO_DURATION=0.1,
        MAX_AUDIO_DURATION=10.0, TORGO_MANUAL_SEGMENT_LABELS=tmp_path / "none.json",
        MAX_AUDIO_LENGTH=2.0)
    result = load_torgo_samples(cfg)
    # See the fake_torgo docstring: 7 usable rows, all short enough to keep whole
    assert len(result.samples) == 7
    assert {"kept_s", "seg_start", "seg_end"} <= set(result.samples.columns)
    assert result.dropped.empty
