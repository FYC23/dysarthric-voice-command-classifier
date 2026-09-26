"""Tests for TORGO scanning and labelling."""

from src.data.preprocessing import scan_torgo_dataset

TARGETS = ["one", "two", "yes", "no", "up", "down", "left", "right"]
BOTH_MICS = ("wav_arrayMic", "wav_headMic")


def rows(df):
    return {(r.speaker_id, r.utterance_id, r.mic, r.label) for r in df.itertuples()}


def test_labels_only_exact_single_word_prompts_and_uses_both_mics(fake_torgo):
    df = scan_torgo_dataset(fake_torgo, TARGETS, BOTH_MICS)
    assert rows(df) == {
        ("F01", "0001", "wav_arrayMic", "one"),
        ("F01", "0001", "wav_headMic", "one"),
        ("F01", "0003", "wav_arrayMic", "yes"),
        ("F01", "0005", "wav_headMic", "up"),
        ("F01", "0006", "wav_headMic", "down"),
        ("F01", "0009", "wav_arrayMic", "two"),
        ("FC01", "0001", "wav_arrayMic", "no"),
    }


def test_sentence_starting_with_a_command_word_is_excluded(fake_torgo):
    df = scan_torgo_dataset(fake_torgo, TARGETS, BOTH_MICS)
    assert "0002" not in set(df[df.speaker_id == "F01"].utterance_id)


def test_unreadable_and_too_short_files_are_skipped(fake_torgo):
    df = scan_torgo_dataset(fake_torgo, TARGETS, BOTH_MICS)
    f01 = df[df.speaker_id == "F01"]
    assert not ((f01.utterance_id == "0007")).any()
    assert set(f01[f01.utterance_id == "0006"].mic) == {"wav_headMic"}


def test_implausibly_long_files_are_skipped(fake_torgo):
    df = scan_torgo_dataset(fake_torgo, TARGETS, BOTH_MICS, max_duration_s=10.0)
    assert set(df[df.utterance_id == "0009"].mic) == {"wav_arrayMic"}


def test_mic_selection_is_respected(fake_torgo):
    df = scan_torgo_dataset(fake_torgo, TARGETS, ("wav_arrayMic",))
    assert set(df.mic) == {"wav_arrayMic"}
    assert len(df) == 4


def test_speaker_metadata(fake_torgo):
    df = scan_torgo_dataset(fake_torgo, TARGETS, BOTH_MICS)
    meta = df.drop_duplicates("speaker_id").set_index("speaker_id")
    assert bool(meta.loc["F01", "is_dysarthric"]) is True
    assert bool(meta.loc["FC01", "is_dysarthric"]) is False
    assert meta.loc["F01", "gender"] == "F"
    assert set(df.session) == {"Session1"}


def test_empty_result_keeps_columns(fake_torgo):
    df = scan_torgo_dataset(fake_torgo, ["menu"], BOTH_MICS)
    assert len(df) == 0
    assert {"file_path", "speaker_id", "mic", "label", "is_dysarthric"} <= set(df.columns)
