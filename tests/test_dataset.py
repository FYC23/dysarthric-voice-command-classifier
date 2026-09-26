"""Integration test: scan -> dataset item, with and without augmentation."""

import numpy as np
import pytest
from transformers import Wav2Vec2FeatureExtractor

from src.config import config
from src.data.dataset import TORGOCommandDataset
from src.data.preprocessing import create_label_mapping, scan_torgo_dataset


@pytest.fixture
def df(fake_torgo):
    frame = scan_torgo_dataset(fake_torgo, ["one", "yes", "no", "up", "down"],
                               ("wav_arrayMic", "wav_headMic"))
    label2id, _ = create_label_mapping(frame)
    return frame.assign(label_id=frame.label.map(label2id))


@pytest.mark.parametrize("augment", [False, True])
def test_items_have_fixed_length_and_label(df, augment):
    ds = TORGOCommandDataset(df, Wav2Vec2FeatureExtractor(), config,
                             max_length=config.MAX_AUDIO_SAMPLES, augment=augment)
    for i in range(len(ds)):
        item = ds[i]
        assert item["input_values"].shape == (config.MAX_AUDIO_SAMPLES,)
        assert int(item["label"]) == df.iloc[i].label_id


def test_labelled_segment_rows_are_cropped_to_the_segment(df):
    labelled = df.assign(seg_start=np.nan, seg_end=np.nan)
    labelled.loc[0, ["seg_start", "seg_end"]] = [0.8, 1.2]  # the word in word_clip()
    ds = TORGOCommandDataset(labelled, Wav2Vec2FeatureExtractor(do_normalize=False), config,
                             max_length=config.MAX_AUDIO_SAMPLES, augment=False)
    values = ds[0]["input_values"].numpy()
    assert values.shape == (config.MAX_AUDIO_SAMPLES,)
    nonzero = np.flatnonzero(values)
    assert abs((nonzero[-1] - nonzero[0]) / config.SAMPLE_RATE - (0.4 + 2 * 0.15)) < 0.02


def test_window_is_two_seconds():
    assert config.MAX_AUDIO_SAMPLES == int(2.0 * config.SAMPLE_RATE)
