"""The step 1 report: tables and figures from saved ASR runs, per mic."""

import pandas as pd
import pytest

from src.baselines.asr.report import write_asr_report
from src.baselines.asr.pipeline import run_baseline
from src.eval.constants import ARRAY_MIC, HEAD_MIC
from tests.baselines.test_asr_pipeline import FakeTranscriber, clips_table, load_label


def saved_runs(tmp_path, models=("fake-a", "fake-b")):
    clips = pd.concat([clips_table(), clips_table().assign(mic=HEAD_MIC)], ignore_index=True)
    for model in models:
        run_baseline(model, clips, FakeTranscriber, tmp_path / "runs", batch_size=8,
                     load_window=load_label)
    return tmp_path / "runs"


def test_report_has_a_row_per_model_and_scorer_with_its_cost(tmp_path):
    out = tmp_path / "report"
    write_asr_report(saved_runs(tmp_path), out, ["fake-a", "fake-b"])
    table = pd.read_csv(out / "results.csv")
    assert table["model"].tolist() == ["fake-a-strict", "fake-a-lenient",
                                       "fake-b-strict", "fake-b-lenient"]
    assert set(table["mic"]) == {ARRAY_MIC}
    assert (table["params"] == 20).all()  # FakeTranscriber's nn.Linear(4, 4)
    for name in ("results.md", "per_speaker.csv", "accuracy_vs_macs.png",
                 "confusion_fake-a-lenient.png"):
        assert (out / name).exists()


def test_head_mic_is_reported_separately_never_combined(tmp_path):
    out = tmp_path / "report"
    write_asr_report(saved_runs(tmp_path), out, ["fake-a", "fake-b"])
    head = pd.read_csv(out / "head-mic" / "results.csv")
    assert set(head["mic"]) == {HEAD_MIC}
    assert (out / "head-mic" / "confusion_fake-b-strict.png").exists()


def test_missing_model_runs_are_named(tmp_path):
    with pytest.raises(FileNotFoundError, match="fake-c"):
        write_asr_report(saved_runs(tmp_path), tmp_path / "report", ["fake-a", "fake-c"])
