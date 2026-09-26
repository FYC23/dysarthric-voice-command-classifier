"""Tests for the results tables, their formatting, and the figures."""

import pandas as pd
import pytest

from src.eval.aggregate import summarize
from src.eval.constants import HEAD_MIC
from src.eval.cost import CostProfile
from src.eval.plots import plot_accuracy_vs_macs, plot_confusion
from src.eval.report import (
    human_count, per_speaker_table, results_table, to_markdown, write_report,
)
from src.eval.schema import Run
from tests.eval.factories import ALL_DYSARTHRIC, dysarthric_preds, loso_folds, speaker_rows


def summary(model, k, zero_shot=False):
    """Every dysarthric speaker gets k of 10 right, 3 seeds (1 run if zero-shot)."""
    correct = {s: k for s in ALL_DYSARTHRIC}
    if zero_shot:
        controls = pd.DataFrame(speaker_rows("FC01", 10, 9))
        preds = pd.concat([dysarthric_preds(correct), controls], ignore_index=True)
        return summarize([Run(model=model, seed=0, predictions=preds, zero_shot=True)])
    return summarize([Run(model=model, seed=i, predictions=dysarthric_preds(correct),
                          fold_train_speakers=loso_folds()) for i in range(3)])


COSTS = {
    "bcresnet1": CostProfile(params=9232, macs=4_900_000, input_seconds=2.0),
    "whisper": CostProfile(params=39_000_000, macs=1_200_000_000, input_seconds=30.0,
                           note="encoder padded to 30 s"),
}


@pytest.mark.parametrize("n, text", [
    (950, "950"), (9232, "9.2k"), (321_068, "321k"), (3_100_000, "3.1M"),
    (89_100_000, "89.1M"), (315_000_000, "315M"), (1_200_000_000, "1.2G"),
    (999.6, "1.0k"), (99_960, "100k"), (999_950, "1.0M"),  # rounding crosses a unit
])
def test_human_count(n, text):
    assert human_count(n) == text


def test_results_table_has_one_row_per_model_with_accuracy_and_cost():
    table = results_table([summary("bcresnet1", 6), summary("whisper", 3, zero_shot=True)], COSTS)
    row = table.set_index("model").loc["bcresnet1"]
    assert row["accuracy"] == pytest.approx(0.6)
    assert row["params"] == 9232
    assert row["macs"] == 4_900_000
    assert row["n_seeds"] == 3
    assert list(table["model"]) == ["bcresnet1", "whisper"]


def test_control_accuracy_appears_only_for_zero_shot_models():
    table = results_table([summary("bcresnet1", 6), summary("whisper", 3, zero_shot=True)],
                          COSTS).set_index("model")
    assert table.loc["whisper", "control_accuracy"] == pytest.approx(0.9)
    assert pd.isna(table.loc["bcresnet1", "control_accuracy"])


def test_model_without_a_cost_profile_is_an_error():
    with pytest.raises(ValueError, match="bcresnet3"):
        results_table([summary("bcresnet3", 6)], COSTS)


def test_markdown_formats_percentages_intervals_and_counts():
    md = to_markdown(results_table([summary("bcresnet1", 6)], COSTS))
    lines = md.strip().splitlines()
    assert len(lines) == 3  # header, separator, one model
    row = lines[2]
    assert "| bcresnet1 |" in row
    assert "60.0" in row and "9.2k" in row and "4.9M" in row


def test_per_speaker_table_has_models_as_rows_and_speakers_as_columns():
    table = per_speaker_table([summary("bcresnet1", 6), summary("whisper", 3, zero_shot=True)])
    assert table.loc["bcresnet1", "M04"] == pytest.approx(0.6)
    assert table.loc["whisper", "F01"] == pytest.approx(0.3)
    assert table.loc["n clips", "F01"] == 10


def test_accuracy_vs_macs_figure_is_written(tmp_path):
    path = tmp_path / "acc_vs_macs.png"
    plot_accuracy_vs_macs([summary("bcresnet1", 6), summary("whisper", 3, zero_shot=True)],
                          COSTS, path, secondary={"bcresnet1": 0.97})
    assert path.stat().st_size > 0


def test_accuracy_vs_macs_needs_a_cost_for_every_model(tmp_path):
    with pytest.raises(ValueError, match="bcresnet3"):
        plot_accuracy_vs_macs([summary("bcresnet3", 6)], COSTS, tmp_path / "x.png")


def test_confusion_figure_is_written(tmp_path):
    path = tmp_path / "confusion.png"
    plot_confusion(summary("bcresnet1", 6), path)
    assert path.stat().st_size > 0


def test_write_report_produces_tables_and_figure(tmp_path):
    write_report([summary("bcresnet1", 6)], COSTS, tmp_path)
    for name in ("results.csv", "results.md", "per_speaker.csv", "accuracy_vs_macs.png"):
        assert (tmp_path / name).stat().st_size > 0


def test_markdown_rows_each_show_their_own_model_values():
    a = summarize([Run(model="bcresnet1", seed=i, fold_train_speakers=loso_folds(),
                       predictions=dysarthric_preds({"M05": 2})) for i in range(3)])
    b = summarize([Run(model="whisper", seed=i, fold_train_speakers=loso_folds(),
                       predictions=dysarthric_preds({"M05": 7})) for i in range(3)])
    rows = to_markdown(results_table([a, b], COSTS)).strip().splitlines()[2:]
    moderate_severe = [r.split("|")[7].strip() for r in rows]  # 7th column
    assert moderate_severe == ["20.0", "70.0"]


def subset_summary(model):
    """Same accuracy pattern as summary(), but scored without F01's first clip."""
    preds = dysarthric_preds({s: 6 for s in ALL_DYSARTHRIC}).iloc[1:]
    return summarize([Run(model=model, seed=i, predictions=preds,
                          fold_train_speakers=loso_folds()) for i in range(3)])


def test_models_scored_on_different_clips_cannot_share_a_table(tmp_path):
    summaries = [summary("bcresnet1", 6), subset_summary("whisper")]
    with pytest.raises(ValueError, match="same clips"):
        results_table(summaries, COSTS)
    with pytest.raises(ValueError, match="same clips"):
        per_speaker_table(summaries)
    with pytest.raises(ValueError, match="same clips"):
        plot_accuracy_vs_macs(summaries, COSTS, tmp_path / "x.png")


def test_models_on_different_mics_cannot_share_a_table():
    both = pd.concat([dysarthric_preds({}), dysarthric_preds({}, mic=HEAD_MIC)],
                     ignore_index=True)
    runs = [Run(model="bcresnet1", seed=i, predictions=both, fold_train_speakers=loso_folds())
            for i in range(3)]
    head = summarize(runs, mic=HEAD_MIC)
    other = summarize([Run(model="whisper", seed=i, predictions=both,
                           fold_train_speakers=loso_folds()) for i in range(3)])
    with pytest.raises(ValueError, match="mic"):
        results_table([head, other], COSTS)


def test_the_same_model_twice_in_a_table_is_rejected():
    with pytest.raises(ValueError, match="bcresnet1"):
        per_speaker_table([summary("bcresnet1", 6), summary("bcresnet1", 7)])
