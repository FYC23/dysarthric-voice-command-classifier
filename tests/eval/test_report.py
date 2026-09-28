"""Tests for the results tables, their formatting, and the figures."""

import pandas as pd
import pytest
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.transforms import Bbox

from src.eval.aggregate import summarize
from src.eval.constants import HEAD_MIC
from src.eval.cost import CostProfile
from src.eval.compare import compare
from src.eval.plots import (
    _anchor, plot_accuracy_vs_cost, plot_accuracy_vs_macs, plot_confusion,
)
from src.eval.report import (
    _signed_pts, comparisons_markdown, comparisons_table, human_count, per_speaker_table,
    results_table, to_markdown, write_report,
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


def test_accuracy_vs_params_figure_is_written(tmp_path):
    path = tmp_path / "acc_vs_params.png"
    plot_accuracy_vs_cost([summary("bcresnet1", 6), summary("whisper", 3, zero_shot=True)],
                          COSTS, path, axis="params", secondary={"bcresnet1": 0.97})
    assert path.stat().st_size > 0


def test_unknown_cost_axis_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="latency"):
        plot_accuracy_vs_cost([summary("bcresnet1", 6)], COSTS, tmp_path / "x.png",
                              axis="latency")


def seed_runs(model, correct):
    return [Run(model=model, seed=i, predictions=dysarthric_preds(correct),
                fold_train_speakers=loso_folds()) for i in range(3)]


def test_comparisons_table_has_one_row_per_pair():
    base = seed_runs("whisper", {s: 5 for s in ALL_DYSARTHRIC})
    cand = seed_runs("bcresnet1", {**{s: 7 for s in ALL_DYSARTHRIC}, "M03": 5})
    table = comparisons_table([compare(base, cand)])
    row = table.iloc[0]
    assert (row["candidate"], row["baseline"]) == ("bcresnet1", "whisper")
    assert (row["n_better"], row["n_worse"], row["n_tied"]) == (7, 0, 1)
    assert row["mean_diff"] == pytest.approx(7 * 0.2 / 8)


def test_comparisons_markdown_shows_signed_gains_in_points():
    base = seed_runs("whisper", {s: 5 for s in ALL_DYSARTHRIC})
    cand = seed_runs("bcresnet1", {s: 7 for s in ALL_DYSARTHRIC})
    lines = comparisons_markdown(comparisons_table([compare(base, cand)])).strip().splitlines()
    assert len(lines) == 3
    assert "| bcresnet1 | whisper | 8 of 8 | 0 | 0 | +20.0 |" in lines[2]


@pytest.mark.parametrize("x, text", [(-0.0004, "+0.0"), (-0.0, "+0.0"), (0.0, "+0.0"),
                                     (-0.0006, "-0.1"), (0.227, "+22.7")])
def test_signed_points_never_print_negative_zero(x, text):
    assert _signed_pts(x) == text


def test_a_model_with_zero_cost_does_not_break_the_log_axis_figure(tmp_path):
    costs = {**COSTS, "free": CostProfile(params=0, macs=0, input_seconds=2.0)}
    path = tmp_path / "x.png"
    plot_accuracy_vs_cost([summary("bcresnet1", 6), summary("free", 3)], costs, path)
    assert path.stat().st_size > 0


def spread_summary(model, correct):
    """3 seeds; the 8 speakers get the given numbers of 10 clips right, so the CI has width."""
    by_speaker = dict(zip(ALL_DYSARTHRIC, correct))
    return summarize([Run(model=model, seed=i, predictions=dysarthric_preds(by_speaker),
                          fold_train_speakers=loso_folds()) for i in range(3)])


# Shaped like the real size curve: pretrained models and ASR crowded into two
# decades, a precise model above wide intervals on both sides, and a small
# model with a second (Speech Commands) series and so a legend
CROWDED = {
    "BC-ResNet-8": (3.2e5, [7, 8, 8, 9, 9, 9, 9, 10]),
    "DistilHuBERT": (2.4e7, [6, 7, 8, 8, 9, 8, 9, 9]),
    "HuBERT-base": (9.5e7, [5, 6, 8, 9, 10, 9, 9, 9]),
    "HuBERT-large": (3.2e8, [8, 8, 9, 9, 10, 9, 9, 9]),
    "Parakeet-TDT 0.6B": (6.3e8, [3, 5, 6, 8, 9, 9, 8, 9]),
    "Whisper large-v3": (1.5e9, [3, 4, 6, 7, 9, 9, 7, 8]),
}
# Like the smallest BC-ResNets: a narrow interval between two that reach
# lower on both sides, too close for its label to fit under it, with a
# Speech Commands point above it
SQUEEZED = {
    "BC-ResNet-1": (9.8e3, [5, 7, 8, 8, 9, 9, 9, 9]),
    "BC-ResNet-2": (2.8e4, [8, 8, 9, 9, 9, 9, 9, 9]),
    "BC-ResNet-3": (5.5e4, [7, 8, 9, 9, 9, 9, 9, 10]),
    "Whisper large-v3": (1.5e9, [3, 4, 6, 7, 9, 9, 7, 8]),
}
SPEECH_COMMANDS = {"BC-ResNet-1": 0.95, "BC-ResNet-2": 0.97, "BC-ResNet-3": 0.978,
                   "BC-ResNet-8": 0.982}
MAX_LABEL_GAP_PT = 30  # a label never sits further than this from its own bar


def label_geometry(fig):
    """Each label's box and every model's error-bar box, in display units, in x order."""
    renderer = FigureCanvasAgg(fig).get_renderer()  # at the figure's own dpi
    ax = fig.axes[0]
    labels = [t.get_window_extent(renderer) for t in ax.texts]
    bars = []
    for container in ax.containers:
        (x, lo), (_, hi) = container.lines[2][0].get_segments()[0]
        (x0, y0), (x1, y1) = ax.transData.transform([(x, lo), (x, hi)])
        bars.append((x0, min(y0, y1), x1, max(y0, y1)))
    return labels, sorted(bars)


def marker_boxes(fig):
    """A box around every marker drawn (points, error-bar caps, second series)."""
    ax, pt = fig.axes[0], fig.dpi / 72
    boxes = []
    for line in ax.lines:
        if line.get_marker() in (None, "None", "", " "):
            continue
        half = (line.get_markersize() + line.get_markeredgewidth()) / 2 * pt
        boxes += [Bbox.from_extents(x - half, y - half, x + half, y + half)
                  for x, y in ax.transData.transform(line.get_xydata())]
    return boxes


def gap(box, bar):
    """Distance from a label box to a vertical bar box (0 if they touch)."""
    dx = max(bar[0] - box.x1, box.x0 - bar[2], 0)
    dy = max(bar[1] - box.y1, box.y0 - bar[3], 0)
    return (dx ** 2 + dy ** 2) ** 0.5


def cost_figures(models, tmp_path):
    """The params and MACs figures of `models` (name: (cost, correct per speaker))."""
    summaries = [spread_summary(m, c) for m, (_, c) in models.items()]
    for axis, scale in (("params", 1), ("macs", 1e-3)):
        costs = {m: CostProfile(params=int(x), macs=int(x * scale * 50), input_seconds=2.0)
                 for m, (x, _) in models.items()}
        yield plot_accuracy_vs_cost(summaries, costs, tmp_path / "x.png", axis=axis,
                                    secondary=SPEECH_COMMANDS)


def assert_labels_clear_everything(fig, n_models):
    """Inside the axes, and clear of each other, the legend, every marker and other bars."""
    labels, bars = label_geometry(fig)
    ax = fig.axes[0]
    legend = ax.get_legend().get_window_extent()
    assert len(labels) == len(bars) == n_models
    assert not any(a.overlaps(b) for i, a in enumerate(labels) for b in labels[i + 1:])
    for i, box in enumerate(labels):
        assert ax.bbox.contains(box.x0, box.y0) and ax.bbox.contains(box.x1, box.y1)
        assert not box.overlaps(legend), f"label {i} covers the legend"
        assert not any(box.overlaps(m) for m in marker_boxes(fig)), f"label {i} covers a marker"
        assert gap(box, bars[i]) <= MAX_LABEL_GAP_PT * fig.dpi / 72, f"label {i} is far off"
        assert all(gap(box, bar) > 0 for j, bar in enumerate(bars) if j != i), \
            f"label {i} touches another model's error bar"
    return labels, bars


def test_crowded_labels_sit_nearest_their_own_error_bar(tmp_path):
    for fig in cost_figures(CROWDED, tmp_path):
        labels, bars = assert_labels_clear_everything(fig, len(CROWDED))
        for i, box in enumerate(labels):
            others = [gap(box, bar) for j, bar in enumerate(bars) if j != i]
            assert gap(box, bars[i]) < min(others), f"label {i} is nearer another point"


def test_a_squeezed_label_still_centres_nearest_its_own_point(tmp_path):
    for fig in cost_figures(SQUEEZED, tmp_path):
        labels, bars = assert_labels_clear_everything(fig, len(SQUEEZED))
        for i, box in enumerate(labels):
            centre = (box.x0 + box.x1) / 2
            offsets = [abs(centre - bar[0]) for bar in bars]
            assert offsets[i] < min(o for j, o in enumerate(offsets) if j != i), \
                f"label {i} is centred nearer another point"


@pytest.mark.parametrize("along, y", [(-1, 70), (-0.5, 75), (0, 80), (0.5, 85), (1, 90)])
def test_a_label_anchor_runs_from_the_lower_end_of_the_bar_through_the_point_to_the_upper(along, y):
    assert _anchor((1e4, 80, 70, 90), along) == (1e4, pytest.approx(y))
