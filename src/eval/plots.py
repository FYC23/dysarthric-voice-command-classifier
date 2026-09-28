"""
Figures: accuracy vs MACs (after the BC-ResNet paper's Figure 3) and the
confusion matrix. Static PNGs for the README, built with matplotlib's
object API so importing this module never changes the global backend.

Colours are the dataviz reference palette: categorical slots 1-2 (validated
colour-blind safe as a pair) and its blue sequential ramp.
"""

import math
from pathlib import Path
from typing import List, Mapping, Optional, Sequence, Tuple

from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure

from src.eval.aggregate import ModelSummary, check_summaries_comparable
from src.eval.constants import MIC_LABELS
from src.eval.cost import CostProfile

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SERIES_1 = "#2a78d6"  # TORGO dysarthric
SERIES_2 = "#eb6834"  # secondary series, e.g. Speech Commands
SEQUENTIAL = (SURFACE, "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b")
DPI = 200

LABEL_NEIGHBOURHOOD_DECADES = 1.0  # models this close on the log x axis share label rows
LABEL_GAP_PT = 6   # between a label and the error bar it hangs from
LABEL_ROW_PT = 12  # one row of 8 pt labels

# Cost axis: (x-axis label, title)
COST_AXES = {
    "macs": ("MACs per forward pass (log scale)", "Accuracy vs. compute"),
    "params": ("Parameters (log scale)", "Accuracy vs. model size"),
}


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=8)


def plot_accuracy_vs_macs(summaries: Sequence[ModelSummary], costs: Mapping[str, CostProfile],
                          path: Path, secondary: Optional[Mapping[str, float]] = None,
                          secondary_label: str = "Speech Commands v2 (typical speech)",
                          labels: Optional[Mapping[str, str]] = None) -> None:
    """plot_accuracy_vs_cost with MACs on the x axis."""
    plot_accuracy_vs_cost(summaries, costs, path, "macs", secondary, secondary_label, labels)


def _label_anchors(xs: Sequence[float], lows: Sequence[float]) -> list:
    """
    Height to hang each label from: the lowest CI among models within a decade
    on the log x axis, so alternating label rows line up across neighbours.
    A non-positive cost has no place on a log axis and keeps its own CI.
    """
    logs = [math.log10(x) if x > 0 else None for x in xs]
    return [low if logs[i] is None else
            min(other for lx, other in zip(logs, lows)
                if lx is not None and abs(lx - logs[i]) <= LABEL_NEIGHBOURHOOD_DECADES)
            for i, low in enumerate(lows)]


def _label_rows(boxes: Sequence[Tuple[float, float, float, float]],
                row_height: float) -> List[int]:
    """
    Row for each label box (x0, y0, x1, y1, display units, y up), in the given
    order: the first row, counting down, where the box shifted down by
    row x `row_height` clears every box placed before it.
    """
    placed: List[Tuple[float, float, float, float]] = []
    rows = []
    for x0, y0, x1, y1 in boxes:
        row = 0
        while any(x0 < px1 and px0 < x1 and y0 - row * row_height < py1
                  and py0 < y1 - row * row_height for px0, py0, px1, py1 in placed):
            row += 1
        placed.append((x0, y0 - row * row_height, x1, y1 - row * row_height))
        rows.append(row)
    return rows


def _stack_labels(fig: Figure, texts: Sequence) -> None:
    """Move each label (in x order) down as many rows as it needs to overlap none before it."""
    boxes = [tuple(t.get_window_extent().extents) for t in texts]
    rows = _label_rows(boxes, LABEL_ROW_PT * fig.dpi / 72)
    for text, row in zip(texts, rows):
        text.xyann = (0, -LABEL_GAP_PT - LABEL_ROW_PT * row)


def _plot_points(ax, summaries: Sequence[ModelSummary], costs: Mapping[str, CostProfile],
                 axis: str, labels: Mapping[str, str]) -> list:
    """Each model's accuracy and CI at its cost, labelled; returns the labels in x order."""
    ordered = sorted(summaries, key=lambda s: getattr(costs[s.model], axis))
    anchors = _label_anchors([getattr(costs[s.model], axis) for s in ordered],
                             [100 * s.ci_low for s in ordered])
    texts = []
    for i, s in enumerate(ordered):
        x, y = getattr(costs[s.model], axis), 100 * s.headline
        err = [[y - 100 * s.ci_low], [100 * s.ci_high - y]]
        ax.errorbar(x, y, yerr=err, fmt="o", color=SERIES_1, ms=8, mec=SURFACE, mew=2,
                    elinewidth=1.2, capsize=3, zorder=3,
                    label="TORGO dysarthric (95% CI over speakers)" if i == 0 else None)
        # Under the error bar (Speech Commands markers sit above); _stack_labels
        # then moves labels of nearby models onto separate rows
        texts.append(ax.annotate(labels.get(s.model, s.model), (x, anchors[i]),
                                 xytext=(0, -LABEL_GAP_PT), textcoords="offset points",
                                 ha="center", va="top", fontsize=8, color=TEXT_SECONDARY))
    return texts


def plot_accuracy_vs_cost(summaries: Sequence[ModelSummary], costs: Mapping[str, CostProfile],
                          path: Path, axis: str = "macs",
                          secondary: Optional[Mapping[str, float]] = None,
                          secondary_label: str = "Speech Commands v2 (typical speech)",
                          labels: Optional[Mapping[str, str]] = None) -> Figure:
    """
    One point per model: speaker-averaged dysarthric accuracy against a cost
    (`axis`: "macs" or "params", log x), with the speaker-bootstrap 95% CI as
    error bars. `secondary` optionally adds each model's accuracy on another
    dataset, to show whether the size gap grows on dysarthric speech. `labels`
    maps model names to the names shown on the figure (default: the model name).
    Returns the saved figure.
    """
    if axis not in COST_AXES:
        raise ValueError(f"unknown cost axis {axis!r}; choose from {sorted(COST_AXES)}")
    check_summaries_comparable(summaries)
    missing = [s.model for s in summaries if s.model not in costs]
    if missing:
        raise ValueError(f"no cost profile for models: {missing}")
    labels = labels or {}

    fig = Figure(figsize=(8, 4.5), facecolor=SURFACE)
    ax = fig.add_subplot()
    _style(ax)
    texts = _plot_points(ax, summaries, costs, axis, labels)
    if secondary:
        pts = [(getattr(costs[m], axis), 100 * acc) for m, acc in secondary.items()
               if m in costs]
        ax.plot(*zip(*pts), "s", color=SERIES_2, ms=7, mec=SURFACE, mew=2, zorder=3,
                label=secondary_label)
        ax.legend(frameon=False, fontsize=8, labelcolor=TEXT_SECONDARY, loc="lower right")

    xs = [x for x in (getattr(costs[s.model], axis) for s in summaries) if x > 0]
    if xs:
        ax.set_xlim(min(xs) / 3, max(xs) * 3)  # room for the centred labels at both ends
    ax.set_xscale("log")
    ax.set_ylim(0, 102)  # headroom so markers at 100% are not clipped
    ax.set_yticks(range(0, 101, 20))
    x_label, title = COST_AXES[axis]
    ax.set_xlabel(x_label, color=TEXT, fontsize=9)
    mic = MIC_LABELS[summaries[0].mic]
    ax.set_ylabel(f"Speaker-averaged accuracy (%), {mic}", color=TEXT, fontsize=9)
    ax.set_title(title, color=TEXT, fontsize=11, loc="left")
    fig.tight_layout()
    _stack_labels(fig, texts)
    fig.savefig(path, dpi=DPI)
    return fig


def plot_confusion(summary: ModelSummary, path: Path) -> None:
    """Row-normalised confusion (share of each true word), summed over seeds."""
    counts = summary.confusion
    shares = counts.div(counts.sum(axis=1).where(lambda n: n > 0), axis=0).fillna(0)

    fig = Figure(figsize=(7.5, 6.5), facecolor=SURFACE)
    ax = fig.add_subplot()
    cmap = LinearSegmentedColormap.from_list("seq_blue", SEQUENTIAL)
    image = ax.imshow(shares.to_numpy(), cmap=cmap, vmin=0, vmax=1)
    ax.set_xticks(range(len(shares.columns)), shares.columns, rotation=90, fontsize=7,
                  color=TEXT_SECONDARY)
    ax.set_yticks(range(len(shares.index)), shares.index, fontsize=7, color=TEXT_SECONDARY)
    ax.set_xlabel("Predicted", color=TEXT, fontsize=9)
    ax.set_ylabel("Said", color=TEXT, fontsize=9)
    mic = MIC_LABELS[summary.mic]
    ax.set_title(f"{summary.model}: share of each word ({mic}, {summary.n_seeds} seeds)",
                 color=TEXT, fontsize=10, loc="left")
    bar = fig.colorbar(image, ax=ax, fraction=0.04)
    bar.ax.tick_params(labelsize=7, colors=TEXT_SECONDARY)
    bar.outline.set_visible(False)
    for spine in ax.spines.values():
        spine.set_color(AXIS)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
