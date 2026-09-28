"""
Figures: accuracy vs MACs (after the BC-ResNet paper's Figure 3) and the
confusion matrix. Static PNGs for the README, built with matplotlib's
object API so importing this module never changes the global backend.

Colours are the dataviz reference palette: categorical slots 1-2 (validated
colour-blind safe as a pair) and its blue sequential ramp.
"""

from pathlib import Path
from typing import List, Mapping, Optional, Sequence, Tuple

from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure

from src.eval.aggregate import ModelSummary, check_summaries_comparable
from src.eval.constants import MIC_LABELS
from src.eval.cost import CostProfile
from src.eval.labels import (
    Box, Position, candidate_positions, choose_placements, label_box,
)

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SERIES_1 = "#2a78d6"  # TORGO dysarthric
SERIES_2 = "#eb6834"  # secondary series, e.g. Speech Commands
SEQUENTIAL = (SURFACE, "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b")
DPI = 200

LABEL_GAP_PT = 6       # between a label and the end of its error bar
LABEL_SIDE_PT = 11     # from a point's centre to a label beside it (clears the clearance)
LABEL_STEP_PT = 6      # a label moves out half a row (8 pt text) at a time
LABEL_MAX_STEPS = 8    # how far past the end of its error bar a label may go
LABEL_SHIFT_PT = 4     # a label under or over its bar slides sideways this much at a time
LABEL_MAX_SHIFTS = 8   # ... up to this many times each way
LABEL_CLEARANCE_PT = 5  # sideways room kept between a label and another model's bar
LABEL_PAD_PT = 1.5     # room kept around every label, so neighbouring labels never touch
LABEL_POSITIONS = candidate_positions(LABEL_GAP_PT, LABEL_SIDE_PT, LABEL_STEP_PT,
                                      LABEL_SHIFT_PT, LABEL_MAX_STEPS, LABEL_MAX_SHIFTS)

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


def _box(ax, x0: float, y0: float, x1: float, y1: float, pad_x: float, pad_y: float) -> Box:
    """Data-space rectangle -> display box, padded by pad_x / pad_y display units."""
    (ax0, ay0), (ax1, ay1) = ax.transData.transform([(x0, y0), (x1, y1)])
    return (min(ax0, ax1) - pad_x, min(ay0, ay1) - pad_y,
            max(ax0, ax1) + pad_x, max(ay0, ay1) + pad_y)


def _obstacles(fig: Figure, ax) -> List[Box]:
    """Everything a label must not cover: markers (points, caps), error bars, the legend."""
    pt = fig.dpi / 72
    boxes = []
    for line in ax.lines:
        if line.get_marker() in (None, "None", "", " "):
            continue
        half = (line.get_markersize() + line.get_markeredgewidth()) / 2 * pt
        boxes += [_box(ax, x, y, x, y, half + LABEL_CLEARANCE_PT * pt, half)
                  for x, y in line.get_xydata()]
    for collection in ax.collections:
        boxes += [_box(ax, *seg[0], *seg[-1], LABEL_CLEARANCE_PT * pt, 0)
                  for seg in collection.get_segments()]
    legend = ax.get_legend()
    if legend is not None:
        boxes.append(tuple(legend.get_window_extent().extents))
    return boxes


def _anchor(point: Tuple[float, float, float, float], along: float) -> Tuple[float, float]:
    """Data spot on a point's (x, y, ci_low, ci_high) error bar at `along` (see Position)."""
    x, y, low, high = point
    return x, y + abs(along) * ((high if along > 0 else low) - y)


def _move_label(text, point: Tuple[float, float, float, float], position: Position) -> None:
    """Put `text` at `position` around point (x, y, ci_low, ci_high)."""
    text.xy = _anchor(point, position.along)
    text.xyann = (position.dx, position.dy)
    text.set_horizontalalignment(position.ha)
    text.set_verticalalignment(position.va)


def _candidates(fig: Figure, ax, text, point: tuple) -> List[Box]:
    """Display box of `text` at each of LABEL_POSITIONS around `point`, LABEL_PAD_PT wider."""
    extent, pad = text.get_window_extent(), LABEL_PAD_PT * fig.dpi / 72
    boxes = [label_box(tuple(ax.transData.transform(_anchor(point, p.along))),
                       (extent.width, extent.height), p, fig.dpi / 72) for p in LABEL_POSITIONS]
    return [(x0 - pad, y0 - pad, x1 + pad, y1 + pad) for x0, y0, x1, y1 in boxes]


def _place_labels(fig: Figure, ax, labelled: Sequence[tuple]) -> None:
    """
    Each (label, point) at the most preferred position that keeps it inside
    the axes, clear of every obstacle and label, and nearest its own bar
    (least nearer another, where neighbours leave no such spot).
    """
    candidates = [_candidates(fig, ax, text, point) for text, point in labelled]
    bars = [_box(ax, x, low, x, high, 0, 0) for _, (x, _, low, high) in labelled]
    chosen = choose_placements(candidates, _obstacles(fig, ax),
                               tuple(ax.get_window_extent().extents), bars)
    for (text, point), i in zip(labelled, chosen):
        _move_label(text, point, LABEL_POSITIONS[i])


def _plot_points(ax, summaries: Sequence[ModelSummary], costs: Mapping[str, CostProfile],
                 axis: str, labels: Mapping[str, str]) -> list:
    """Each model's accuracy and CI at its cost; returns (label, point) pairs in x order."""
    ordered = sorted(summaries, key=lambda s: getattr(costs[s.model], axis))
    labelled = []
    for i, s in enumerate(ordered):
        x, y = getattr(costs[s.model], axis), 100 * s.headline
        err = [[y - 100 * s.ci_low], [100 * s.ci_high - y]]
        ax.errorbar(x, y, yerr=err, fmt="o", color=SERIES_1, ms=8, mec=SURFACE, mew=2,
                    elinewidth=1.2, capsize=3, zorder=3,
                    label="TORGO dysarthric (95% CI over speakers)" if i == 0 else None)
        # Starts under the error bar; _place_labels moves it if that spot is taken
        text = ax.annotate(labels.get(s.model, s.model), (x, 100 * s.ci_low),
                           xytext=(0, -LABEL_GAP_PT), textcoords="offset points",
                           ha="center", va="top", fontsize=8, color=TEXT_SECONDARY)
        labelled.append((text, (x, y, 100 * s.ci_low, 100 * s.ci_high)))
    return labelled


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
    labelled = _plot_points(ax, summaries, costs, axis, labels)
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
    _place_labels(fig, ax, labelled)
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
