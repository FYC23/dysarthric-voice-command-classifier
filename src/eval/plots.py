"""
Figures: accuracy vs MACs (after the BC-ResNet paper's Figure 3) and the
confusion matrix. Static PNGs for the README, built with matplotlib's
object API so importing this module never changes the global backend.

Colours are the dataviz reference palette: categorical slots 1-2 (validated
colour-blind safe as a pair) and its blue sequential ramp.
"""

from pathlib import Path
from typing import Mapping, Optional, Sequence

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
                          secondary_label: str = "Speech Commands v2 (typical speech)") -> None:
    """
    One point per model: speaker-averaged dysarthric accuracy against MACs (log x),
    with the speaker-bootstrap 95% CI as error bars. `secondary` optionally adds each
    model's accuracy on another dataset, to show whether the size gap grows on
    dysarthric speech.
    """
    check_summaries_comparable(summaries)
    missing = [s.model for s in summaries if s.model not in costs]
    if missing:
        raise ValueError(f"no cost profile for models: {missing}")

    fig = Figure(figsize=(7, 4.5), facecolor=SURFACE)
    ax = fig.add_subplot()
    _style(ax)
    for s in summaries:
        x, y = costs[s.model].macs, 100 * s.headline
        err = [[y - 100 * s.ci_low], [100 * s.ci_high - y]]
        ax.errorbar(x, y, yerr=err, fmt="o", color=SERIES_1, ms=8, mec=SURFACE, mew=2,
                    elinewidth=1.2, capsize=3, zorder=3,
                    label="TORGO dysarthric (95% CI over speakers)" if s is summaries[0] else None)
        ax.annotate(s.model, (x, y), xytext=(7, 5), textcoords="offset points",
                    fontsize=8, color=TEXT_SECONDARY)
    if secondary:
        pts = [(costs[m].macs, 100 * acc) for m, acc in secondary.items() if m in costs]
        ax.plot(*zip(*pts), "s", color=SERIES_2, ms=7, mec=SURFACE, mew=2, zorder=3,
                label=secondary_label)
        ax.legend(frameon=False, fontsize=8, labelcolor=TEXT_SECONDARY, loc="lower right")

    ax.set_xscale("log")
    ax.set_ylim(0, 102)  # headroom so markers at 100% are not clipped
    ax.set_yticks(range(0, 101, 20))
    ax.set_xlabel("MACs per forward pass (log scale)", color=TEXT, fontsize=9)
    mic = MIC_LABELS[summaries[0].mic]
    ax.set_ylabel(f"Speaker-averaged accuracy (%), {mic}", color=TEXT, fontsize=9)
    ax.set_title("Accuracy vs. compute", color=TEXT, fontsize=11, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)


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
