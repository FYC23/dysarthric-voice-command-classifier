"""
Results tables: one row per model, accuracy and cost side by side, in the
layout of the BC-ResNet paper's Tables 2-3 plus our uncertainty columns.
"""

import math
from pathlib import Path
from typing import Mapping, Optional, Sequence

import pandas as pd

from src.eval.aggregate import ModelSummary, check_summaries_comparable
from src.eval.compare import PairedComparison
from src.eval.cost import CostProfile
from src.eval.plots import plot_accuracy_vs_macs
from src.eval.units import human_count

MISSING = "—"


def results_table(summaries: Sequence[ModelSummary],
                  costs: Mapping[str, CostProfile]) -> pd.DataFrame:
    """Numeric table (fractions, raw counts), one row per model in the given order."""
    check_summaries_comparable(summaries)
    missing = [s.model for s in summaries if s.model not in costs]
    if missing:
        raise ValueError(f"no cost profile for models: {missing}")
    return pd.DataFrame([_row(s, costs[s.model]) for s in summaries])


def _row(s: ModelSummary, cost: CostProfile) -> dict:
    return {
        "model": s.model,
        "mic": s.mic,
        "n_seeds": s.n_seeds,
        "accuracy": s.headline,
        "seed_std": s.seed_std,
        "ci_low": s.ci_low,
        "ci_high": s.ci_high,
        **s.severity,
        "pooled": s.pooled,
        **s.pretraining_overlap,
        "oov_rate": s.oov_rate,
        "reject_rate": s.reject_rate,
        "control_accuracy": s.control_headline if s.control_headline is not None else math.nan,
        "params": cost.params,
        "macs": cost.macs,
        "int8_mb": cost.int8_mb,
        "input_seconds": cost.input_seconds,
        "note": cost.note,
    }


def per_speaker_table(summaries: Sequence[ModelSummary]) -> pd.DataFrame:
    """Rows: one per model, then clip counts. Columns: speakers. Cells: accuracy."""
    check_summaries_comparable(summaries)
    rows = {s.model: s.per_speaker["accuracy"] for s in summaries}
    rows["n clips"] = summaries[0].per_speaker["n"]
    return pd.DataFrame(rows).T


def comparisons_table(comparisons: Sequence[PairedComparison]) -> pd.DataFrame:
    """One row per (candidate, baseline) pair: speakers better/worse/tied, mean gain, CI."""
    return pd.DataFrame([{
        "candidate": c.candidate,
        "baseline": c.baseline,
        "mic": c.mic,
        "n_better": c.n_better,
        "n_worse": c.n_worse,
        "n_tied": c.n_tied,
        "mean_diff": c.mean_diff,
        "ci_low": c.ci_low,
        "ci_high": c.ci_high,
    } for c in comparisons])


def _signed_pts(x: float) -> str:
    return f"{100 * x:+.1f}"


def comparisons_markdown(table: pd.DataFrame) -> str:
    """comparisons_table as markdown, gains in percentage points."""
    header = ["Candidate", "Baseline", "Better on", "Worse on", "Tied",
              "Mean gain (pts)", "95% CI (speakers)"]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for r in table.to_dict("records"):
        n_speakers = r["n_better"] + r["n_worse"] + r["n_tied"]
        cells = [
            r["candidate"], r["baseline"], f"{r['n_better']} of {n_speakers}",
            str(r["n_worse"]), str(r["n_tied"]), _signed_pts(r["mean_diff"]),
            f"[{_signed_pts(r['ci_low'])}, {_signed_pts(r['ci_high'])}]",
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def _pct(x: float) -> str:
    return MISSING if pd.isna(x) else f"{100 * x:.1f}"


def to_markdown(table: pd.DataFrame) -> str:
    """The headline columns of results_table as a markdown table."""
    header = ["Model", "Seeds", "Acc. %", "± seed std", "95% CI (speakers)", "Severe",
              "Mod.-severe", "Mild", "Control", "#Params", "#MACs", "Input s", "Note"]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for r in table.to_dict("records"):
        cells = [
            r["model"], str(r["n_seeds"]), _pct(r["accuracy"]), _pct(r["seed_std"]),
            f"[{_pct(r['ci_low'])}, {_pct(r['ci_high'])}]",
            _pct(r["severe"]), _pct(r["moderate-severe"]),
            _pct(r["mild"]), _pct(r["control_accuracy"]), human_count(r["params"]),
            human_count(r["macs"]), f"{r['input_seconds']:g}", r["note"] or "",
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def write_report(summaries: Sequence[ModelSummary], costs: Mapping[str, CostProfile],
                 out_dir: Path, secondary: Optional[Mapping[str, float]] = None,
                 labels: Optional[Mapping[str, str]] = None) -> None:
    """results.csv / .md, per_speaker.csv and the accuracy-vs-MACs figure."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    table = results_table(summaries, costs)
    table.to_csv(out_dir / "results.csv", index=False)
    (out_dir / "results.md").write_text(to_markdown(table))
    per_speaker_table(summaries).to_csv(out_dir / "per_speaker.csv")
    plot_accuracy_vs_macs(summaries, costs, out_dir / "accuracy_vs_macs.png", secondary,
                          labels=labels)
