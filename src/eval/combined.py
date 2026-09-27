"""
One report across steps: zero-shot baselines and trained models on the same
clips, with every trained model compared speaker by speaker against every
baseline. Array mic is the top folder; the head mic gets its own subfolder,
since the two mics are never combined.

Saved runs live at runs/<model>/seed<k>/eval/ (src.eval.io); a seed without
eval/run.json is still training and is skipped.
"""

import re
from pathlib import Path
from typing import List, Mapping, Optional, Sequence

from src.eval.aggregate import summarize
from src.eval.compare import compare
from src.eval.constants import ARRAY_MIC, HEAD_MIC
from src.eval.cost import CostProfile
from src.eval.io import METADATA_FILE, load_run
from src.eval.plots import plot_accuracy_vs_cost, plot_confusion
from src.eval.report import comparisons_markdown, comparisons_table, write_report
from src.eval.schema import EvalValidationError, Run

MIC_DIRS = {ARRAY_MIC: ".", HEAD_MIC: "head-mic"}
SEED_DIR = re.compile(r"seed(\d+)")


def finished_seeds(runs_dir: Path, model: str) -> List[int]:
    """Seeds of `model` with a saved eval run, in numeric order."""
    model_dir = Path(runs_dir) / model
    if not model_dir.is_dir():
        return []
    seeds = [int(m.group(1)) for d in model_dir.iterdir()
             if (m := SEED_DIR.fullmatch(d.name)) and (d / "eval" / METADATA_FILE).exists()]
    return sorted(seeds)


def load_seed_runs(runs_dir: Path, model: str,
                   seeds: Optional[Sequence[int]] = None) -> List[Run]:
    """
    Every finished seed of `model`, or exactly `seeds` if given. A requested
    seed that has not finished is an error, so a table never silently mixes
    seed counts the caller did not ask for.
    """
    available = finished_seeds(runs_dir, model)
    if not available:
        raise FileNotFoundError(f"no finished seed of {model} under {Path(runs_dir) / model}")
    if seeds is not None:
        missing = sorted(set(seeds) - set(available))
        if missing:
            raise FileNotFoundError(f"{model}: seed {', '.join(map(str, missing))} not "
                                    f"finished (have {available})")
        available = sorted(set(seeds))

    runs = [load_run(Path(runs_dir) / model / f"seed{s}" / "eval") for s in available]
    wrong = sorted({r.model for r in runs} - {model})
    if wrong:
        raise EvalValidationError(f"{Path(runs_dir) / model} holds runs of {wrong}")
    return runs


def write_combined_report(baselines: Mapping[str, Sequence[Run]],
                          candidates: Mapping[str, Sequence[Run]],
                          costs: Mapping[str, CostProfile], out_dir: Path,
                          secondary: Optional[Mapping[str, float]] = None,
                          labels: Optional[Mapping[str, str]] = None) -> None:
    """
    Per mic: results.csv/.md, per_speaker.csv, accuracy vs MACs and vs params,
    comparisons.csv/.md (each candidate against each baseline) and a confusion
    figure per model. `secondary` adds e.g. Speech Commands accuracy to the plots;
    `labels` gives the figures' display names.
    """
    both = sorted(set(baselines) & set(candidates))
    if both:
        raise EvalValidationError(f"listed as both baseline and candidate: {both}")

    for mic, sub in MIC_DIRS.items():
        mic_dir = Path(out_dir) / sub
        runs = {**baselines, **candidates}
        summaries = [summarize(r, mic=mic) for r in runs.values()]
        write_report(summaries, costs, mic_dir, secondary, labels)
        plot_accuracy_vs_cost(summaries, costs, mic_dir / "accuracy_vs_params.png",
                              axis="params", secondary=secondary, labels=labels)
        for summary in summaries:
            plot_confusion(summary, mic_dir / f"confusion_{summary.model}.png")

        pairs = [compare(baselines[b], candidates[c], mic=mic)
                 for c in candidates for b in baselines]
        table = comparisons_table(pairs)
        table.to_csv(mic_dir / "comparisons.csv", index=False)
        (mic_dir / "comparisons.md").write_text(comparisons_markdown(table))
