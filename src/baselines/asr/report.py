"""
Step 1 report: the saved ASR runs through the eval harness's tables and
figures. Array mic is the headline folder; the head mic gets its own
subfolder, since the two mics are never combined.
"""

from pathlib import Path
from typing import Sequence

from src.baselines.asr.pipeline import COST_FILE, load_cost, run_dir
from src.baselines.asr.scoring import SCORERS
from src.eval.aggregate import summarize
from src.eval.constants import ARRAY_MIC, HEAD_MIC
from src.eval.io import load_run
from src.eval.plots import plot_confusion
from src.eval.report import write_report

MIC_DIRS = {ARRAY_MIC: ".", HEAD_MIC: "head-mic"}


def write_asr_report(runs_dir: Path, out_dir: Path, models: Sequence[str]) -> None:
    """results.csv/.md, per_speaker.csv, accuracy_vs_macs.png and a confusion per run, per mic."""
    runs_dir, out_dir = Path(runs_dir), Path(out_dir)
    names = [f"{m}-{s}" for m in models for s in SCORERS]
    missing = [n for n in names if not run_dir(runs_dir, n).exists()]
    missing += [m for m in models if not (runs_dir / m / COST_FILE).exists()]
    if missing:
        raise FileNotFoundError(f"no saved run or cost for {missing} under {runs_dir}; "
                                "run scripts/run_asr_baseline.py first")

    runs = [load_run(run_dir(runs_dir, n)) for n in names]
    costs = {f"{m}-{s}": load_cost(runs_dir / m / COST_FILE) for m in models for s in SCORERS}
    for mic, sub in MIC_DIRS.items():
        mic_dir = out_dir / sub
        summaries = [summarize([run], mic=mic) for run in runs]
        write_report(summaries, costs, mic_dir)
        for summary in summaries:
            plot_confusion(summary, mic_dir / f"confusion_{summary.model}.png")
