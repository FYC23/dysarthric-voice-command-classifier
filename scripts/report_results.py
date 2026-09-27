#!/usr/bin/env python
"""
Combined results: every BC-ResNet width against the zero-shot ASR baselines,
on the same TORGO clips. Tables, paired per-speaker comparisons, accuracy vs
MACs and vs parameters (with Speech Commands accuracy as a second series) and
a confusion figure per model; array mic in the output folder, head mic in its
head-mic/ subfolder.

Uses every finished seed of each width (a seed still training has no eval/
yet and is skipped) unless --seeds pins them.

Usage:
    python scripts/report_results.py
    python scripts/report_results.py --seeds 0 1 --out outputs/step3-bcresnet
"""

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.baselines.asr.scoring import SCORERS
from src.baselines.asr.transcribers import MODELS as ASR_MODELS
from src.config import config
from src.eval.combined import load_seed_runs, write_combined_report
from src.eval.cost import COST_FILE, load_cost
from src.training.bcresnet_recipe import TAUS, run_name
from src.training.pretrain import METRICS_FILE

ASR_LABELS = {"whisper-large-v3": "Whisper large-v3", "parakeet-tdt-0.6b-v3": "Parakeet-TDT 0.6B"}


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BC-ResNet vs zero-shot ASR on TORGO")
    parser.add_argument("--out", type=Path, default=config.OUTPUT_DIR / "step3-bcresnet")
    parser.add_argument("--runs-dir", type=Path, default=config.RUNS_DIR)
    parser.add_argument("--taus", type=float, nargs="+", default=list(TAUS),
                        help="BC-ResNet widths to report (default: all)")
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                        help="BC-ResNet seeds to report (default: every finished seed)")
    parser.add_argument("--scorer", choices=sorted(SCORERS), default="lenient",
                        help="ASR scoring; lenient maps each transcript to the nearest command")
    return parser.parse_args(argv)


def speech_commands_accuracy(runs_dir: Path, model: str, seeds: Sequence[int]) -> Optional[float]:
    """Mean Speech Commands test accuracy over `seeds`; None if any seed lacks its metrics."""
    paths = [Path(runs_dir) / model / f"seed{s}" / METRICS_FILE for s in seeds]
    if not all(p.exists() for p in paths):
        return None
    return sum(json.loads(p.read_text())["test_acc"] for p in paths) / len(paths)


def main(argv=None) -> None:
    args = parse_args(argv)
    runs_dir = args.runs_dir
    baselines = {f"{m}-{args.scorer}": load_seed_runs(runs_dir, f"{m}-{args.scorer}")
                 for m in ASR_MODELS}
    costs = {f"{m}-{args.scorer}": load_cost(runs_dir / m / COST_FILE) for m in ASR_MODELS}
    labels = {f"{m}-{args.scorer}": ASR_LABELS.get(m, m) for m in ASR_MODELS}

    candidates, secondary = {}, {}
    for tau in args.taus:
        model = run_name(tau)
        labels[model] = f"BC-ResNet-{tau:g}"
        candidates[model] = load_seed_runs(runs_dir, model, args.seeds)
        costs[model] = load_cost(runs_dir / model / COST_FILE)
        acc = speech_commands_accuracy(runs_dir, model, [r.seed for r in candidates[model]])
        if acc is None:
            print(f"{model}: no {METRICS_FILE} for every seed; left out of the "
                  "Speech Commands series")
        else:
            secondary[model] = acc

    # The harness warns on every summary of a model with < 3 seeds (both mics,
    # every comparison); report each such model once instead
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        write_combined_report(baselines, candidates, costs, args.out, secondary, labels)
    for message in dict.fromkeys(str(w.message) for w in caught):
        print(f"note: {message}")
    pd.DataFrame([{"model": m, "seeds": " ".join(str(r.seed) for r in candidates[m]),
                   "test_acc": secondary.get(m)} for m in candidates]
                 ).to_csv(args.out / "speech_commands.csv", index=False)

    print((args.out / "results.md").read_text())
    print((args.out / "comparisons.md").read_text())
    print(f"Report written to {args.out}")


if __name__ == "__main__":
    main()
