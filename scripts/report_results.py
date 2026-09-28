#!/usr/bin/env python
"""
Combined results: the zero-shot ASR baselines, every BC-ResNet width and the
pretrained SSL backbones (each leave-one-speaker-out, and its control-stage
model scored before any dysarthric training), on the same TORGO clips.
Tables, accuracy vs MACs and vs parameters (with Speech Commands accuracy as
a second series; the controls-only runs share their backbone's cost and stay
off these two figures), a confusion figure per model and paired per-speaker
comparisons answering three questions: trained models vs ASR, the effect of
dysarthric fine-tuning, and small models vs HuBERT-large. Array mic in the
output folder, head mic in its head-mic/ subfolder.

Uses every finished seed of each model (a seed still training has no eval/
yet and is skipped); --seeds pins the BC-ResNet seeds only.

Usage:
    python scripts/report_results.py
    python scripts/report_results.py --seeds 0 1 --backbones hubert-large --out outputs/results
"""

import argparse
import dataclasses
import json
import sys
import warnings
from pathlib import Path
from typing import List, Mapping, Optional, Sequence

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.baselines.asr.scoring import SCORERS
from src.baselines.asr.transcribers import MODELS as ASR_MODELS
from src.config import config
from src.eval.combined import PairGroup, every_pair, load_seed_runs, write_combined_report
from src.eval.cost import COST_FILE, CostProfile, load_cost
from src.model.backbones import BACKBONES
from src.training.bcresnet_recipe import TAUS, run_name
from src.training.pretrain import METRICS_FILE
from src.training.ssl_recipe import controls_run_name

ASR_LABELS = {"whisper-large-v3": "Whisper large-v3", "parakeet-tdt-0.6b-v3": "Parakeet-TDT 0.6B"}
SSL_LABELS = {"hubert-large": "HuBERT-large", "hubert-base": "HuBERT-base",
              "distilhubert": "DistilHuBERT"}
REFERENCE = "hubert-large"  # the large pretrained model the small ones are measured against
# src.training.ssl_finetune.ssl_cost counts the whole model; say so next to
# BC-ResNet's "log-Mel front end not counted"
SSL_MACS_NOTE = "MACs include the CNN front end, every transformer layer and the head"
CONTROLS_NOTE = "trained on control speakers only"

ASR_QUESTION = "Trained models vs zero-shot ASR"
FINETUNE_QUESTION = "Effect of dysarthric fine-tuning"
REFERENCE_QUESTION = "Small models vs the large pretrained reference"


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BC-ResNet and SSL backbones vs zero-shot "
                                                 "ASR on TORGO")
    parser.add_argument("--out", type=Path, default=config.OUTPUT_DIR / "results")
    parser.add_argument("--runs-dir", type=Path, default=config.RUNS_DIR)
    parser.add_argument("--taus", type=float, nargs="+", default=list(TAUS),
                        help="BC-ResNet widths to report (default: all)")
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                        help="BC-ResNet seeds to report (default: every finished seed); "
                             "the SSL backbones always use every finished seed")
    parser.add_argument("--backbones", nargs="*", choices=list(BACKBONES),
                        default=list(BACKBONES),
                        help="SSL backbones to report, each with its controls-only run "
                             "(default: all; pass none to leave them out)")
    parser.add_argument("--scorer", choices=sorted(SCORERS), default="lenient",
                        help="ASR scoring; lenient maps each transcript to the nearest command")
    return parser.parse_args(argv)


def speech_commands_accuracy(runs_dir: Path, model: str, seeds: Sequence[int]) -> Optional[float]:
    """Mean Speech Commands test accuracy over `seeds`; None if any seed lacks its metrics."""
    paths = [Path(runs_dir) / model / f"seed{s}" / METRICS_FILE for s in seeds]
    if not all(p.exists() for p in paths):
        return None
    return sum(json.loads(p.read_text())["test_acc"] for p in paths) / len(paths)


@dataclasses.dataclass(frozen=True)
class Family:
    """One model family's runs, costs, figure labels and Speech Commands accuracies."""
    runs: Mapping[str, list]
    costs: Mapping[str, CostProfile]
    labels: Mapping[str, str]
    secondary: Mapping[str, float] = dataclasses.field(default_factory=dict)


def asr_family(runs_dir: Path, scorer: str) -> Family:
    names = {m: f"{m}-{scorer}" for m in ASR_MODELS}
    return Family(runs={n: load_seed_runs(runs_dir, n) for n in names.values()},
                  costs={n: load_cost(runs_dir / m / COST_FILE) for m, n in names.items()},
                  labels={n: ASR_LABELS.get(m, m) for m, n in names.items()})


def bcresnet_family(runs_dir: Path, taus: Sequence[float],
                    seeds: Optional[Sequence[int]]) -> Family:
    models = {run_name(tau): f"BC-ResNet-{tau:g}" for tau in taus}
    runs = {m: load_seed_runs(runs_dir, m, seeds) for m in models}
    accuracies = {m: speech_commands_accuracy(runs_dir, m, [r.seed for r in runs[m]])
                  for m in models}
    for m in (m for m, acc in accuracies.items() if acc is None):
        print(f"{m}: no {METRICS_FILE} for every seed; left out of the Speech Commands series")
    return Family(runs=runs, costs={m: load_cost(runs_dir / m / COST_FILE) for m in models},
                  labels=models,
                  secondary={m: acc for m, acc in accuracies.items() if acc is not None})


def ssl_family(runs_dir: Path, backbones: Sequence[str]) -> Family:
    """Each backbone's LOSO run, then every controls-only run; every finished seed."""
    labels = {**{b: SSL_LABELS.get(b, b) for b in backbones},
              **{controls_run_name(b): f"{SSL_LABELS.get(b, b)} (controls only)"
                 for b in backbones}}
    notes = {m: f"{CONTROLS_NOTE}; {SSL_MACS_NOTE}" if m not in backbones else SSL_MACS_NOTE
             for m in labels}
    costs = {m: dataclasses.replace(load_cost(runs_dir / m / COST_FILE), note=note)
             for m, note in notes.items()}
    return Family(runs={m: load_seed_runs(runs_dir, m) for m in labels}, costs=costs,
                  labels=labels)


def pair_groups(candidates: Sequence[str], baselines: Sequence[str], taus: Sequence[float],
                backbones: Sequence[str]) -> List[PairGroup]:
    """Trained models vs ASR; each backbone vs its controls-only run; BC-ResNet vs HuBERT-large."""
    groups = [every_pair(ASR_QUESTION, candidates, baselines)]
    if backbones:
        groups.append(PairGroup(FINETUNE_QUESTION,
                                tuple((b, controls_run_name(b)) for b in backbones)))
    if REFERENCE in backbones and taus:
        groups.append(every_pair(REFERENCE_QUESTION, [run_name(t) for t in taus], [REFERENCE]))
    return groups


def main(argv=None) -> None:
    args = parse_args(argv)
    asr = asr_family(args.runs_dir, args.scorer)
    bcresnet = bcresnet_family(args.runs_dir, args.taus, args.seeds)
    ssl = ssl_family(args.runs_dir, args.backbones)
    candidates = {**bcresnet.runs, **ssl.runs}
    groups = pair_groups(list(candidates), list(asr.runs), args.taus, args.backbones)

    # The harness warns on every summary of a model with < 3 seeds (both mics,
    # every comparison); report each such model once instead
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        write_combined_report(asr.runs, candidates, {**asr.costs, **bcresnet.costs, **ssl.costs},
                              args.out, bcresnet.secondary,
                              {**asr.labels, **bcresnet.labels, **ssl.labels}, groups,
                              off_curve=[controls_run_name(b) for b in args.backbones])
    for message in dict.fromkeys(str(w.message) for w in caught):
        print(f"note: {message}")
    pd.DataFrame([{"model": m, "seeds": " ".join(str(r.seed) for r in runs),
                   "test_acc": bcresnet.secondary.get(m)} for m, runs in bcresnet.runs.items()]
                 ).to_csv(args.out / "speech_commands.csv", index=False)

    print((args.out / "results.md").read_text())
    print((args.out / "comparisons.md").read_text())
    print(f"Report written to {args.out}")


if __name__ == "__main__":
    main()
