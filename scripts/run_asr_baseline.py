#!/usr/bin/env python
"""
Zero-shot ASR baseline (step 1): transcribe every TORGO command clip (all
speakers, both mics, the classifiers' 2 s window) and score it through the
eval harness, strict and lenient.

Writes runs/<model>-strict/seed0/eval and runs/<model>-lenient/seed0/eval
(src.eval.io.load_run), runs/<model>/transcripts.csv (resumable cache) and
runs/<model>/cost.json.

Usage:
    python scripts/run_asr_baseline.py --model whisper-large-v3
    python scripts/run_asr_baseline.py --model parakeet-tdt-0.6b-v3 --device cpu
"""

import argparse
import functools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.baselines.asr.pipeline import COST_FILE, load_cost, run_baseline
from src.baselines.asr.transcribers import DTYPES, MODELS, build_transcriber, resolve_device
from src.config import config
from src.data.segments import evaluation_clips, load_torgo_samples
from src.eval.aggregate import summarize
from src.eval.constants import ARRAY_MIC


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Zero-shot ASR baseline on TORGO commands")
    parser.add_argument("--model", required=True, choices=sorted(MODELS))
    parser.add_argument("--device", choices=("cuda", "mps", "cpu"), default=None,
                        help="default: cuda, else mps, else cpu")
    parser.add_argument("--dtype", choices=list(DTYPES), default="float32")
    # fp32 Whisper large-v3 at 16 clips per batch swapped heavily on a 16 GB Mac
    parser.add_argument("--batch-size", type=int, default=4)
    return parser.parse_args(argv)


def run_name(args: argparse.Namespace) -> str:
    """
    The name transcripts, cost and runs are saved under. Precision is part of
    it when not the default, so float16 results never mix with float32 ones.
    """
    default_dtype = next(iter(DTYPES))
    return args.model if args.dtype == default_dtype else f"{args.model}-{args.dtype}"


def pct(x) -> str:
    return "—" if x is None else f"{100 * x:.1f}%"


def main(argv=None) -> None:
    args = parse_args(argv)
    device = resolve_device(args.device)
    print(f"{args.model} on {device} ({args.dtype})")

    clips = evaluation_clips(load_torgo_samples(config).samples)
    print(f"{len(clips)} clips: {clips.groupby(['is_dysarthric', 'mic']).size().to_dict()}")

    get_transcriber = functools.cache(
        lambda: build_transcriber(args.model, device, DTYPES[args.dtype]))
    name = run_name(args)
    runs = run_baseline(name, clips, get_transcriber, config.RUNS_DIR, args.batch_size)

    for run in runs:
        s = summarize([run], mic=ARRAY_MIC)
        print(f"\n{run.model} (array mic): dysarthric {pct(s.headline)} "
              f"[95% CI {pct(s.ci_low)}, {pct(s.ci_high)}], control {pct(s.control_headline)}")
        print("  " + ", ".join(f"{g} {pct(a)}" for g, a in s.severity.items())
              + f", oov {pct(s.oov_rate)}")
    cost = load_cost(config.RUNS_DIR / name / COST_FILE)
    print(f"\ncost: {cost.params:,} params, {cost.macs:,} MACs ({cost.note})")


if __name__ == "__main__":
    main()
