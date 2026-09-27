#!/usr/bin/env python
"""
Train one pretrained speech backbone on TORGO for one seed (step 2): the
control stage, the controls-only run, then the dysarthric LOSO folds.

    python scripts/finetune_ssl.py --backbone hubert-large --seed 0
    python scripts/finetune_ssl.py --backbone hubert-large --seed 0 --resume
    python scripts/finetune_ssl.py --backbone distilhubert --seed 0 --smoke

Writes runs/<backbone>/seed<k>/{controls.pt, fold<i>_<speaker>.pt, eval/},
runs/<backbone>-controls/seed<k>/eval/ and a cost.json beside each.
Backbones download from Hugging Face; set HF_ENDPOINT to use a mirror.
After a crash, --resume keeps controls.pt and every finished fold and trains
only the rest; without it, an unfinished seed is refused.
"""

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import config
from src.data.segments import load_torgo_samples
from src.eval.constants import ARRAY_MIC, CONTROL_SPEAKERS
from src.eval.schema import Run
from src.model.backbones import BACKBONES
from src.training.device import pick_device
from src.training.ssl_finetune import SslJob, run_ssl_finetuning

SMOKE_DIR = "smoke"  # runs/smoke/...: never mistaken for, or skipped as, a real run


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--backbone", required=True, choices=sorted(BACKBONES))
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default=None, help="default: cuda, then mps, then cpu")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--smoke", action="store_true",
                        help="1 epoch per stage, written under runs/smoke/ (a pipeline check)")
    parser.add_argument("--resume", action="store_true",
                        help="continue an unfinished seed: keep controls.pt and finished folds")
    return parser.parse_args(argv)


def make_job(args: argparse.Namespace, runs_dir: Path, noise_dir: Path, cache_dir: Path,
             device: torch.device) -> SslJob:
    job = SslJob(backbone=BACKBONES[args.backbone], seed=args.seed, runs_dir=Path(runs_dir),
                 noise_dir=Path(noise_dir), device=device, cache_dir=Path(cache_dir),
                 num_workers=args.num_workers, resume=args.resume)
    if not args.smoke:
        return job
    return replace(job, runs_dir=Path(runs_dir) / SMOKE_DIR,
                   control_stages=tuple(replace(s, epochs=1) for s in job.control_stages),
                   dysarthric_stage=replace(job.dysarthric_stage, epochs=1))


def check_control_speakers(samples: pd.DataFrame) -> None:
    """A real run trains its control stage on all 7 TORGO control speakers."""
    found = tuple(sorted(samples.loc[~samples["is_dysarthric"], "speaker_id"].unique()))
    if found != CONTROL_SPEAKERS:
        raise ValueError(f"expected the control speakers {CONTROL_SPEAKERS}, got {found}; "
                         "is data/raw/TORGO complete (bash scripts/download_torgo.sh)?")


def array_mic_summary(run: Run) -> str:
    """Per-speaker and speaker-averaged accuracy on the array mic (the headline mic)."""
    array = run.predictions[run.predictions["mic"] == ARRAY_MIC]
    per_speaker = (array["pred"] == array["label"]).groupby(array["speaker_id"]).mean()
    return (f"{run.model}, seed {run.seed}\n{per_speaker.round(3).to_string()}\n"
            f"Array mic, speaker-averaged: {per_speaker.mean():.4f}")


def main(argv=None) -> None:
    args = parse_args(argv)
    job = make_job(args, config.RUNS_DIR, config.NOISE_DIR, config.MODEL_CACHE_DIR,
                   pick_device(args.device))
    print(f"{job.backbone.name} ({job.backbone.hf_id}), seed {job.seed}, on {job.device}")
    samples = load_torgo_samples(config).samples
    check_control_speakers(samples)
    loso, controls = run_ssl_finetuning(job, samples)
    for run in (controls, loso):
        print(array_mic_summary(run))
    print(f"-> {job.out_dir / 'eval'} and {job.controls_out_dir / 'eval'}")


if __name__ == "__main__":
    main()
