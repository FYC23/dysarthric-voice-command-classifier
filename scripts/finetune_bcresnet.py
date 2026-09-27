#!/usr/bin/env python
"""
Stages 2-3 of BC-ResNet training on TORGO: control speakers, then the
dysarthric fine-tune per LOSO fold and for deployment. One invocation per
width and seed, after scripts/pretrain_bcresnet.py.

    python scripts/finetune_bcresnet.py --tau 8 --seed 0
    python scripts/finetune_bcresnet.py --tau 8 --seed 1 --pretrained runs/bcresnet-8/seed0/pretrain.pt

Writes runs/bcresnet-<tau>/seed<k>/{controls.pt, fold<i>_<speaker>.pt,
deploy.pt, eval/} and runs/bcresnet-<tau>/cost.json.
"""

import argparse
import sys
from pathlib import Path
from typing import Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import config
from src.data.segments import load_torgo_samples
from src.eval.constants import ARRAY_MIC
from src.eval.cost import COST_FILE, save_cost
from src.training.bcresnet_recipe import TAUS, pick_device, run_name, seed_dir
from src.training.finetune import FinetuneJob, bcresnet_cost, run_finetuning
from src.training.pretrain import BEST_CHECKPOINT


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--tau", type=float, required=True, choices=TAUS)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--pretrained", type=Path, default=None,
                        help="stage 1 checkpoint (default: this seed's pretrain.pt); "
                             "point several seeds at one file to share a pretraining run")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="default: runs/bcresnet-<tau>/seed<seed>")
    parser.add_argument("--device", default=None, help="default: cuda, then mps, then cpu")
    parser.add_argument("--num-workers", type=int, default=2)
    return parser.parse_args(argv)


def paths_for(args: argparse.Namespace, runs_dir: Path) -> Tuple[Path, Path]:
    """(output directory, pretrained checkpoint)."""
    default_dir = seed_dir(runs_dir, args.tau, args.seed)
    return args.out_dir or default_dir, args.pretrained or default_dir / BEST_CHECKPOINT


def main(argv=None) -> None:
    args = parse_args(argv)
    out_dir, pretrained = paths_for(args, config.RUNS_DIR)
    job = FinetuneJob(tau=args.tau, seed=args.seed, out_dir=out_dir, pretrained=pretrained,
                      noise_dir=config.NOISE_DIR, device=pick_device(args.device),
                      num_workers=args.num_workers)
    save_cost(bcresnet_cost(args.tau), config.RUNS_DIR / run_name(args.tau) / COST_FILE)
    samples = load_torgo_samples(config).samples
    run = run_finetuning(job, samples)
    array = run.predictions[run.predictions["mic"] == ARRAY_MIC]
    per_speaker = (array["pred"] == array["label"]).groupby(array["speaker_id"]).mean()
    print(per_speaker.round(3).to_string())
    print(f"Array mic, speaker-averaged: {per_speaker.mean():.4f}  -> {out_dir / 'eval'}")


if __name__ == "__main__":
    main()
