#!/usr/bin/env python
"""
Stage 1 of BC-ResNet training: pretrain on Speech Commands v0.02 (35 words +
silence, 2 s window, the paper's recipe). One invocation per width and seed.

    python scripts/pretrain_bcresnet.py --tau 8 --seed 0
    python scripts/pretrain_bcresnet.py --tau 8 --seed 0 --resume
    python scripts/pretrain_bcresnet.py --tau 1 --seed 0 --epochs 1 --out-dir runs/timing/bcresnet-1

Writes runs/bcresnet-<tau>/seed<k>/pretrain.pt (best validation epoch),
pretrain_last.pt (resume point) and pretrain_metrics.json.
"""

import argparse
import sys
from dataclasses import replace
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import config
from src.training.bcresnet_recipe import PRETRAIN, TAUS, SgdStage, pick_device, seed_dir
from src.training.pretrain import PretrainJob, run_pretraining


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--tau", type=float, required=True, choices=TAUS)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=None,
                        help="shorten the paper's 200 epochs, e.g. 1 to time an epoch "
                             "(needs --out-dir)")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="default: runs/bcresnet-<tau>/seed<seed>")
    parser.add_argument("--device", default=None, help="default: cuda, then mps, then cpu")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--resume", action="store_true",
                        help="continue from pretrain_last.pt in the output directory")
    args = parser.parse_args(argv)
    if args.epochs is not None and args.out_dir is None:
        parser.error("--epochs needs --out-dir, so a shortened run never replaces "
                     "the pretrain.pt that stage 2 uses")
    if args.epochs is not None and args.epochs < 1:
        parser.error("--epochs must be at least 1")
    return args


def stage_for(epochs: Optional[int]) -> SgdStage:
    if epochs is None:
        return PRETRAIN
    return replace(PRETRAIN, epochs=epochs,
                   warmup_epochs=min(PRETRAIN.warmup_epochs, epochs))


def out_dir_for(args: argparse.Namespace, runs_dir: Path) -> Path:
    return args.out_dir or seed_dir(runs_dir, args.tau, args.seed)


def main(argv=None) -> None:
    args = parse_args(argv)
    job = PretrainJob(
        tau=args.tau, seed=args.seed, out_dir=out_dir_for(args, config.RUNS_DIR),
        sc_root=config.SPEECH_COMMANDS_ROOT, cache_dir=config.CACHE_DIR,
        noise_dir=config.NOISE_DIR, device=pick_device(args.device),
        stage=stage_for(args.epochs), num_workers=args.num_workers, resume=args.resume,
    )
    print(f"BC-ResNet-{args.tau:g}, seed {args.seed}, {job.stage.epochs} epochs on "
          f"{job.device} -> {job.out_dir}")
    metrics = run_pretraining(job)
    print(f"Best validation {metrics['best_val_acc']:.4f} at epoch {metrics['best_epoch']}; "
          f"test {metrics['test_acc']:.4f}")


if __name__ == "__main__":
    main()
