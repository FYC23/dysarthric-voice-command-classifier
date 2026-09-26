"""
Stages 2-3 of BC-ResNet training on TORGO. Stage 2: the control speakers,
with a fresh 20-class head (head warmup, then the whole network); it runs once
and every fold starts from it. Stage 3: the dysarthric fine-tune once per LOSO
fold (evaluated) and once on all 8 speakers (the deploy model, not evaluated).
Hyperparameters are fixed (src/training/bcresnet_recipe.py); last epoch kept.
Design: docs/superpowers/specs/2026-09-26-bcresnet-training-design.md
"""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence, Tuple

import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.config import Config
from src.data.dataset import TORGOCommandDataset, collate_fn
from src.data.noise import NoiseBank
from src.eval.cost import CostProfile, profile_model
from src.eval.io import load_run
from src.eval.schema import Run
from src.model.bcresnet import N_MELS, BCResNet, load_pretrained_body
from src.model.frontend import LogMel, spec_augment_params
from src.training.bcresnet_loop import predict, run_stage
from src.training.bcresnet_recipe import (
    CONTROL_FINETUNE, CONTROL_HEAD_WARMUP, DYSARTHRIC_FINETUNE, SAMPLE_RATE, WINDOW_FRAMES,
    WINDOW_S, WINDOW_SAMPLES, SgdStage, run_name,
)
from src.training.loso import (
    balanced_class_weights, fold_predictions, save_loso_run, set_seed, split_fold,
)

CONTROLS_CHECKPOINT = "controls.pt"
DEPLOY_CHECKPOINT = "deploy.pt"
EVAL_DIR = "eval"
EVAL_BATCH_SIZE = 64
TORGO_CLASSES: Tuple[str, ...] = tuple(sorted(Config.TARGET_COMMANDS))
LABEL2ID = {word: i for i, word in enumerate(TORGO_CLASSES)}
ID2LABEL = dict(enumerate(TORGO_CLASSES))


@dataclass(frozen=True)
class FinetuneJob:
    tau: float
    seed: int
    out_dir: Path
    pretrained: Path
    noise_dir: Path
    device: torch.device
    control_stages: Tuple[SgdStage, ...] = (CONTROL_HEAD_WARMUP, CONTROL_FINETUNE)
    dysarthric_stage: SgdStage = DYSARTHRIC_FINETUNE
    num_workers: int = 2


def bcresnet_cost(tau: float) -> CostProfile:
    """Params and MACs of the 20-class model on one 2 s window."""
    return profile_model(BCResNet(tau, len(TORGO_CLASSES)),
                         torch.zeros(1, 1, N_MELS, WINDOW_FRAMES), WINDOW_S,
                         note="log-Mel front end not counted")


def _with_label_ids(samples: pd.DataFrame) -> pd.DataFrame:
    unknown = sorted(set(samples["label"]) - set(TORGO_CLASSES))
    if unknown:
        raise ValueError(f"labels outside the 20 commands: {unknown}")
    return samples.assign(label_id=samples["label"].map(LABEL2ID).astype(int))


def _load_pretrained(job: FinetuneJob) -> BCResNet:
    path = Path(job.pretrained)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run scripts/pretrain_bcresnet.py "
                                f"--tau {job.tau:g} --seed {job.seed} first")
    checkpoint = torch.load(path, map_location="cpu")
    if float(checkpoint["tau"]) != float(job.tau):
        raise ValueError(f"{path} is BC-ResNet tau={checkpoint['tau']:g}, "
                         f"not tau={job.tau:g}")
    model = BCResNet(job.tau, len(TORGO_CLASSES))
    load_pretrained_body(model, checkpoint["model_state_dict"])
    return model.to(job.device)


def _save(model: BCResNet, path: Path, job: FinetuneJob, stages: Sequence[SgdStage]) -> None:
    """Weights plus what produced them: width, seed, classes, source and every stage run."""
    torch.save({"model_state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                "tau": float(job.tau), "seed": job.seed, "classes": list(TORGO_CLASSES),
                "pretrained": str(job.pretrained), "stages": [asdict(s) for s in stages]}, path)


def _load(path: Path, job: FinetuneJob) -> BCResNet:
    model = BCResNet(job.tau, len(TORGO_CLASSES))
    model.load_state_dict(torch.load(path, map_location="cpu")["model_state_dict"])
    return model.to(job.device)


def _train(model: BCResNet, df: pd.DataFrame, stages: Sequence[SgdStage], job: FinetuneJob,
           noise: NoiseBank, generator: torch.Generator) -> None:
    """Train in place on `df` through `stages`, TORGO recipe + SpecAugment."""
    logmel = LogMel().to(job.device)
    spec = spec_augment_params(job.tau, WINDOW_S)
    weights = balanced_class_weights(df["label_id"], len(TORGO_CLASSES)).to(job.device)
    dataset = TORGOCommandDataset(df, None, Config, max_length=WINDOW_SAMPLES,
                                  target_sr=SAMPLE_RATE, augment=True, noise=noise)
    for stage in stages:
        loader = DataLoader(dataset, batch_size=stage.batch_size, shuffle=True,
                            drop_last=stage.drop_last, num_workers=job.num_workers,
                            collate_fn=collate_fn, generator=generator)
        history = run_stage(model, stage, loader, logmel, spec, job.device, generator,
                            weights if stage.class_weighted else None)
        print(f"  {len(history)} epochs, last: loss {history[-1]['loss']:.4f} "
              f"acc {history[-1]['acc']:.4f}")


def _predict_fold(model: BCResNet, test_df: pd.DataFrame, job: FinetuneJob) -> pd.DataFrame:
    dataset = TORGOCommandDataset(test_df, None, Config, max_length=WINDOW_SAMPLES,
                                  target_sr=SAMPLE_RATE, augment=False)
    loader = DataLoader(dataset, batch_size=EVAL_BATCH_SIZE, shuffle=False,
                        num_workers=job.num_workers, collate_fn=collate_fn)
    preds, labels = predict(model, loader, LogMel().to(job.device), job.device)
    return fold_predictions(test_df, preds, labels, ID2LABEL)


def run_finetuning(job: FinetuneJob, samples: pd.DataFrame) -> Run:
    set_seed(job.seed)
    out_dir = Path(job.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = _with_label_ids(samples)
    controls, dysarthric = df[~df["is_dysarthric"]], df[df["is_dysarthric"]]
    noise = NoiseBank.from_dir(job.noise_dir, SAMPLE_RATE)
    generator = torch.Generator().manual_seed(job.seed)

    print(f"Stage 2: {controls['speaker_id'].nunique()} control speakers, {len(controls)} clips")
    model = _load_pretrained(job)
    _train(model, controls, job.control_stages, job, noise, generator)
    _save(model, out_dir / CONTROLS_CHECKPOINT, job, job.control_stages)
    dysarthric_stages = (*job.control_stages, job.dysarthric_stage)

    frames, fold_train = [], {}
    for i, speaker in enumerate(sorted(dysarthric["speaker_id"].unique()), start=1):
        train_df, test_df = split_fold(dysarthric, speaker)
        print(f"Stage 3, fold {i} (hold out {speaker}): {len(train_df)} training clips")
        model = _load(out_dir / CONTROLS_CHECKPOINT, job)
        _train(model, train_df, (job.dysarthric_stage,), job, noise, generator)
        _save(model, out_dir / f"fold{i}_{speaker}.pt", job, dysarthric_stages)
        frames.append(_predict_fold(model, test_df, job))
        fold_train[speaker] = frozenset(train_df["speaker_id"])

    print(f"Deploy model: all {dysarthric['speaker_id'].nunique()} dysarthric speakers")
    model = _load(out_dir / CONTROLS_CHECKPOINT, job)
    _train(model, dysarthric, (job.dysarthric_stage,), job, noise, generator)
    _save(model, out_dir / DEPLOY_CHECKPOINT, job, dysarthric_stages)

    eval_dir = save_loso_run(frames, fold_train, sorted(controls["speaker_id"].unique()),
                             job.seed, run_name(job.tau), out_dir / EVAL_DIR)
    return load_run(eval_dir)
